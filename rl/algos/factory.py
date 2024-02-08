from datetime import datetime
import functools
import os
from pathlib import Path
from typing import Callable, NamedTuple


import jax
import jax.numpy as jnp
from jrd_extensions import Seeded


from rl.base import Base, DeployedJit
from rl.types import Params

from rl.algos.pipelines.experience_pipeline import (
    ExperienceTransform,
    process_experience_pipeline_factory,
)
from rl.algos.pipelines.update_pipeline import update_pipeline_fn
from rl.buffer import Experience, stack_experiences
from rl.config import AlgoConfig
from rl.modules.train_state import TrainState
from rl.save import Saver

Factory = Callable[..., Callable]


def fn_parallel(fn: Callable) -> Callable:
    """args must be entered in the same order as in fn to allow vmapping"""

    def _fn(params: Params, *trees, **hyperparams):
        res = jax.tree_util.tree_map(
            functools.partial(fn, params, **hyperparams), *trees
        )

        n_outputs = len(list(res.values())[0])
        outputs = [{} for _ in range(n_outputs)]
        for a, v in res.items():
            for i in range(len(outputs)):
                outputs[i][a] = v[i]
        return outputs

    return _fn


# TODO works but not for the good reasons
# the first input is no longer params !!
def explore_general_factory(
    explore_fn: Callable, batched: bool, parallel: bool
) -> Callable:
    def input_fn(inputs):
        if not batched:
            return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), inputs)
        return inputs

    explore_fn = fn_parallel(explore_fn) if parallel else explore_fn

    def output_fn(outputs):
        if not batched:
            return jax.tree_map(lambda x: jnp.squeeze(x, axis=0), outputs)
        return outputs

    def fn(params: Params, key, *trees, **hyperparams):
        inputs = input_fn(trees)
        results = explore_fn(params, key, *inputs, **hyperparams)
        outputs = output_fn(results)
        return outputs

    return jax.jit(fn)


def process_experience_general_factory(
    process_experience_fn: Callable,
    vectorized: bool,
    parallel: bool,
    experience_type: NamedTuple = Experience,
):
    def fn(state: TrainState, key: jax.Array, experience: experience_type):
        experience = stack_experiences(experience)

        def _process_experience_fn(key, *experience):
            return process_experience_fn(state, key, experience_type(*experience))

        if parallel and vectorized:
            keys = {}
            for agent, obs in experience.observation.items():
                new_keys = jax.random.split(key, obs.shape[1] + 1)
                # swap axes for vmapping
                key, keys[agent] = new_keys[0], new_keys[1:].swapaxes(0, 1)

            # TODO make a test for this section
            # especially the part where a new tuple of length 1 is created
            outputs = jax.tree_map(
                jax.vmap(_process_experience_fn, in_axes=1, out_axes=1),
                keys,
                *experience,
            )

            def concat_and_reshape(*x):
                out = jnp.concatenate(x, axis=1)
                return jnp.reshape(out, (-1, *out.shape[2:]))

            return jax.tree_map(concat_and_reshape, *zip(outputs.values()))[0]

        elif parallel:
            keys = {}
            for agent in experience.observation.keys():
                key, _k = jax.random.split(key, 2)
                keys[agent] = _k

            # TODO make a test for this section
            # especially the part where a new tuple of length 1 is created
            outputs = jax.tree_map(_process_experience_fn, keys, *experience)

            def stack_and_reshape(*x):
                out = jnp.stack(x, axis=1)
                return jnp.reshape(out, (-1, *out.shape[2:]))

            return jax.tree_map(stack_and_reshape, *zip(outputs.values()))[0]

        elif vectorized:
            keys = jax.random.split(key, experience.observation.shape[1]).swapaxes(0, 1)
            outputs = jax.vmap(_process_experience_fn, in_axes=1, out_axes=1)(
                keys, *experience
            )
            outputs = jax.tree_map(
                lambda x: jnp.reshape(x, (-1, *x.shape[2:])), outputs
            )
            return outputs

        else:
            return process_experience_fn(state, key, experience)

    return jax.jit(fn)


class AlgoFactory:
    @staticmethod
    def intialize(
        self: Base,
        config: AlgoConfig,
        train_state_factory: Callable[..., TrainState],
        explore_factory: Factory,
        process_experience_factory: Factory,
        update_step_factory: Factory,
        *,
        rearrange_pattern: str = "b h w c -> b h w c",
        preprocess_fn: Callable = None,
        run_name: str | None = None,
        tabulate: bool = False,
        experience_type: NamedTuple = Experience,
    ) -> None:
        Seeded.__init__(self, config.seed)
        self.config = config

        self.rearrange_pattern = rearrange_pattern
        self.preprocess_fn = preprocess_fn
        self.tabulate = tabulate

        self.vectorized = self.config.env_cfg.n_envs > 1
        self.parallel = self.config.env_cfg.n_agents > 1

        self.state = train_state_factory(
            self.nextkey(),
            self.config,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
        )

        self.explore_fn = explore_general_factory(
            explore_factory(self.state, self.config.algo_params),
            self.vectorized,
            self.parallel,
        )

        self.process_experience_pipeline = jax.jit(
            process_experience_pipeline_factory(
                self.vectorized, self.parallel, experience_type
            )
        )
        self.process_experience_fn = process_experience_factory(self.state, self.config)

        self.update_pipeline_fn = update_pipeline_fn
        self.update_step_fn = update_step_factory(self.state, self.config)

        self.explore_factory = explore_factory

        self.run_name = run_name
        if self.run_name is None:
            self.run_name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.saver = Saver(
            Path(os.path.join("./results", self.run_name)).absolute(), self
        )

    @staticmethod
    def deploy_agent(self: Base, batched: bool) -> DeployedJit:
        return DeployedJit(
            params=self.state.policy_state.params,
            select_action=explore_general_factory(
                self.explore_factory(self.state, self.config),
                batched=batched,
                parallel=False,
            ),
        )
