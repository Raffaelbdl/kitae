from datetime import datetime
import functools
import os
from pathlib import Path
from typing import Any, Callable, NamedTuple


import jax
import jax.numpy as jnp
from jrd_extensions import Seeded


from rl_tools.base import PipelineAgent
from rl_tools.types import Params

from rl_tools.algos.pipeline import PipelineModule
from rl_tools.algos.pipeline import process_experience_pipeline_factory
from rl_tools.algos.pipeline import update_pipeline
from rl_tools.buffer import Experience, stack_experiences
from rl_tools.config import AlgoConfig
from rl_tools.modules.train_state import TrainState
from rl_tools.save import Saver

Factory = Callable[..., Callable]


def fn_parallel(fn: Callable) -> Callable:
    """Parallelizes a function for mutliple agents.

    Typical usage for function with args:
        - state
        - trees of structure: {"agent_0": Array, "agent_1": Array, ...}
        - hyperparameters

    The wrapped function returns a list of trees with the same structure as input.

    Warning: args must be entered in the same order as in fn to allow vmapping.
    """

    def wrapped(state: Any, *trees, **hyperparams):
        results = jax.tree_util.tree_map(
            functools.partial(fn, state, **hyperparams), *trees
        )

        # transform the structure of results
        # output = {"agent_0": [out1, out2], "agent_1": [out1, out2]}
        # -> [{"agent_1": out1, "agent_2": out1}, {"agent_1": out2, "agent_2": out2}]

        n_outputs = len(list(results.values())[0])
        outputs = [{} for _ in range(n_outputs)]

        for key, value in results.items():
            for out_pos in range(n_outputs):
                outputs[out_pos][key] = value[out_pos]

        return outputs

    return wrapped


def explore_general_factory(
    explore_fn: Callable, vectorized: bool, parallel: bool
) -> Callable:
    """Generalizes a explore_fn to vector and parallel envs."""

    def input_fn(inputs):
        if not vectorized:
            return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), inputs)
        return inputs

    explore_fn = fn_parallel(explore_fn) if parallel else explore_fn

    def output_fn(outputs):
        if not vectorized:
            return jax.tree_map(lambda x: jnp.squeeze(x, axis=0), outputs)
        return outputs

    def general_fn(state: Any, key: jax.Array, *trees, **hyperparams):
        inputs = input_fn(trees)
        results = explore_fn(state, key, *inputs, **hyperparams)
        outputs = output_fn(results)
        return outputs

    return general_fn


class AlgoFactory:
    @staticmethod
    def intialize(
        self: PipelineAgent,
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

        self.vectorized = self.config.env_cfg.n_envs > 1
        self.parallel = self.config.env_cfg.n_agents > 1

        state = train_state_factory(
            self.nextkey(),
            self.config,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
        )
        process_experience_fn = process_experience_factory(self.config)
        update_fn = update_step_factory(self.config)
        self.main_pipeline_module = PipelineModule(
            state=state,
            process_experience_fn=process_experience_fn,
            update_fn=update_fn,
        )

        self.explore_fn = explore_general_factory(
            explore_factory(self.config.algo_params),
            self.vectorized,
            self.parallel,
        )

        self.process_experience_pipeline = jax.jit(
            process_experience_pipeline_factory(
                self.vectorized, self.parallel, experience_type
            )
        )

        self.update_pipeline = update_pipeline

        self.explore_factory = explore_factory

        self.run_name = run_name
        if run_name is None:
            self.run_name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.saver = Saver(
            Path(os.path.join("./results", self.run_name)).absolute(), self
        )
