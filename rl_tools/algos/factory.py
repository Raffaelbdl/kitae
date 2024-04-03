import functools
from typing import Any, Callable, NamedTuple

from flax import struct
import jax
import jax.numpy as jnp

from rl_tools.buffer import jax_stack_experiences


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


class ExperienceTransform(struct.PyTreeNode):
    """Experience Transform PyTreeNode.

    This class encapsulates both a state and a process_experience function.
    """

    process_experience_fn: Callable = struct.field(pytree_node=False)
    state: struct.PyTreeNode = struct.field(pytree_node=True)


def process_experience_pipeline_factory(
    vectorized: bool,
    parallel: bool,
    experience_type: NamedTuple,
) -> Callable:
    """Wrap ExperienceTransform for vectorized and parallel envs.

    If vectorized:
        inputs: tuple with elements of shape [T, N_envs, ...]
        outputs: tuple with elements of shape [T * N_envs, ...]

    If parallel:
        inputs: tuple with elements of shape [{str: [T, ...]}]
        outputs: tuple with elements of shape [T * N_agents, ...]

    If vectorized and parallel:
        inputs: tuple with elements of shape [{str: [T, N_envs, ...]}]
        outputs: tuple with elements of shape [T * N_agents * N_envs, ...]
    """

    def process_experience(
        experience_transforms: ExperienceTransform | list[ExperienceTransform],
        key: jax.Array,
        experiences: tuple | list[tuple],
    ):
        if not isinstance(experience_transforms, list):
            experience_transforms = [experience_transforms]

        def process_experience_pipeline(
            key: jax.Array, *_experiences: tuple[jax.Array, ...]
        ) -> tuple[jax.Array, ...]:
            _experiences = experience_type(*_experiences)

            for transform in experience_transforms:
                key, _k = jax.random.split(key)
                _experiences = transform.process_experience_fn(
                    transform.state, _k, _experiences
                )

            return _experiences

        if isinstance(experiences, list):
            experiences = jax_stack_experiences(experiences)

        if parallel and vectorized:
            keys = {}
            for agent, value in experiences[0].items():
                # vectorized => shape [T, n_envs, ...]
                _keys = jax.random.split(key, value.shape[1] + 1)
                key, keys[agent] = _keys[0], _keys[1:]

            in_axes = (0,) + (1,) * len(experiences)
            processed = jax.tree_map(
                jax.vmap(process_experience_pipeline, in_axes=in_axes, out_axes=1),
                keys,
                *experiences,
            )

            def concat_and_reshape(*x: tuple[jax.Array, ...]) -> jax.Array:
                # x n_agents * (T, n_envs, ...)
                # concat > (T, n_agents * n_envs, ...)
                # reshape => (T * n_agents * n_envs, ...)
                out = jnp.concatenate(x, axis=1)
                return jnp.reshape(out, (-1, *out.shape[2:]))

            return jax.tree_map(concat_and_reshape, *zip(processed.values()))[0]

        if vectorized:
            keys = jax.random.split(key, experiences[0].shape[1])
            in_axes = (0,) + (1,) * len(experiences)
            processed = jax.vmap(
                process_experience_pipeline, in_axes=in_axes, out_axes=1
            )(keys, *experiences)
            return jax.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), processed)

        if parallel:
            keys = {}
            for agent, value in experiences[0].items():
                key, keys[agent] = jax.random.split(key, 2)

            processed = jax.tree_map(process_experience_pipeline, keys, *experiences)

            def stack_and_reshape(*x: tuple[jax.Array, ...]) -> jax.Array:
                # x n_agents * (T,  ...)
                # stack > (T, n_agents, ...)
                # reshape => (T * n_agents, ...)
                out = jnp.stack(x, axis=1)
                return jnp.reshape(out, (-1, *out.shape[2:]))

            return jax.tree_map(stack_and_reshape, *zip(processed.values()))[0]

        return process_experience_pipeline(key, *experiences)

    return process_experience
