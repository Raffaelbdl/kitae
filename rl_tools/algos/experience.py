from typing import Callable, NamedTuple

from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp

from rl_tools.buffer import stack_experiences


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
    """Wrap multiple ExperienceTransform for vectorized and parallel envs.

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
        experience_transforms: list[ExperienceTransform],
        key: jax.Array,
        experiences: tuple | list[tuple],
    ):
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
            experiences = stack_experiences(experiences)

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


def process_experience_factory(
    vectorized: bool,
    parallel: bool,
    experience_type: NamedTuple,
) -> Callable:
    """Wraps a single ExperienceTransform for vectorized and parallel envs.

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
        experience_transform: ExperienceTransform,
        key: jax.Array,
        experiences: tuple | list[tuple],
    ):
        return process_experience_pipeline_factory(
            vectorized=vectorized, parallel=parallel, experience_type=experience_type
        )([experience_transform], key, experiences)

    return process_experience
