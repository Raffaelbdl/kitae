from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from kitae.modules.pytree import AgentPyTree
from kitae.types import LossDict, PRNGKeyArray, ProcessedExperienceTuple


BatchifyFn = Callable[[NamedTuple], NamedTuple]
UpdateStepFn = Callable[
    [AgentPyTree, PRNGKeyArray, ProcessedExperienceTuple],
    tuple[AgentPyTree, LossDict],
]


def update_epoch(
    key: PRNGKeyArray,
    state: AgentPyTree,
    experience: ProcessedExperienceTuple,
    batchify_fn: Callable,
    update_batch_fn: UpdateStepFn,
    *,
    experience_type: type[ProcessedExperienceTuple],
    batch_size: int,
) -> tuple[AgentPyTree, LossDict]:
    """Updates a state in a single epoch.

    This function uses `jax.lax.scan` which can reduce compilation time.

    Args:
        key: A PRNGKeyArray for reproducibility.
        state: A AgentPyTree containing the agent's state.
        experience: An ExperienceTuple containing processed trajectories.
        batchify_fn: A Callable that processes the experience into batches.
        update_batch_fn: An UpdateStepFn that updates the agent's state.
        experience_type: A custom experience type.
        batch_size: An int that determines the size of a batch.

    Returns:
        An updated agent's state and the corresponding loss dictionary.
    """
    key, _key = jax.random.split(key)
    batches = batchify_fn(_key, experience, batch_size)

    def _update_batch(
        carry: tuple[PRNGKeyArray, AgentPyTree], batch: tuple[jax.Array, ...]
    ):
        key, state = carry
        batch = experience_type(*batch)
        key, _key = jax.random.split(key)

        state, info = update_batch_fn(state, _key, batch)
        return (_key, state), info

    init = (key, state)
    (_, state), info = jax.lax.scan(_update_batch, init, batches)
    info = jax.tree_util.tree_map(lambda x: jnp.mean(x), info)
    return state, info
