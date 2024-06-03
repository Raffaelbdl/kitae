from typing import Callable, NamedTuple

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp

from kitae.types import ExperienceTuple, LossDict, PRNGKeyArray


BatchifyFn = Callable[[NamedTuple], NamedTuple]
UpdateStepFn = Callable[
    [TrainState, PRNGKeyArray, ExperienceTuple],
    tuple[TrainState, LossDict],
]


def update_epoch(
    key: jax.Array,
    state: TrainState,
    experience: ExperienceTuple,
    batchify_fn: Callable,
    update_batch_fn: UpdateStepFn,
    *,
    experience_type: type[ExperienceTuple],
    batch_size: int,
) -> tuple[TrainState, dict]:
    key, _key = jax.random.split(key)
    batches = batchify_fn(_key, experience, batch_size)

    def _update_batch(
        carry: tuple[PRNGKeyArray, TrainState], batch: tuple[jax.Array, ...]
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
