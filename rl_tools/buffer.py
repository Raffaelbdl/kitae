from collections import namedtuple, deque

import chex
import jax
import jax.numpy as jnp
import numpy as np

from rl_tools.interface import AlgoType, IBuffer

Experience = namedtuple(
    "Experience",
    ["observation", "action", "reward", "done", "next_observation", "log_prob"],
)


def array_of_name(experiences: list[Experience], name: str) -> np.ndarray:
    return np.array([e.__getattribute__(name) for e in experiences])


def stack_experiences(experiences: list[Experience]) -> Experience:
    return jax.tree_map(lambda *xs: jnp.stack(xs, axis=0), *experiences)


def batchify(data: tuple[jax.Array, ...], batch_size: int) -> tuple[jax.Array, ...]:
    """Creates batches from a tuple of Array.

    Typical usage:
        batches = batchify(data, batch_size)
        for batch in zip(*batches):
            ...

    Args:
        data: A tuple of Array of shape [T, ...]
        batch_size: A int that represents the length of each batches
    Returns:
        batches: A tuple of Array of shape [T // batch_size, batch_size, ...]
    """
    n_elements = data[0].shape[0]
    n_batches = n_elements // batch_size  # will truncate last elements

    chex.assert_scalar_positive(n_batches)

    n = n_batches * batch_size
    return jax.tree_util.tree_map(
        lambda x: x[:n].reshape((n_batches, batch_size) + x.shape[1:]), data
    )


def batchify_and_randomize(
    key: jax.Array, data: tuple[jax.Array, ...], batch_size: int
) -> tuple[jax.Array, ...]:
    """Creates batches from a tuple of Array.

    Typical usage:
        batches = batchify_and_randomize(key, data, batch_size)
        for batch in zip(*batches):
            ...

    Args:
        key: An Array for randomness
        data: A tuple of Array of shape [T, ...]
        batch_size: A int that represents the length of each batches
    Returns:
        batches: A tuple of Array of shape [T // batch_size, batch_size, ...]
    """
    inds = jax.random.permutation(key, data[0].shape[0])
    data = jax.tree_util.tree_map(lambda x: x[inds], data)

    return batchify(data, batch_size)


class Buffer(IBuffer):
    def __init__(self, seed: int, max_buffer_size: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_buffer_size = max_buffer_size

        self.buffer: list | deque = None

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)


class OffPolicyBuffer(Buffer):
    def __init__(self, seed: int, max_buffer_size: int = 0) -> None:
        Buffer.__init__(self, seed=seed, max_buffer_size=max_buffer_size)

        if max_buffer_size > 0:
            self.buffer: deque[Experience] = deque(maxlen=max_buffer_size)
        else:
            self.buffer: list[Experience] = []

    def sample(self, batch_size: int) -> list[Experience]:
        indices = self.rng.permutation(len(self.buffer))[:batch_size]
        return [self.buffer[i] for i in indices]


class OnPolicyBuffer(Buffer):
    def __init__(self, seed: int, max_buffer_size: int = 0) -> None:
        Buffer.__init__(self, seed=seed, max_buffer_size=max_buffer_size)

        self.buffer = []

    def sample(self, batch_size: int) -> list[Experience]:
        sample, self.buffer = self.buffer, []
        return sample


def buffer_factory(seed: int, algo_type: AlgoType, max_buffer_size: int) -> Buffer:
    if algo_type == AlgoType.ON_POLICY:
        return OnPolicyBuffer(seed, max_buffer_size)
    return OffPolicyBuffer(seed, max_buffer_size)
