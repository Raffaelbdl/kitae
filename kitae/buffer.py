from collections import namedtuple, deque

import chex
import jax
import jax.numpy as jnp
import numpy as np

from kitae.interface import AlgoType, IBuffer

Experience = namedtuple(
    "Experience",
    ["observation", "action", "reward", "done", "next_observation", "log_prob"],
)


def jax_stack_experiences(experiences: list[Experience]) -> Experience:
    """Stacks list of Experience into a single Experience.

    Args:
        experiences: a list of Experience to stack.

    Returns:
        An Experience of the stacked inputs.
    """
    _cls = experiences[0].__class__
    return _cls(*jax.tree_map(lambda *xs: jnp.stack(xs, axis=0), *experiences))


def numpy_stack_experiences(experiences: list[Experience]) -> Experience:
    """Stacks list of Experience into a single Experience.

    Args:
        experiences: a list of Experience to stack.

    Returns:
        An Experience of the stacked inputs.

    Todo:
        Make compatible with multi-agents environments.
    """
    _cls = experiences[0].__class__
    return _cls(*[np.stack(v, axis=0) for v in zip(*experiences)])


def batchify(data: tuple[jax.Array, ...], batch_size: int) -> tuple[jax.Array, ...]:
    """Creates batches from a tuple of Array.

    Tip:
        Typical Usage::

            batches = batchify(data, batch_size)
            for batch in zip(*batches):
                ...

    Args:
        data (tuple[jax.Array, ...]): A tuple of Array of shape [T, ...]
        batch_size (int): A int that represents the length of each batches

    Returns:
        A tuple of Array of shape [T // batch_size, batch_size, ...]

    Raises:
        AssertionError: if the batch_size is strictly greater than the number of elements
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
    """Randomizes and creates batches from a tuple of Array.

    Tip:
        Typical Usage::

            batches = batchify_and_randomize(key, data, batch_size)
            for batch in zip(*batches):
                ...

    Args:
        key (jax.Array): An Array for randomness
        data (tuple[jax.Array, ...]): A tuple of Array of shape [T, ...]
        batch_size (int): A int that represents the length of each batches

    Returns:
        A tuple of Array of shape [T // batch_size, batch_size, ...]

    Raises:
        AssertionError: if the batch_size is strictly greater than the number of elements
    """
    inds = jax.random.permutation(key, data[0].shape[0])
    data = jax.tree_util.tree_map(lambda x: x[inds], data)

    return batchify(data, batch_size)


class Buffer(IBuffer):
    """Base Buffer class.

    Attributes:
        rng (np.random.Generator): A numpy random number generator.
        max_buffer_size (int): The maximum size of the buffer.
        buffer (list | deque): A list or deque that contains the transitions.
    """

    def __init__(self, seed: int, max_buffer_size: int = 0) -> None:
        """Instantiates the buffer."""
        self.rng = np.random.default_rng(seed)
        self.max_buffer_size = max_buffer_size
        self.buffer: list | deque = None

    def __len__(self) -> int:
        """Returns the length of the buffer."""
        return len(self.buffer)

    def add(self, experience: Experience) -> None:
        """Adds a transition to the buffer."""
        self.buffer.append(experience)


class OffPolicyBuffer(Buffer):
    """OffPolicy variant of the buffer class."""

    def __init__(self, seed: int, max_buffer_size: int = 0) -> None:
        """Instantiates an OffPolicy buffer."""
        Buffer.__init__(self, seed=seed, max_buffer_size=max_buffer_size)

        if max_buffer_size > 0:
            self.buffer: deque[Experience] = deque(maxlen=max_buffer_size)
        else:
            self.buffer: list[Experience] = []

    def sample(self, batch_size: int) -> list[Experience]:
        """Samples from the OffPolicy buffer.

        Args:
            batch_size (int): the number of elements to sample.

        Returns:
            A list of transitions as tuples.
        """
        indices = self.rng.permutation(len(self.buffer))[:batch_size]
        return [self.buffer[i] for i in indices]


class OnPolicyBuffer(Buffer):
    """OnPolicy variant of the buffer class."""

    def __init__(self, seed: int, max_buffer_size: int = 0) -> None:
        """Instantiates an OnPolicy buffer."""
        Buffer.__init__(self, seed=seed, max_buffer_size=max_buffer_size)

        self.buffer = []

    def sample(self, batch_size: int = -1) -> list[Experience]:
        """Samples from the OnPolicy buffer and then empties it.

        Returns:
            A list of transitions as tuples.
        """
        sample, self.buffer = self.buffer, []
        return sample


def buffer_factory(seed: int, algo_type: AlgoType, max_buffer_size: int) -> Buffer:
    """Generates a buffer based on the AlgoType.

    Args:
        seed (int): An int for reproducibility.
        algo_type (AlgoType)
        max_buffer_size (int): The maximum size of the buffer.

    Returns:
        An empty instance of the corresponding buffer.
    """
    if algo_type == AlgoType.ON_POLICY:
        return OnPolicyBuffer(seed, max_buffer_size)
    return OffPolicyBuffer(seed, max_buffer_size)
