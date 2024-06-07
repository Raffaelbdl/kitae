from abc import ABC, abstractmethod
from collections import deque
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from kitae.interface import AlgoType
from kitae.buffers.buffer import Experience
from kitae.types import PRNGKeyArray


class BufferState(NamedTuple):
    buffer: Experience
    index: int
    length: int


class jBuffer(ABC):
    """Base jit-Buffer class."""

    def __init__(self, max_buffer_size: int, n_envs: int, sample_size: int):
        """Instantiates the buffer."""
        self.max_buffer_size = max_buffer_size
        self.n_envs = n_envs
        self.sample_size = sample_size

    def init(self, experience: Experience) -> BufferState:
        default_values = [
            jnp.empty((self.max_buffer_size, self.n_envs, *v.shape), v.dtype)
            for v in experience
        ]
        buffer = experience.__class__(*default_values)
        return BufferState(
            buffer, jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.int32)
        )

    def add(self, buffer_state: BufferState, experience: Experience) -> BufferState:
        buffer, i, _ = buffer_state

        # update index (only used in offpolicy buffer)
        i, length = jax.lax.cond(
            i >= self.max_buffer_size,
            lambda i: (0, self.max_buffer_size),
            lambda i: (i, i),
            i,
        )

        buffer = jax.tree_map(lambda x, y: x.at[i].set(y), buffer, experience)
        return BufferState(buffer, i + 1, length)

    @abstractmethod
    def sample(
        self, buffer_state: BufferState, key: PRNGKeyArray
    ) -> tuple[BufferState, Experience]: ...


class OnPolicyBuffer(jBuffer):
    """OnPolicy variant of the jit-buffer class."""

    def sample(
        self, buffer_state: BufferState, key: PRNGKeyArray
    ) -> tuple[BufferState, Experience]:
        sample = buffer_state.buffer
        buffer_state = BufferState(
            buffer_state.buffer,
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )
        return buffer_state, sample


class OffPolicyBuffer(jBuffer):
    """OffPolicy variant of the buffer class."""

    def sample(
        self, buffer_state: BufferState, key: PRNGKeyArray
    ) -> tuple[BufferState, Experience]:
        inds = jax.random.permutation(key, self.max_buffer_size)[: self.sample_size]
        sample = jax.tree_map(lambda x: x[inds], buffer_state.buffer)
        return buffer_state, sample


def buffer_factory(
    algo_type: AlgoType, max_buffer_size: int, n_envs: int, sample_size: int
) -> jBuffer:
    """Generates a buffer given the AlgoType."""

    if algo_type == AlgoType.ON_POLICY:
        return OnPolicyBuffer(max_buffer_size, n_envs, sample_size)
    return OffPolicyBuffer(max_buffer_size, n_envs, sample_size)
