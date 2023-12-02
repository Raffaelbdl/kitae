from abc import ABC, abstractmethod
from collections import namedtuple, deque

import jax
import jax.numpy as jnp
import numpy as np


OffPolicyExp = namedtuple(
    "OffPolicyExp", ["observation", "action", "reward", "done", "next_observation"]
)
OnPolicyExp = namedtuple(
    "OnPolicyExp",
    ["observation", "action", "reward", "done", "next_observation", "log_prob"],
)
Experience = OffPolicyExp | OnPolicyExp


def array_of_name(experiences: list[Experience], name: str) -> np.ndarray:
    return np.array([e.__getattribute__(name) for e in experiences])


def stack_experiences(experiences: list[Experience]) -> Experience:
    return jax.tree_map(lambda *xs: jnp.stack(xs, axis=0), *experiences)


class Buffer(ABC):
    def __init__(self, seed: int, max_buffer_size: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_buffer_size = max_buffer_size

        self.buffer: list | deque = None

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    @abstractmethod
    def sample(self, batch_size: int) -> list[Experience]:
        ...


class OffPolicyBuffer(Buffer):
    def __init__(self, seed: int, max_buffer_size: int = 0) -> None:
        Buffer.__init__(self, seed=seed, max_buffer_size=max_buffer_size)

        if max_buffer_size > 0:
            self.buffer: deque[OffPolicyExp] = deque(maxlen=max_buffer_size)
        else:
            self.buffer: list[OffPolicyExp] = []

    def sample(self, batch_size: int) -> list[OffPolicyExp]:
        indices = self.rng.permutation(len(self.buffer))[:batch_size]
        return [self.buffer[i] for i in indices]


class OnPolicyBuffer(Buffer):
    def __init__(self, seed: int, max_buffer_size: int = 0) -> None:
        Buffer.__init__(self, seed=seed, max_buffer_size=max_buffer_size)

        self.buffer = []

    def sample(self) -> list[OnPolicyExp]:
        sample, self.buffer = self.buffer, []
        return sample
