from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypeVar

import flax
import jax.random as jrd

ActionType = TypeVar("ActionType")
ObsType = TypeVar("ObsType")
Params = flax.core.FrozenDict

from rl.buffer import Buffer


class EnvProcs(Enum):
    ONE = "one"
    MANY = "many"


class EnvType(Enum):
    SINGLE = "single"
    PARALLEL = "parallel"


class Seeded:
    def __init__(self, seed: int):
        self.key = jrd.PRNGKey(seed)

    def nextkey(self):
        self.key, _k = jrd.split(self.key)
        return _k


class Base(ABC, Seeded):
    def __init__(self, seed: int):
        Seeded.__init__(self, seed)

    @abstractmethod
    def select_action(self, observation: ObsType) -> ActionType:
        ...

    @abstractmethod
    def explore(self, observation: ObsType) -> ActionType:
        ...

    @abstractmethod
    def update(self, buffer: Buffer) -> None:
        ...

    @abstractmethod
    def train(self, env: Any, n_env_steps: int) -> None:
        ...
