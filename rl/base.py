from abc import ABC, abstractmethod
from typing import TypeVar

import flax
import jax.random as jrd

ActionType = TypeVar("ActionType")
ObsType = TypeVar("ObsType")
Params = flax.core.FrozenDict

from rl.buffer import Buffer


class Base(ABC):
    def __init__(self, seed: int):
        self.key = jrd.key(seed)

    def nextkey(self):
        self.key, _k = jrd.split(self.key)
        return _k

    @abstractmethod
    def select_action(self, observation: ObsType) -> ActionType:
        ...

    @abstractmethod
    def explore(self, observation: ObsType) -> ActionType:
        ...

    @abstractmethod
    def update(self, buffer: Buffer) -> None:
        ...
