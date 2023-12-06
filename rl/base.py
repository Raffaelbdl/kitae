from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import os
from typing import Any, Callable

import chex
from flax import struct
from flax.training.train_state import TrainState
from jrd_extensions import Seeded

from rl.buffer import Buffer
from rl.save import Saver
from rl.types import ActionType, ObsType, Params


class EnvProcs(Enum):
    ONE = "one"
    MANY = "many"


class EnvType(Enum):
    SINGLE = "single"
    PARALLEL = "parallel"


class AlgoType(Enum):
    ON_POLICY = "on_policy"
    OFF_POLICY = "off_policy"


@chex.dataclass
class Deployed:
    params: Params
    select_action: Callable = struct.field(pytree_node=False)


class Base(ABC, Seeded):
    def __init__(
        self,
        seed: int,
        config,
        train_state_factory: Callable,
        explore_factory: Callable,
        process_experience_factory: Callable,
        update_step_factory: Callable,
        *,
        rearrange_pattern: str = "b h w c -> b h w c",
        preprocess_fn: Callable = None,
        n_envs: int = 1,
        run_name: str = None,
        tabulate: bool = False,
    ):
        Seeded.__init__(self, seed)
        self.config = config

        self.rearrange_pattern = rearrange_pattern
        self.preprocess_fn = preprocess_fn
        self.n_envs = n_envs
        self.tabulate = tabulate

        self.state: TrainState = train_state_factory(
            self.nextkey(),
            self.config,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            n_envs=n_envs,
            tabulate=tabulate,
        )
        self.explore_fn: Callable = explore_factory(self.state, self.config, n_envs > 1)
        self.process_experience_fn: Callable = process_experience_factory(
            self.state, self.config, n_envs > 1
        )
        self.update_step_fn = update_step_factory(self.state, self.config)

        self.explore_factory = explore_factory

        self.run_name = run_name
        if self.run_name is None:
            self.run_name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.saver = Saver(os.path.join("./results", self.run_name), self)

    @abstractmethod
    def select_action(self, observation: ObsType) -> ActionType:
        ...

    @abstractmethod
    def explore(self, observation: ObsType) -> ActionType:
        ...

    @abstractmethod
    def should_update(self, step: int, buffer: Buffer) -> None:
        ...

    @abstractmethod
    def update(self, buffer: Buffer) -> None:
        ...

    @abstractmethod
    def train(self, env: Any, n_env_steps: int, callbacks: list) -> None:
        ...

    @abstractmethod
    def resume(self, env: Any, n_env_steps: int, callbacks: list) -> None:
        ...

    def restore(self) -> int:
        step, self.state = self.saver.restore_latest_step(self.state)
        return step

    def to_dict(self) -> dict:
        """Serialize metadata"""

        return {
            "seed": self.seed,
            "config": self.config,
            "n_envs": self.n_envs,
            "run_name": self.run_name,
            "rearrange_pattern": self.rearrange_pattern,
            "preprocess_fn": self.preprocess_fn,
            "tabulate": self.tabulate,
        }

    @classmethod
    def unserialize(cls, metadata_path: str):
        """Unserialize metadata"""
        import cloudpickle

        with open(metadata_path, "rb") as f:
            base_kwargs = cloudpickle.load(f)

        return cls(**base_kwargs)

    def deploy_agent(self, batched: bool) -> Deployed:
        return Deployed(
            params=self.state.params,
            select_action=self.explore_factory(self.state, self.config, batched),
        )
