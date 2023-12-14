from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import os
from typing import Any, Callable

import chex
import yaml
import cloudpickle
from flax import struct
from flax.training.train_state import TrainState
from jrd_extensions import Seeded

from rl.buffer import Buffer
from rl.save import Saver
from rl.types import ActionType, ObsType, Params

from rl.algos.general_fns import (
    explore_general_factory,
    process_experience_general_factory,
)
from ml_collections import FrozenConfigDict, ConfigDict
from rl.config import AlgoConfig


class EnvProcs(Enum):
    ONE = "one"
    MANY = "many"


class EnvType(Enum):
    SINGLE = "single"
    PARALLEL = "parallel"


class AlgoType(Enum):
    ON_POLICY = "on_policy"
    OFF_POLICY = "off_policy"


class Deployed(Seeded):
    def __init__(self, seed: int, params: Params, select_action_fn: Callable) -> None:
        Seeded.__init__(self, seed)
        self.params = params
        self.select_action_fn = select_action_fn

    def select_action(self, observation):
        return self.select_action_fn(self.params, self.nextkey(), observation)


@chex.dataclass
class DeployedJit:
    params: Params
    select_action: Callable = struct.field(pytree_node=False)


class Base(ABC, Seeded):
    def __init__(
        self,
        config: AlgoConfig,
        train_state_factory: Callable,
        explore_factory: Callable,
        process_experience_factory: Callable,
        update_step_factory: Callable,
        *,
        rearrange_pattern: str = "b h w c -> b h w c",
        preprocess_fn: Callable = None,
        run_name: str = None,
        tabulate: bool = False,
    ):
        Seeded.__init__(self, config.seed)
        self.config = config
        self.algo_params = FrozenConfigDict(config.algo_params)

        self.rearrange_pattern = rearrange_pattern
        self.preprocess_fn = preprocess_fn
        self.tabulate = tabulate

        self.vectorized = self.config.env_cfg.n_envs > 1
        self.parallel = self.config.env_cfg.n_agents > 1

        self.state: TrainState = train_state_factory(
            self.nextkey(),
            self.config,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
        )

        self.explore_fn: Callable = explore_general_factory(
            explore_factory(self.state, self.config.algo_params),
            self.vectorized,
            self.parallel,
        )
        self.process_experience_fn: Callable = process_experience_general_factory(
            process_experience_factory(
                self.state,
                self.config.algo_params,
                self.vectorized,
                self.parallel,
            ),
            self.vectorized,
            self.parallel,
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

    @classmethod
    def unserialize(cls, data_dir: str):
        config_path = os.path.join(data_dir, "config")
        with open(config_path, "r") as f:
            config_dict = yaml.load(f, yaml.SafeLoader)
        config = ConfigDict(config_dict)

        extra_path = os.path.join(data_dir, "extra")
        with open(extra_path, "rb") as f:
            extra = cloudpickle.load(f)

        config.env_cfg = extra.pop("env_config")

        return cls(seed=config.seed, config=config, **extra)

    def deploy_agent(self, batched: bool) -> Deployed:
        return DeployedJit(
            params=self.state.params,
            select_action=explore_general_factory(
                self.explore_factory(self.state, self.config),
                batched=batched,
                parallel=False,
            ),
        )
