"""Contains the base classes for reinforcement learning."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import os
from pathlib import Path
from typing import Any, Callable, NamedTuple

import chex
import cloudpickle
import yaml
from flax import struct

from jrd_extensions import Seeded


from rl.buffer import Buffer, Experience
from rl.callbacks.callback import Callback
from rl.save import Saver
from rl.types import ActionType, ObsType, Params, Array

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
    """Algorithm-agnostic class for agents."""

    def __init__(self, seed: int, params: Params, select_action_fn: Callable) -> None:
        """Initializes a Deployed instance of an agent.

        Args:
            seed: An int for reproducibility.
            params: The Params of the agent.
            select_action_fn: The Callable select_action method of the agent.
        """
        Seeded.__init__(self, seed)
        self.params = params
        self.select_action_fn = select_action_fn

    def select_action(self, observation: ObsType) -> ActionType:
        """Exploits the policy to interact with the environment."""
        return self.select_action_fn(self.params, self.nextkey(), observation)


@chex.dataclass
class DeployedJit:
    """Jittable algorithm-agnostic class for agents."""

    params: Params
    select_action: Callable = struct.field(pytree_node=False)


from rl.algos.pipelines.experience_pipeline import ExperienceTransform
from rl.algos.pipelines.update_pipeline import UpdateModule
from rl.algos.pipelines.intrinsic_reward import IntrinsicRewardModule


class Base(ABC, Seeded):
    intrinsic_reward_module: IntrinsicRewardModule = None

    def experience_transforms(self, state) -> list[ExperienceTransform]:
        transforms = []

        if self.intrinsic_reward_module:
            transforms.append(
                ExperienceTransform(
                    process_experience_fn=self.intrinsic_reward_module.process_experience_fn,
                    state=self.intrinsic_reward_module.state,
                )
            )

        transforms.append(
            ExperienceTransform(
                process_experience_fn=self.process_experience_fn, state=state
            )
        )

        return transforms

    def make_update_modules(self, state) -> list[UpdateModule]:
        modules = []
        self.modules_to_state = {}

        if self.intrinsic_reward_module:
            self.modules_to_state[len(modules)] = "intrinsic_reward_module"
            modules.append(
                UpdateModule(
                    update_fn=self.intrinsic_reward_module.update_fn,
                    state=self.intrinsic_reward_module.state,
                )
            )

        self.modules_to_state[len(modules)] = "state"
        modules.append(UpdateModule(update_fn=self.update_step_fn, state=state))

        return modules

    def apply_updates(self, update_modules: list[UpdateModule]):
        for i in range(len(update_modules)):
            if self.modules_to_state[i] == "intrinsic_reward_module":
                self.intrinsic_reward_module = self.intrinsic_reward_module.replace(
                    state=update_modules[i].state
                )
            elif self.modules_to_state[i] == "state":
                self.state = update_modules[i].state

    @abstractmethod
    def select_action(self, observation: ObsType) -> tuple[ActionType, Array]:
        """Exploits the policy to interact with the environment.

        Args:
            observation: An ObsType within the observation_space.

        Returns:
            An ActionType within the action_space.
        """
        ...

    @abstractmethod
    def explore(self, observation: ObsType) -> tuple[ActionType, Array]:
        """Uses the policy to explore the environment.

        Args:
            observation: An ObsType within the observation_space.

        Returns:
            An ActionType within the action_space.
        """
        ...

    @abstractmethod
    def should_update(self, step: int, buffer: Buffer) -> bool:
        """Determines if the agent should be updated.

        Args:
            step: An int representing the current step for a single environment.
            buffer: A Buffer containing the transitions obtained from the environment.

        Returns:
            A boolean expliciting if the agent should be updated.
        """
        ...

    @abstractmethod
    def update(self, buffer: Buffer) -> dict:
        """Updates the agent.

        Args:
            buffer: A Buffer containing the transitions obtained from the environment.

        Returns:
            A dict containing the information from the update step.
        """
        ...

    @abstractmethod
    def train(self, env: Any, n_env_steps: int, callbacks: list[Callback]) -> None:
        """Starts the training of the agent.

        Args:
            env: An EnvLike environment to train in.
            n_env_steps: An int representing the number of steps in a single environment.
            callbacks: A list of Callbacks called during training
        """
        ...

    @abstractmethod
    def resume(self, env: Any, n_env_steps: int, callbacks: list[Callback]) -> None:
        """Resumes the training of the agent from the last training step.

        Args:
            env: An EnvLike environment to train in.
            n_env_steps: An int representing the number of steps in a single environment.
            callbacks: A list of Callbacks called during training
        """
        ...

    def restore(self) -> int:
        """Restores the agent's states from the last training step.

        Returns:
            The latest training step.
        """
        latest_step, self.state = self.saver.restore_latest_step(self.state)
        return latest_step

    @classmethod
    def unserialize(cls, data_dir: str | Path):
        """Creates an instance of the agent from a training directory.

        Args:
            data_dir: A string representing the path to the training directory.

        Returns:
            An instance of the specific Base class.

        # TODO : Raise if training directory does not correspond to class
        """
        path = data_dir if isinstance(data_dir, Path) else Path(data_dir)

        config_path = Path.joinpath(path, "config")
        with open(config_path, "r") as f:
            config_dict = yaml.load(f, yaml.SafeLoader)
        config = ConfigDict(config_dict)

        extra_path = Path.joinpath(path, "extra")
        with open(extra_path, "rb") as f:
            extra = cloudpickle.load(f)

        config.env_cfg = extra.pop("env_config")
        extra["run_name"] = path.parts[-1]

        return cls(config=config, **extra)
