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


from rl.algos.pipelines import PipelineModule
from rl.algos.pipelines.experience_pipeline import ExperienceTransform
from rl.algos.pipelines.update_pipeline import UpdateModule


class Base(ABC, Seeded):
    intrinsic_reward_module: PipelineModule = None

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


class IntrinsicRewardManager:
    prefix: str = "intrinsic_"
    modules: dict[str, PipelineModule] = {}
    current_idx = 0

    def add(self, module: PipelineModule) -> None:
        self.modules[str(self.current_idx)] = module
        self.current_idx += 1

    @property
    def state_dict(self):
        return {self.prefix + i: m.state for i, m in self.modules.items()}

    def load_experience_transforms(self) -> list[ExperienceTransform]:
        return [m.experience_transform for m in self.modules.values()]

    def load_update_modules(self) -> dict[str, UpdateModule]:
        return {self.prefix + i: m.update_module for i, m in self.modules.items()}

    def apply_update(self, prefix_idx: str, state: Any) -> None:
        _idx = prefix_idx[len(self.prefix) :]
        self.modules[_idx] = self.modules[_idx].replace(state=state)


class Agent(ABC, Seeded):
    config: AlgoConfig = None

    rearrange_pattern: str = None
    preprocess_fn: Callable = None
    vectorized: bool = None
    parallel: bool = None

    main_pipeline_module: PipelineModule = None
    intrinsc_reward_manager = IntrinsicRewardManager()

    process_experience_pipeline: Callable = None
    update_pipeline: Callable = None

    explore_fn: Callable = None
    run_name: str = None
    saver: Saver = None

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

    @property
    def state(self) -> Any:
        return self.main_pipeline_module.state

    @property
    def state_dict(self) -> dict:
        _dict = {"main": self.main_pipeline_module.state}
        _dict |= self.intrinsc_reward_manager.state_dict

        return _dict

    def add_intrinsic_reward_module(self, module: PipelineModule) -> None:
        self.intrinsc_reward_manager.add(module)

    def load_experience_transforms(self) -> list[ExperienceTransform]:
        transforms = []
        transforms += self.intrinsc_reward_manager.load_experience_transforms()
        transforms.append(self.main_pipeline_module.experience_transform)

        return transforms

    def load_update_modules(self) -> list[UpdateModule]:
        modules = []
        self.module_to_state = {}

        for _id, module in self.intrinsc_reward_manager.load_update_modules().items():
            self.module_to_state[len(modules)] = _id
            modules.append(module)

        self.module_to_state[len(modules)] = "main"
        modules.append(self.main_pipeline_module.update_module)

        return modules

    def apply_updates(self, update_modules: list[UpdateModule]) -> None:
        for i, module in enumerate(update_modules):
            module_id = self.module_to_state[i]

            if "main" in module_id:
                self.main_pipeline_module = self.main_pipeline_module.replace(
                    state=module.state
                )

            elif "intrinsic_" in module_id:
                self.intrinsc_reward_manager.apply_update(module_id, module.state)

    def update(self, buffer: Buffer) -> dict:
        sample = buffer.sample(self.config.update_cfg.batch_size)
        experiences = self.process_experience_pipeline(
            self.load_experience_transforms(), self.nextkey(), sample
        )
        update_modules = self.load_update_modules()

        for epoch in range(self.config.update_cfg.n_epochs):
            update_modules, info = self.update_pipeline(
                update_modules, self.nextkey(), experiences
            )

        self.apply_updates(update_modules)

        return info

    def restore(self) -> int:
        """Restores the agent's states from the last training step.

        Returns:
            The latest training step.
        """
        latest_step, state_dict = self.saver.restore_latest_step(self.state_dict)
        for module_id, state in state_dict.items():

            if "main" in module_id:
                self.main_pipeline_module = self.main_pipeline_module.replace(
                    state=state
                )

            elif "intrinsic_" in module_id:
                self.intrinsc_reward_manager.apply_update(module_id, state)

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


class OffPolicyAgent(Agent):
    step = 0

    def should_update(self, step: int, buffer: Buffer) -> bool:
        self.step = step
        return (
            len(buffer) >= self.config.update_cfg.batch_size
            and step % self.config.algo_params.skip_steps == 0
            and step >= self.config.algo_params.start_step
        )


class OnPolicyAgent(Agent):
    def should_update(self, step: int, buffer: Buffer) -> bool:
        return len(buffer) >= self.config.update_cfg.max_buffer_size
