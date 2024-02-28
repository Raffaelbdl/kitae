"""Contains the base classes for reinforcement learning."""

from pathlib import Path
from typing import Any, Callable

import chex
import cloudpickle
import yaml
from flax import struct
import jax

from jrd_extensions import Seeded

from rl_tools.interface import IAgent, IBuffer, AlgoType
from rl_tools.train import train

from rl_tools.save import Saver
from rl_tools.types import ActionType, ObsType, Params

from ml_collections import ConfigDict
from rl_tools.config import AlgoConfig

from rl_tools.algos.pipeline import PipelineModule
from rl_tools.algos.pipeline import ExperienceTransform
from rl_tools.algos.pipeline import UpdateModule

import numpy as np


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


class PipelineAgent(IAgent, Seeded):
    config: AlgoConfig = None
    algo_type: AlgoType = None

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

    train_fn: Callable = None

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

    def update(self, buffer: IBuffer) -> dict:
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

    def interact_keys(self, observation: ObsType) -> jax.Array | dict[str : jax.Array]:
        if self.parallel:
            return {a: self.nextkey() for a in observation.keys()}
        return self.nextkey()

    def train(self, env, n_env_steps, callbacks):
        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            self.parallel,
            self.vectorized,
            self.algo_type,
            saver=self.saver,
            callbacks=callbacks,
        )

    def resume(self, env, n_env_steps, callbacks):
        step, self.state = self.saver.restore_latest_step(self.state)
        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            self.parallel,
            self.vectorized,
            self.algo_type,
            saver=self.saver,
            callbacks=callbacks,
            start_step=step,
        )


class OffPolicyAgent(PipelineAgent):
    algo_type = AlgoType.OFF_POLICY
    step = 0

    def should_update(self, step: int, buffer: IBuffer) -> bool:
        self.step = step
        return (
            len(buffer) >= self.config.update_cfg.batch_size
            and step % self.config.algo_params.skip_steps == 0
            and step >= self.config.algo_params.start_step
        )


class OnPolicyAgent(PipelineAgent):
    algo_type = AlgoType.ON_POLICY

    def should_update(self, step: int, buffer: IBuffer) -> bool:
        return len(buffer) >= self.config.update_cfg.max_buffer_size
