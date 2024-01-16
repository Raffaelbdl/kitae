"""Contains the base classes for reinforcement learning."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import os
from typing import Any, Callable, NamedTuple

import chex
import cloudpickle
import yaml
from flax import struct
from flax.training.train_state import TrainState

from jrd_extensions import Seeded

from rl.algos.general_fns import explore_general_factory
from rl.algos.general_fns import process_experience_general_factory

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
        experience_type: NamedTuple = Experience,
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
            ),
            self.vectorized,
            self.parallel,
            experience_type,
        )

        self.update_step_fn = update_step_factory(self.state, self.config)

        self.explore_factory = explore_factory

        self.run_name = run_name
        if self.run_name is None:
            self.run_name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.saver = Saver(os.path.join("./results", self.run_name), self)

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
    def unserialize(cls, data_dir: str):
        """Creates an instance of the agent from a training directory.

        Args:
            data_dir: A string representing the path to the training directory.

        Returns:
            An instance of the specific Base class.

        # TODO : Raise if training directory does not correspond to class
        """
        config_path = os.path.join(data_dir, "config")
        with open(config_path, "r") as f:
            config_dict = yaml.load(f, yaml.SafeLoader)
        config = ConfigDict(config_dict)

        extra_path = os.path.join(data_dir, "extra")
        with open(extra_path, "rb") as f:
            extra = cloudpickle.load(f)

        config.env_cfg = extra.pop("env_config")

        return cls(seed=config.seed, config=config, **extra)

    def deploy_agent(self, batched: bool) -> DeployedJit:
        """Creates a jittable instance of the agent.

        Args:
            batched: A boolean determining if the observation inputs will be batched.

        Returns:
            A DeployedJit instance of the agent.
        """
        return DeployedJit(
            params=self.state.params,
            select_action=explore_general_factory(
                self.explore_factory(self.state, self.config),
                batched=batched,
                parallel=False,
            ),
        )
