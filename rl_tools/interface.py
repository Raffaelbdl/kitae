from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, NamedTuple

from rl_tools.types import ActionType, ObsType, Array


class AlgoType(Enum):
    ON_POLICY = "on_policy"
    OFF_POLICY = "off_policy"


class IBuffer(ABC):
    """Interface for Buffer instances."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def add(self, experience: type[NamedTuple]) -> None: ...

    @abstractmethod
    def sample(self, sample_size: int) -> list[type[NamedTuple]]: ...


class IActor(ABC):
    """Interface for Actor instances."""

    @abstractmethod
    def select_action(self, observation: ObsType) -> tuple[ActionType, Array]:
        """Exploits the policy to interact with the environment.

        Args:
            observation: An ObsType within the observation_space.

        Returns:
            An ActionType within the action_space.
        """
        ...


class IAgent(ABC):
    """Interface for Agent instances."""

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
    def select_action(self, observation: ObsType) -> tuple[ActionType, Array]:
        """Exploits the policy to interact with the environment.

        Args:
            observation: An ObsType within the observation_space.

        Returns:
            An ActionType within the action_space.
        """
        ...

    @abstractmethod
    def should_update(self, step: int, buffer: IBuffer) -> bool:
        """Determines if the agent should be updated.

        Args:
            step: An int representing the current step for a single environment.
            buffer: A Buffer containing the transitions obtained from the environment.

        Returns:
            A boolean expliciting if the agent should be updated.
        """
        ...

    @abstractmethod
    def train(self, env: Any, n_env_steps: int) -> None:
        """Starts the training of the agent.

        Args:
            env: An EnvLike environment to train in.
            n_env_steps: An int representing the number of steps in a single environment.
        """
        ...

    @abstractmethod
    def resume(self, env: Any, n_env_steps: int) -> None:
        """Resumes the training of the agent from the last training step.

        Args:
            env: An EnvLike environment to train in.
            n_env_steps: An int representing the number of steps in a single environment.
        """
        ...
