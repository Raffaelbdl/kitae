from typing import Any

import gymnasium as gym
import numpy as np
from rl_tools.callbacks.callback import Callback

from rl_tools.interface import IAgent, IBuffer
from rl_tools.types import Array

from rl_tools.config import AlgoConfig


class DummySaver:
    def save(self, *args, **kwargs): ...


class RandomAgent(IAgent):
    def __init__(self, config: AlgoConfig, action_space: gym.Space) -> None:
        super().__init__()
        self.config = config
        self.action_space = action_space
        self.state_dict = {}
        self.saver = DummySaver()

    def select_action(self, observation: Array) -> tuple[Array, Array]:
        return self.explore(observation)

    def explore(self, observation: Array) -> tuple[Array, Array]:
        action = [self.action_space.sample() for _ in range(len(observation))]
        action = np.array(action)
        log_prob = np.zeros_like(action)
        return action, log_prob

    def should_update(self, step: int, buffer: IBuffer) -> bool:
        return False

    def train(self, env: gym.Env, n_env_steps: int, callbacks: list[Callback]) -> None:
        return None

    def resume(self, env: gym.Env, n_env_steps: int, callbacks: list[Callback]) -> None:
        return None


class RandomParallelAgent(IAgent):
    def __init__(self, config: AlgoConfig, action_space: gym.Space) -> None:
        super().__init__()
        self.config = config
        self.action_space = action_space
        self.state_dict = {}
        self.saver = DummySaver()

    def select_action(self, observation: Array) -> tuple[Array, Array]:
        return self.explore(observation)

    def explore(self, observation: dict[Any, Array]) -> tuple[Array, Array]:
        action = {}
        log_prob = {}
        for agent, obs in observation.items():
            _action = np.array([self.action_space.sample() for _ in range(len(obs))])
            action[agent] = _action
            log_prob[agent] = np.zeros_like(_action)

        return action, log_prob

    def should_update(self, step: int, buffer: IBuffer) -> bool:
        return False

    def train(self, env: gym.Env, n_env_steps: int, callbacks: list[Callback]) -> None:
        return None

    def resume(self, env: gym.Env, n_env_steps: int, callbacks: list[Callback]) -> None:
        return None
