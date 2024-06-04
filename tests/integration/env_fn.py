from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from kitae.envs.make import wrap_single_env


class RandomEnv(gym.Env):
    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs = self.observation_space.sample()
        return obs, 1.0, False, False, {}


class DiscreteEnv(RandomEnv):
    action_space = Discrete(10)
    observation_space = Box(-1.0, 1.0, (20,))


def make_discrete_env():
    return wrap_single_env(DiscreteEnv())


class ContinuousEnv(RandomEnv):
    action_space = Discrete(10)
    observation_space = Box(-1.0, 1.0, (20,))


def make_continuous_env():
    return wrap_single_env(ContinuousEnv())
