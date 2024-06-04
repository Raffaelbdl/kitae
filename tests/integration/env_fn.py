from typing import Any, SupportsFloat
import envpool
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import supersuit as ss
from kitae.envs.wrappers.compatibility import EnvpoolCompatibility
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


def create_atari() -> gym.Env:
    env = gym.make("ALE/Pong-v5", frameskip=1)
    env = AtariPreprocessing(env, grayscale_newaxis=False)
    env = ss.frame_stack_v2(env, 4, -1)
    return env


def create_cartpole() -> gym.Env:
    env = gym.make("CartPole-v1")
    return env


def create_atari_envpool(n_envs: int):
    env = envpool.make("Pong-v5", env_type="gymnasium", num_envs=n_envs, seed=0)
    return EnvpoolCompatibility(env)


def create_cartpole_envpool(n_envs: int):
    env = envpool.make("CartPole-v1", env_type="gymnasium", num_envs=n_envs, seed=0)
    return EnvpoolCompatibility(env)
