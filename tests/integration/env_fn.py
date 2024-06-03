import envpool
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import supersuit as ss
from kitae.envs.wrappers.compatibility import EnvpoolCompatibility


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
