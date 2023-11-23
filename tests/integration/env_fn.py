import envpool
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack


def create_atari():
    env = gym.make("ALE/Pong-v5", frameskip=1)
    env = AtariPreprocessing(env, grayscale_newaxis=False)
    env = FrameStack(env, 4, False)
    return env


def create_cartpole():
    env = gym.make("CartPole-v1")
    return env


def create_atari_envpool(n_envs: int):
    env = envpool.make("Pong-v5", env_type="gymnasium", num_envs=n_envs, seed=0)
    return env


def create_cartpole_envpool(n_envs: int):
    env = envpool.make("CartPole-v1", env_type="gymnasium", num_envs=n_envs, seed=0)
    return env
