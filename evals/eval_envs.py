from typing import Any

import envpool
import gymnasium as gym
from pettingzoo import ParallelEnv
from vec_parallel_env import SubProcVecParallelEnv

from rl.config import EnvConfig
from rl.wrapper import EnvpoolCompatibility, SubProcVecParallelEnvCompatibility


class CartPoleParallel(ParallelEnv):
    metadata = {"name": "CartPoleParallel"}
    agents = ["agent_0", "agent_1"]

    def __init__(self) -> None:
        super().__init__()

        self.envs = [gym.make("CartPole-v1"), gym.make("CartPole-v1")]

        self.possible_agents = self.agents
        self.action_spaces = {
            agent: env.action_space for agent, env in zip(self.agents, self.envs)
        }
        self.observation_spaces = {
            agent: env.observation_space for agent, env in zip(self.agents, self.envs)
        }

    def step(
        self, actions: dict
    ) -> tuple[
        dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]
    ]:
        observations = {}
        rewards = {}
        dones = {}
        truncs = {}
        infos = {}
        for (agent, action), env in zip(actions.items(), self.envs):
            o, r, d, t, i = env.step(int(action))
            observations[agent] = o
            rewards[agent] = r
            dones[agent] = d
            truncs[agent] = t
            infos[agent] = i

        return observations, rewards, dones, truncs, infos

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        observations = {}
        infos = {}
        for agent, env in zip(self.agents, self.envs):
            o, i = env.reset(seed=seed, options=options)
            observations[agent] = o
            infos[agent] = i
        return observations, infos


def make_cartpole():
    env = gym.make("CartPole-v1")
    return env, EnvConfig(env.observation_space, env.action_space, 1, 1)


def make_cartpole_vector(n_envs: int):
    env = envpool.make("CartPole-v1", env_type="gymnasium", num_envs=n_envs)
    env_config = EnvConfig(env.observation_space, env.action_space, n_envs, 1)
    env = EnvpoolCompatibility(env)
    return env, env_config


def make_cartpole_parallel():
    env = CartPoleParallel()
    env.reset()
    env_config = EnvConfig(
        env.observation_space(env.agents[0]),
        env.action_space(env.agents[0]),
        1,
        2,
    )
    return env, env_config


def make_cartpole_parallel_vector(n_envs: int):
    env = CartPoleParallel()
    env.reset()
    env_config = EnvConfig(
        env.observation_space(env.agents[0]),
        env.action_space(env.agents[0]),
        n_envs,
        2,
    )
    env = SubProcVecParallelEnv([lambda: env for _ in range(n_envs)])
    env = SubProcVecParallelEnvCompatibility(env)
    return env, env_config


def make_pong_vector(n_envs: int):
    env = EnvpoolCompatibility(
        envpool.make("Pong-v5", env_type="gymnasium", num_envs=n_envs)
    )
    return env, EnvConfig(env.observation_space, env.action_space, n_envs, 1)
