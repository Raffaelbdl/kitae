from typing import Any

import gymnasium as gym
from pettingzoo import ParallelEnv


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
