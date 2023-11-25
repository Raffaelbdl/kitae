from typing import Callable, TypeVar

import chex
from flax import linen as nn
import gymnasium as gym
import jax
from jax import numpy as jnp
import ml_collections
import numpy as np
import pettingzoo
import vec_parallel_env

from rl.base import Base
from rl.buffer import OnPolicyBuffer, OnPolicyExp

ParallelEnv = pettingzoo.ParallelEnv
SubProcParallelEnv = vec_parallel_env.SubProcVecParallelEnv

from rl.algos.ppo import (
    ParamsPPO,
    create_modules,
    create_params_ppo,
    create_train_state,
    explore,
    explore_unbatched,
)

from rl.algos.ippo import process_experience, process_experience_vectorized
from rl.populations.pop_ppo import update_step


class PopulationPPO(Base):
    def __init__(
        self,
        seed: int,
        population_size: int,
        config: ml_collections.ConfigDict,
        *,
        create_modules: Callable[..., tuple[nn.Module, nn.Module]] = create_modules,
        create_params: Callable[..., ParamsPPO] = create_params_ppo,
        rearrange_pattern: str = "b h w c -> b h w c",
        n_envs: int = 1,
    ):
        Base.__init__(self, seed)
        self.config = config

        policy, value, encoder = create_modules(
            config.observation_space,
            config.action_space,
            config.shared_encoder,
            rearrange_pattern=rearrange_pattern,
        )
        params_population = [
            create_params(
                self.nextkey(),
                policy,
                value,
                encoder,
                config.observation_space,
                shared_encoder=config.shared_encoder,
            )
            for _ in range(population_size)
        ]
        self.state = create_train_state(
            policy, value, encoder, params_population, config, n_envs=n_envs
        )

        self.explore_fn = explore if n_envs > 1 else explore_unbatched
        self.explore_fn = jax.jit(self.explore_fn, static_argnums=(1, 2))

        self.process_experience_fn = (
            process_experience_vectorized if n_envs > 1 else process_experience
        )

        self.population_size = population_size
        self.n_envs = n_envs

    def select_action(
        self, observations: list[dict[str, jax.Array]]
    ) -> tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]]:
        return self.explore(observations)

    def explore(
        self, observations: list[dict[str, jax.Array]]
    ) -> tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]]:
        actions, log_probs = [], []
        for i, obs in enumerate(observations):
            _actions, _log_probs = {}, {}
            for agent, o in obs.items():
                a, lp = self.explore_fn(
                    self.nextkey(),
                    self.state.policy_fn,
                    self.state.encoder_fn,
                    self.state.params[i],
                    o,
                )
                _actions[agent] = a
                _log_probs[agent] = lp
            actions.append(_actions)
            log_probs.append(_log_probs)
        return actions, log_probs

    def update(self, buffers: list[OnPolicyBuffer]) -> None:
        experiences = [
            self.process_experience_fn(
                self.state.params[i],
                self.state,
                self.config.gamma,
                self.config._lambda,
                self.config.normalize,
                buffers[i].sample(),
            )
            for i in range(len(buffers))
        ]

        loss = 0.0
        for epoch in range(self.config.num_epochs):
            self.state, l, info = jax.jit(update_step, static_argnums=(3, 4, 5, 6, 7))(
                self.nextkey(),
                self.state,
                experiences,
                self.config.clip_eps,
                self.config.entropy_coef,
                self.config.value_coef,
                self.config.jsd_coef,
                self.config.batch_size,
            )
            loss += l

        loss /= self.config.num_epochs
        return info


def train(
    seed: int, population: PopulationPPO, envs: list[ParallelEnv], n_env_steps: int
):
    assert population.n_envs == 1

    buffers = [
        OnPolicyBuffer(seed + i, population.config.max_buffer_size)
        for i in range(population.population_size)
    ]

    observations, infos = zip(*[envs[i].reset(seed=seed + i) for i in range(len(envs))])
    episode_returns = np.zeros((len(observations),))

    update_info = {"kl_divergence": 0.0}

    for step in range(1, n_env_steps + 1):
        actions, log_probs = population.explore(observations)

        next_observations = []
        for i, env in enumerate(envs):
            next_observation, reward, done, trunc, info = env.step(
                {agent: int(a) for agent, a in actions[i].items()}
            )
            episode_returns[i] += sum(reward.values())

            buffers[i].add(
                OnPolicyExp(
                    observation=observations[i],
                    action=actions[i],
                    reward=reward,
                    done=done,
                    next_observation=next_observation,
                    log_prob=log_probs[i],
                )
            )

            if any(done.values()) or any(trunc.values()):
                if i == 0:
                    print(
                        step,
                        " > ",
                        episode_returns[i],
                        " | ",
                        update_info["kl_divergence"],
                    )
                episode_returns[i] = 0.0
                next_observation, info = env.reset()

            next_observations.append(next_observation)

        if len(buffers[0]) >= population.config.max_buffer_size:
            update_info |= population.update(buffers)

        observations = next_observations


def train_vectorized(
    seed: int,
    population: PopulationPPO,
    envs: list[SubProcParallelEnv],
    n_envs_steps: int,
):
    assert population.n_envs >= 1

    buffers = [
        OnPolicyBuffer(seed + i, population.config.max_buffer_size)
        for i in range(population.population_size)
    ]

    observations, infos = zip(*[envs[i].reset() for i in range(len(envs))])
    episode_returns = np.zeros(
        (len(observations), np.array(list(observations[0].values())[0]).shape[0])
    )

    update_info = {"kl_divergence": 0.0}

    for step in range(1, n_envs_steps + 1):
        actions, log_probs = population.explore(observations)

        next_observations = []
        for i, env in enumerate(envs):
            next_observation, reward, done, trunc, info = env.step(actions[i])
            episode_returns[i] += np.sum(np.array(list(reward.values())), axis=0)

            buffers[i].add(
                OnPolicyExp(
                    observation=observations[i],
                    action=actions[i],
                    reward=reward,
                    done=done,
                    next_observation=next_observation,
                    log_prob=log_probs[i],
                )
            )

            check_d, check_t = np.stack(list(done.values()), axis=1), np.stack(
                list(trunc.values()), axis=1
            )
            for k, (d, t) in enumerate(zip(check_d, check_t)):
                if np.any(d) or np.any(t):
                    if i == 0 and k == 0:
                        print(
                            step,
                            " > ",
                            np.mean(episode_returns[i]),
                            " | ",
                            update_info["kl_divergence"],
                        )
                    episode_returns[i][k] = 0.0

            next_observations.append(next_observation)

        if len(buffers[0]) >= population.config.max_buffer_size:
            update_info |= population.update(buffers)

        observations = next_observations

    env.close()
