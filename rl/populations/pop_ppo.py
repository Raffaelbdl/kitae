from typing import Callable, TypeVar

import chex
from flax import linen as nn
import gymnasium as gym
import jax
from jax import numpy as jnp
import ml_collections
import numpy as np

from rl.base import Base
from rl.buffer import OnPolicyBuffer, OnPolicyExp
from rl.loss import loss_shannon_jensen_divergence

EnvpoolEnv = TypeVar("EnvpoolEnv")

from rl.algos.ppo import (
    ParamsPPO,
    TrainStatePPO,
    create_modules,
    create_params_ppo,
    create_train_state,
    explore,
    explore_unbatched,
    process_experience,
    process_experience_vectorized,
    loss_fn as loss_single_fn,
)


def loss_fn(
    params_population: list[ParamsPPO],
    policy_fn: Callable,
    value_fn: Callable,
    encoder_fn: Callable,
    batch: list[tuple[jax.Array]],
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    jsd_coef: float,
):
    loss, infos = 0.0, {}
    logits_pop = []
    entropy_pop = []

    for i in range(len(params_population)):
        l, i = loss_single_fn(
            params_population[i],
            policy_fn,
            value_fn,
            encoder_fn,
            batch[i],
            clip_eps,
            entropy_coef,
            value_coef,
        )
        loss += l
        logits_pop.append(i["logits"])
        entropy_pop.append(i["entropy"])
        infos |= i

    logits_average = jnp.array(logits_pop).mean(axis=0)
    entropy_average = jnp.array(entropy_pop).mean(axis=0)[..., None]

    loss += jsd_coef * loss_shannon_jensen_divergence(logits_average, entropy_average)
    return loss, infos


def update_step(
    key: jax.Array,
    state: TrainStatePPO,
    experiences: list[tuple],
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    jsd_coef: float,
    batch_size: int,
):
    num_elems = experiences[0][0].shape[0]
    iterations = num_elems // batch_size
    inds = jax.random.permutation(key, num_elems)[: iterations * batch_size]

    experiences = jax.tree_util.tree_map(
        lambda x: x[inds].reshape((iterations, batch_size) + x.shape[1:]),
        experiences,
    )

    loss = 0.0
    for i in range(iterations):
        batch = [tuple(v[i] for v in e) for e in experiences]

        (l, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
            policy_fn=state.policy_fn,
            value_fn=state.value_fn,
            encoder_fn=state.encoder_fn,
            batch=batch,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            jsd_coef=jsd_coef,
        )
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss, info


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
        self, observations: list[jax.Array]
    ) -> list[tuple[jax.Array, jax.Array]]:
        return self.explore(observations)

    def explore(
        self, observations: list[jax.Array]
    ) -> list[tuple[jax.Array, jax.Array]]:
        actions, log_probs = [], []
        for i, obs in enumerate(observations):
            action, log_prob = self.explore_fn(
                self.nextkey(),
                self.state.policy_fn,
                self.state.encoder_fn,
                self.state.params[i],
                obs,
            )
            actions.append(action)
            log_probs.append(log_prob)
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
        info["total_loss"] = loss
        return info


def train(seed: int, population: PopulationPPO, envs: list[gym.Env], n_env_steps: int):
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
            next_observation, reward, done, trunc, info = env.step(int(actions[i]))
            episode_returns[i] += reward

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

            if done or trunc:
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
    seed: int, population: PopulationPPO, envs: list[EnvpoolEnv], n_envs_steps: int
):
    assert population.n_envs > 1

    buffers = [
        OnPolicyBuffer(seed + i, population.config.max_buffer_size)
        for i in range(population.population_size)
    ]

    observations, infos = zip(*[envs[i].reset() for i in range(len(envs))])
    episode_returns = np.zeros((len(observations), observations[0].shape[0]))

    update_info = {"kl_divergence": 0.0}

    for step in range(1, n_envs_steps + 1):
        actions, log_probs = population.explore(observations)

        next_observations = []
        for i, env in enumerate(envs):
            next_observation, reward, done, trunc, info = env.step(np.array(actions[i]))
            episode_returns[i] += reward

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

            for k, (d, t) in enumerate(zip(done, trunc)):
                if d or t:
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
