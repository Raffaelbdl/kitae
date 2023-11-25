from typing import Callable

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import pettingzoo
import vec_parallel_env

from rl.base import Base, EnvType, EnvProcs
from rl.buffer import OnPolicyBuffer, OnPolicyExp
from rl.timesteps import calculate_gaes_targets

ParallelEnv = pettingzoo.ParallelEnv
SubProcVecParallelEnv = vec_parallel_env.SubProcVecParallelEnv

from rl.algos.ppo import (
    ParamsPPO,
    TrainStatePPO,
    create_modules,
    create_params_ppo,
    create_train_state,
    compute_values,
    explore,
    explore_unbatched,
    update_step,
)


def process_experience(
    params: ParamsPPO,
    train_state: TrainStatePPO,
    gamma: float,
    _lambda: float,
    normalize: bool,
    sample: list[OnPolicyExp],
):
    from rl.buffer import multi_agent_array_of_name

    # could use jax tree_map
    observations = multi_agent_array_of_name(sample, "observation")
    values = {
        agent: jax.jit(compute_values, static_argnums=(1, 2))(
            params, train_state.value_fn, train_state.encoder_fn, observations[agent]
        )
        for agent in observations.keys()
    }

    next_observations = multi_agent_array_of_name(sample, "next_observation")
    next_values = {
        agent: jax.jit(compute_values, static_argnums=(1, 2))(
            params,
            train_state.value_fn,
            train_state.encoder_fn,
            next_observations[agent],
        )
        for agent in next_observations.keys()
    }

    dones = multi_agent_array_of_name(sample, "done")
    discounts = {
        agent: gamma * (1.0 - dones[agent][..., None]) for agent in dones.keys()
    }

    rewards = multi_agent_array_of_name(sample, "reward")
    rewards = {agent: rewards[agent][..., None] for agent in rewards.keys()}
    gaes, targets = {}, {}
    for agent in values.keys():
        g, t = jax.jit(calculate_gaes_targets, static_argnums=(4, 5))(
            values[agent],
            next_values[agent],
            discounts[agent],
            rewards[agent],
            _lambda,
            normalize,
        )
        gaes[agent] = g
        targets[agent] = t

    actions = multi_agent_array_of_name(sample, "action")
    actions = {agent: actions[agent][..., None] for agent in actions.keys()}
    log_probs = multi_agent_array_of_name(sample, "log_prob")
    log_probs = {agent: log_probs[agent][..., None] for agent in log_probs.keys()}

    # reshape everything into arrays
    observations = jnp.concatenate(list(observations.values()), axis=0)
    actions = jnp.concatenate(list(actions.values()), axis=0)
    log_probs = jnp.concatenate(list(log_probs.values()), axis=0)
    gaes = jnp.concatenate(list(gaes.values()), axis=0)
    targets = jnp.concatenate(list(targets.values()), axis=0)
    values = jnp.concatenate(list(values.values()), axis=0)

    return observations, actions, log_probs, gaes, targets, values


def process_experience_vectorized(
    params: ParamsPPO,
    train_state: TrainStatePPO,
    gamma: float,
    _lambda: float,
    normalize: bool,
    sample: list[OnPolicyExp],
):
    from rl.buffer import multi_agent_array_of_name

    # could use jax tree_map
    observations = multi_agent_array_of_name(sample, "observation")
    values = {
        agent: jax.vmap(
            jax.jit(compute_values, static_argnums=(1, 2)),
            in_axes=(None, None, None, 1),
            out_axes=1,
        )(params, train_state.value_fn, train_state.encoder_fn, observations[agent])
        for agent in observations.keys()
    }

    next_observations = multi_agent_array_of_name(sample, "next_observation")
    next_values = {
        agent: jax.vmap(
            jax.jit(compute_values, static_argnums=(1, 2)),
            in_axes=(None, None, None, 1),
            out_axes=1,
        )(
            params,
            train_state.value_fn,
            train_state.encoder_fn,
            next_observations[agent],
        )
        for agent in next_observations.keys()
    }

    dones = multi_agent_array_of_name(sample, "done")
    discounts = {
        agent: gamma * (1.0 - dones[agent][..., None]) for agent in dones.keys()
    }

    rewards = multi_agent_array_of_name(sample, "reward")
    rewards = {agent: rewards[agent][..., None] for agent in rewards.keys()}
    gaes, targets = {}, {}
    for agent in values.keys():
        g, t = jax.vmap(
            jax.jit(calculate_gaes_targets, static_argnums=(4, 5)),
            in_axes=(1, 1, 1, 1, None, None),
            out_axes=1,
        )(
            values[agent],
            next_values[agent],
            discounts[agent],
            rewards[agent],
            _lambda,
            normalize,
        )
        gaes[agent] = g
        targets[agent] = t

    actions = multi_agent_array_of_name(sample, "action")
    actions = {agent: actions[agent][..., None] for agent in actions.keys()}
    log_probs = multi_agent_array_of_name(sample, "log_prob")
    log_probs = {agent: log_probs[agent][..., None] for agent in log_probs.keys()}

    # reshape everything into arrays
    observations = jnp.concatenate(list(observations.values()), axis=0)
    actions = jnp.concatenate(list(actions.values()), axis=0)
    log_probs = jnp.concatenate(list(log_probs.values()), axis=0)
    gaes = jnp.concatenate(list(gaes.values()), axis=0)
    targets = jnp.concatenate(list(targets.values()), axis=0)
    values = jnp.concatenate(list(values.values()), axis=0)

    observations = jnp.reshape(observations, (-1, *observations.shape[2:]))
    actions = jnp.reshape(actions, (-1, *actions.shape[2:]))
    log_probs = jnp.reshape(log_probs, (-1, *log_probs.shape[2:]))
    gaes = jnp.reshape(gaes, (-1, *gaes.shape[2:]))
    targets = jnp.reshape(targets, (-1, *targets.shape[2:]))
    values = jnp.reshape(values, (-1, *values.shape[2:]))

    return observations, actions, log_probs, gaes, targets, values


class PPO(Base):
    def __init__(
        self,
        seed: int,
        env: gym.Env,
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
        params = create_params(
            self.nextkey(),
            policy,
            value,
            encoder,
            config.observation_space,
            shared_encoder=config.shared_encoder,
        )
        self.state = create_train_state(
            policy, value, encoder, params, self.config, n_envs=n_envs
        )

        self.explore_fn = explore if n_envs > 1 else explore_unbatched
        self.explore_fn = jax.jit(self.explore_fn, static_argnums=(1, 2))

        self.process_experience_fn = (
            process_experience_vectorized if n_envs > 1 else process_experience
        )

        self.n_envs = n_envs

    def select_action(
        self, observation: dict[str, jax.Array]
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        # could use a jax tree_map
        action, log_prob = {}, {}
        for agent, obs in observation.items():
            a, lp = self.explore_fn(
                self.nextkey(),
                self.state.policy_fn,
                self.state.encoder_fn,
                self.state.params,
                obs,
            )
            action[agent] = a
            log_prob[agent] = lp
        return action, log_prob

    def explore(self, observation: dict[str, jax.Array]) -> dict[str, jax.Array]:
        # could use a jax tree_map
        action, log_prob = {}, {}
        for agent, obs in observation.items():
            a, lp = self.explore_fn(
                self.nextkey(),
                self.state.policy_fn,
                self.state.encoder_fn,
                self.state.params,
                obs,
            )
            action[agent] = a
            log_prob[agent] = lp
        return action, log_prob

    def update(self, buffer: OnPolicyBuffer):
        sample = buffer.sample()
        experiences = self.process_experience_fn(
            self.state.params,
            self.state,
            self.config.gamma,
            self.config._lambda,
            self.config.normalize,
            sample,
        )

        loss = 0.0
        for epoch in range(self.config.num_epochs):
            self.state, l, info = jax.jit(update_step, static_argnums=(3, 4, 5, 6))(
                self.nextkey(),
                self.state,
                experiences,
                self.config.clip_eps,
                self.config.entropy_coef,
                self.config.value_coef,
                self.config.batch_size,
            )
            loss += l

        loss /= self.config.num_epochs
        return info

    def train(self, env: ParallelEnv | SubProcVecParallelEnv, n_env_steps: int):
        from rl.train import train

        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.PARALLEL,
            EnvProcs.ONE if self.n_envs == 1 else EnvProcs.MANY,
        )
