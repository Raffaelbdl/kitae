from dataclasses import dataclass
from collections import namedtuple
import functools
from typing import Any, Callable

import chex
import distrax as dx
import flax
from flax import struct
import flax.linen as nn
from flax.training import train_state
import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import jax.random as jrd
import ml_collections
import numpy as np
import optax

from rl import Base, Params
from rl.buffer import OnPolicyBuffer, OnPolicyExp
from rl.common import ensure_int, create_params
from rl.loss import loss_policy_ppo_discrete, loss_value_clip
from rl.timesteps import calculate_gaes_targets


@chex.dataclass
class ParamsPPO:
    params_policy: Params
    params_value: Params


class TrainStatePPO(train_state.TrainState):
    policy_fn: Callable = struct.field(pytree_node=False)
    value_fn: Callable = struct.field(pytree_node=False)


def create_modules(
    observation_space: spaces.Space, action_space: spaces.Discrete
) -> tuple[nn.Module, nn.Module]:
    if len(observation_space.shape) > 1:
        raise NotImplementedError

    class Policy(nn.Module):
        num_outputs: int

        @nn.compact
        def __call__(self, x: jax.Array):
            dtype = jnp.float32
            x = x.astype(dtype)
            x = nn.Dense(features=64, name="dense1", dtype=dtype)(x)
            x = nn.tanh(x)
            x = nn.Dense(features=64, name="dense2", dtype=dtype)(x)
            x = nn.tanh(x)
            return nn.Dense(features=self.num_outputs)(x)

    class Value(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            dtype = jnp.float32
            x = x.astype(dtype)
            x = nn.Dense(features=64, name="dense1", dtype=dtype)(x)
            x = nn.tanh(x)
            x = nn.Dense(features=64, name="dense2", dtype=dtype)(x)
            x = nn.tanh(x)
            return nn.Dense(features=1)(x)

    return Policy(action_space.n), Value()


def create_params_ppo(
    key: jax.Array, policy: nn.Module, value: nn.Module, observation_space: spaces.Space
) -> ParamsPPO:
    key1, key2 = jax.random.split(key, 2)
    return ParamsPPO(
        params_policy=create_params(key1, policy, observation_space),
        params_value=create_params(key2, value, observation_space),
    )


def create_train_state(
    policy: nn.Module,
    value: nn.Module,
    params_ppo: ParamsPPO,
    config: ml_collections.ConfigDict,
) -> TrainStatePPO:
    num_batches = config.max_buffer_size // config.batch_size
    if config.learning_rate_annealing:
        n_updates = (
            config.n_env_steps
            // config.max_buffer_size
            * config.num_epochs
            * num_batches
        )
        learning_rate = optax.linear_schedule(config.learning_rate, 0.0, n_updates)
    else:
        learning_rate = config.learning_rate

    tx = optax.chain(
        optax.adam(learning_rate), optax.clip_by_global_norm(config.max_grad_norm)
    )

    return TrainStatePPO.create(
        apply_fn=None,
        params=params_ppo,
        tx=tx,
        policy_fn=policy.apply,
        value_fn=value.apply,
    )


def get_logits_and_log_probs(
    params: Params,
    policy_fn: Callable,
    observations: jax.Array,
):
    logits = policy_fn({"params": params}, observations)
    return logits, nn.log_softmax(logits)


def loss_policy_fn(
    params_policy: Params,
    policy_fn: Callable,
    observations: jax.Array,
    actions: jax.Array,
    log_probs_old: jax.Array,
    gaes: jax.Array,
    clip_eps: float,
    entropy_coef: float,
):
    logits, all_log_probs = get_logits_and_log_probs(
        params_policy, policy_fn, observations
    )
    log_probs = jnp.take_along_axis(all_log_probs, actions, axis=-1)

    return loss_policy_ppo_discrete(
        logits, log_probs, log_probs_old, gaes, clip_eps, entropy_coef
    )


def loss_value_fn(
    params_value: Params,
    value_fn: Callable,
    observations: jax.Array,
    targets: jax.Array,
    values_old: jax.Array,
    clip_eps: jax.Array,
):
    values = value_fn({"params": params_value}, observations)
    return loss_value_clip(values, targets, values_old, clip_eps)


def loss_fn(
    params: ParamsPPO,
    policy_fn: Callable,
    value_fn: Callable,
    batch: tuple[jax.Array],
    # observations: jax.Array,
    # actions: jax.Array,
    # log_probs_old: jax.Array,
    # gaes: jax.Array,
    # targets: jax.Array,
    # values_old: jax.Array,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
):
    observations, actions, log_probs_old, gaes, targets, values_old = batch

    loss_policy, info_policy = loss_policy_fn(
        params.params_policy,
        policy_fn,
        observations,
        actions,
        log_probs_old,
        gaes,
        clip_eps,
        entropy_coef,
    )

    loss_value, info_value = loss_value_fn(
        params.params_value, value_fn, observations, targets, values_old, clip_eps
    )

    infos = info_value | info_policy
    return loss_policy + value_coef * loss_value, infos


def explore(
    key: jax.Array, params: Params, policy_fn: Callable, observations: jax.Array
) -> jax.Array:
    logits = policy_fn({"params": params}, observations)
    return dx.Categorical(logits=logits).sample_and_log_prob(seed=key)


def explore_unbatched(
    key: jax.Array, params: Params, policy_fn: Callable, observation: jax.Array
) -> jax.Array:
    observations = jnp.expand_dims(observation, axis=0)
    actions, log_probs = explore(key, params, policy_fn, observations)
    return jnp.squeeze(actions, axis=0), jnp.squeeze(log_probs, axis=0)


def process_experience(
    params: ParamsPPO,
    value_fn: Callable,
    gamma: float,
    _lambda: float,
    normalize: bool,
    sample: list[OnPolicyExp],
):
    from rl.buffer import array_of_name

    observations = array_of_name(sample, "observation")
    values = jax.jit(value_fn)({"params": params.params_value}, observations)

    next_observations = array_of_name(sample, "next_observation")
    next_values = jax.jit(value_fn)({"params": params.params_value}, next_observations)

    not_dones = 1.0 - array_of_name(sample, "done")[..., None]
    discounts = gamma * not_dones

    rewards = array_of_name(sample, "reward")[..., None]
    gaes, targets = jax.jit(calculate_gaes_targets, static_argnums=(4, 5))(
        values, next_values, discounts, rewards, _lambda, normalize
    )

    actions = array_of_name(sample, "action")[..., None]
    log_probs = array_of_name(sample, "log_prob")[..., None]
    return observations, actions, log_probs, gaes, targets, values


def update_step(
    key: jax.Array,
    state: TrainStatePPO,
    experiences: tuple,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    batch_size: int,
):
    num_elems = experiences[0].shape[0]
    iterations = num_elems // batch_size
    inds = jax.random.permutation(key, num_elems)[: iterations * batch_size]

    experiences = jax.tree_util.tree_map(
        lambda x: x[inds].reshape((iterations, batch_size) + x.shape[1:]),
        experiences,
    )

    loss = 0.0
    for batch in zip(*experiences):
        (l, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
            policy_fn=state.policy_fn,
            value_fn=state.value_fn,
            batch=batch,
            # observations=observations,
            # actions=actions,
            # log_probs_old=log_probs_old,
            # gaes=gaes,
            # targets=targets,
            # values_old=values_old,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss


class PPO(Base):
    def __init__(
        self,
        seed: int,
        env: gym.Env,
        config: ml_collections.ConfigDict,
        *,
        create_modules: Callable[..., tuple[nn.Module, nn.Module]] = create_modules,
        create_params: Callable[..., ParamsPPO] = create_params_ppo
    ):
        Base.__init__(self, seed)
        self.config = config

        policy, value = create_modules(env.observation_space, env.action_space)
        params = create_params(self.nextkey(), policy, value, env.observation_space)
        self.state = create_train_state(policy, value, params, self.config)

    def select_action(self, observation: jax.Array) -> jax.Array:
        """unbatched"""
        action, log_prob = jax.jit(explore_unbatched, static_argnums=2)(
            self.nextkey(),
            self.state.params.params_policy,
            self.state.policy_fn,
            observation,
        )
        return ensure_int(action), log_prob

    def explore(self, observation: jax.Array) -> jax.Array:
        """unbatched"""
        action, log_prob = jax.jit(explore_unbatched, static_argnums=2)(
            self.nextkey(),
            self.state.params.params_policy,
            self.state.policy_fn,
            observation,
        )
        return ensure_int(action), log_prob

    def update(self, buffer: OnPolicyBuffer):
        sample = buffer.sample()
        experiences = process_experience(
            self.state.params,
            self.state.value_fn,
            self.config.gamma,
            self.config._lambda,
            self.config.normalize,
            sample,
        )

        loss = 0.0
        for epoch in range(self.config.num_epochs):
            self.state, l = jax.jit(update_step, static_argnums=(3, 4, 5, 6))(
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


def train(seed: int, ppo: PPO, env: gym.Env, n_env_steps: int):
    buffer = OnPolicyBuffer(seed, ppo.config.max_buffer_size)

    observation, info = env.reset(seed=seed + 1)
    episode_return = 0.0

    for step in range(1, n_env_steps + 1):
        action, log_prob = ppo.explore(observation)
        next_observation, reward, done, trunc, info = env.step(action)
        episode_return += reward

        buffer.add(
            OnPolicyExp(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                next_observation=next_observation,
                log_prob=log_prob,
            )
        )

        if done or trunc:
            print(step, " > ", episode_return)
            episode_return = 0.0
            next_observation, info = env.reset()

        if len(buffer) >= ppo.config.max_buffer_size:
            ppo.update(buffer)

        observation = next_observation
