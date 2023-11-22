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
    params_encoder: Params


class TrainStatePPO(train_state.TrainState):
    policy_fn: Callable = struct.field(pytree_node=False)
    value_fn: Callable = struct.field(pytree_node=False)
    encoder_fn: Callable = struct.field(pytree_node=False)


def create_modules(
    observation_space: spaces.Space, action_space: spaces.Discrete
) -> tuple[nn.Module, nn.Module]:
    if len(observation_space.shape) < 3:
        raise NotImplementedError

    def conv_layer(
        features, kernel_size, strides, std=np.sqrt(2.0), bias_cst=0.0
    ) -> nn.Conv:
        return nn.Conv(
            features,
            (kernel_size, kernel_size),
            strides,
            padding="VALID",
            kernel_init=nn.initializers.orthogonal(std),
            bias_init=nn.initializers.constant(bias_cst),
        )

    class Encoder(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            dtype = jnp.float32
            x = x.astype(dtype) / 255.0
            x = conv_layer(32, 8, 4)(x)
            x = nn.relu(x)
            x = conv_layer(64, 4, 2)(x)
            x = nn.relu(x)
            x = conv_layer(64, 3, 1)(x)
            x = nn.relu(x)

            x = jnp.reshape(x, (x.shape[0], -1))
            x = nn.Dense(
                512,
                kernel_init=nn.initializers.orthogonal(2.0),
                bias_init=nn.initializers.constant(0.0),
            )(x)
            return nn.relu(x)

    class Policy(nn.Module):
        num_outputs: int

        @nn.compact
        def __call__(self, x: jax.Array):
            return nn.Dense(
                features=self.num_outputs,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.constant(0.0),
            )(x)

    class Value(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            return nn.Dense(
                features=1,
                kernel_init=nn.initializers.orthogonal(1.0),
                bias_init=nn.initializers.constant(0.0),
            )(x)

    return Policy(action_space.n), Value(), Encoder()


def create_params_ppo(
    key: jax.Array,
    policy: nn.Module,
    value: nn.Module,
    encoder: nn.Module,
    observation_space: spaces.Space,
) -> ParamsPPO:
    key1, key2, key3 = jax.random.split(key, 3)
    hidden_shape = (1, 512)
    return ParamsPPO(
        params_policy=create_params(key1, policy, hidden_shape),
        params_value=create_params(key2, value, hidden_shape),
        params_encoder=create_params(key3, encoder, observation_space.shape),
    )


def create_train_state(
    policy: nn.Module,
    value: nn.Module,
    encoder: nn.Module,
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
        learning_rate = optax.linear_schedule(config.learning_rate, 0.0, n_updates, 0)
    else:
        learning_rate = config.learning_rate

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate)
    )

    return TrainStatePPO.create(
        apply_fn=None,
        params=params_ppo,
        tx=tx,
        policy_fn=policy.apply,
        value_fn=value.apply,
        encoder_fn=encoder.apply,
    )


def get_logits_and_log_probs(
    params_policy: Params,
    params_encoder: Params,
    policy_fn: Callable,
    encoder_fn: Callable,
    observations: jax.Array,
):
    hiddens = encoder_fn({"params": params_encoder}, observations)
    logits = policy_fn({"params": params_policy}, hiddens)
    return logits, nn.log_softmax(logits)


def compute_values(
    params: ParamsPPO,
    value_fn: Callable,
    encoder_fn: Callable,
    observations: jax.Array,
):
    hiddens = encoder_fn({"params": params.params_encoder}, observations)
    return value_fn({"params": params.params_value}, hiddens)


def loss_policy_fn(
    params_policy: Params,
    params_encoder: Params,
    policy_fn: Callable,
    encoder_fn: Callable,
    observations: jax.Array,
    actions: jax.Array,
    log_probs_old: jax.Array,
    gaes: jax.Array,
    clip_eps: float,
    entropy_coef: float,
):
    logits, all_log_probs = get_logits_and_log_probs(
        params_policy, params_encoder, policy_fn, encoder_fn, observations
    )
    log_probs = jnp.take_along_axis(all_log_probs, actions, axis=-1)

    return loss_policy_ppo_discrete(
        logits, log_probs, log_probs_old, gaes, clip_eps, entropy_coef
    )


def loss_value_fn(
    params_value: Params,
    params_encoder: Params,
    value_fn: Callable,
    encoder_fn: Callable,
    observations: jax.Array,
    targets: jax.Array,
    values_old: jax.Array,
    clip_eps: jax.Array,
):
    hiddens = encoder_fn({"params": params_encoder}, observations)
    values = value_fn({"params": params_value}, hiddens)
    return loss_value_clip(values, targets, values_old, clip_eps)


def loss_fn(
    params: ParamsPPO,
    policy_fn: Callable,
    value_fn: Callable,
    encoder_fn: Callable,
    batch: tuple[jax.Array],
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
):
    observations, actions, log_probs_old, gaes, targets, values_old = batch

    loss_policy, info_policy = loss_policy_fn(
        params.params_policy,
        params.params_encoder,
        policy_fn,
        encoder_fn,
        observations,
        actions,
        log_probs_old,
        gaes,
        clip_eps,
        entropy_coef,
    )

    loss_value, info_value = loss_value_fn(
        params.params_value,
        params.params_encoder,
        value_fn,
        encoder_fn,
        observations,
        targets,
        values_old,
        clip_eps,
    )

    infos = info_value | info_policy
    return loss_policy + value_coef * loss_value, infos


def explore(
    key: jax.Array,
    policy_fn: Callable,
    encoder_fn: Callable,
    params: ParamsPPO,
    observations: jax.Array,
) -> jax.Array:
    hiddens = encoder_fn({"params": params.params_encoder}, observations)
    logits = policy_fn({"params": params.params_policy}, hiddens)
    return dx.Categorical(logits=logits).sample_and_log_prob(seed=key)


def explore_unbatched(
    key: jax.Array,
    policy_fn: Callable,
    encoder_fn: Callable,
    params: ParamsPPO,
    observation: jax.Array,
) -> jax.Array:
    observations = jnp.expand_dims(observation, axis=0)
    actions, log_probs = explore(key, policy_fn, encoder_fn, params, observations)
    return jnp.squeeze(actions, axis=0), jnp.squeeze(log_probs, axis=0)


def process_experience(
    params: ParamsPPO,
    train_state: TrainStatePPO,
    gamma: float,
    _lambda: float,
    normalize: bool,
    sample: list[OnPolicyExp],
):
    from rl.buffer import array_of_name

    observations = array_of_name(sample, "observation")
    values = jax.jit(compute_values, static_argnums=(1, 2))(
        params, train_state.value_fn, train_state.encoder_fn, observations
    )

    next_observations = array_of_name(sample, "next_observation")
    next_values = jax.jit(compute_values, static_argnums=(1, 2))(
        params, train_state.value_fn, train_state.encoder_fn, next_observations
    )

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
            encoder_fn=state.encoder_fn,
            batch=batch,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss, info


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

        policy, value, encoder = create_modules(env.observation_space, env.action_space)
        params = create_params(
            self.nextkey(), policy, value, encoder, env.observation_space
        )
        self.state = create_train_state(policy, value, encoder, params, self.config)

    def select_action(self, observation: jax.Array) -> jax.Array:
        """unbatched"""
        action, log_prob = jax.jit(explore_unbatched, static_argnums=1)(
            self.nextkey(),
            self.state,
            self.state.params,
            observation,
        )
        return ensure_int(action), log_prob

    def explore(self, observation: jax.Array) -> jax.Array:
        """unbatched"""
        action, log_prob = jax.jit(explore_unbatched, static_argnums=(1, 2))(
            self.nextkey(),
            self.state.policy_fn,
            self.state.encoder_fn,
            self.state.params,
            observation,
        )
        return ensure_int(action), log_prob

    def update(self, buffer: OnPolicyBuffer):
        sample = buffer.sample()
        experiences = process_experience(
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


def train(seed: int, ppo: PPO, env: gym.Env, n_env_steps: int):
    buffer = OnPolicyBuffer(seed, ppo.config.max_buffer_size)

    observation, info = env.reset(seed=seed + 1)
    episode_return = 0.0

    update_info = {"kl_divergence": 0.0}

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
            print(step, " > ", episode_return, " | ", update_info["kl_divergence"])
            episode_return = 0.0
            next_observation, info = env.reset()

        if len(buffer) >= ppo.config.max_buffer_size:
            update_info = ppo.update(buffer)

        observation = next_observation
