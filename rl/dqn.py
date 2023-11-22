import functools
from typing import Any, Callable

import flax
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
from rl.buffer import OffPolicyBuffer, OffPolicyExp
from rl.common import ensure_int, create_params

NO_EXPLORATION = 0.0


def create_module(
    observation_space: spaces.Space, action_space: spaces.Discrete
) -> nn.Module:
    if len(observation_space.shape) > 1:
        raise NotImplementedError

    class QValue(nn.Module):
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

    return QValue(action_space.n)


def create_train_state(
    module: nn.Module, params: Params, config: ml_collections.ConfigDict
) -> train_state.TrainState:
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)


def loss_fn(params: Params, apply_fn: Callable, states: jax.Array, returns: jax.Array):
    q_values = apply_fn({"params": params}, states)
    return jnp.mean(jnp.square(q_values - returns))


def compute_q_values(
    params: Params, apply_fn: Callable, observations: jax.Array
) -> jax.Array:
    return apply_fn({"params": params}, observations)


def explore(
    key: jax.Array,
    params: Params,
    apply_fn: Callable,
    exp_coef: float,
    observations: jax.Array,
    n_actions: int,
) -> jax.Array:
    """batched"""
    qvalues = apply_fn({"params": params}, observations)
    greedy_actions = jnp.argmax(qvalues, axis=-1, keepdims=False)

    eps = jrd.uniform(key, greedy_actions.shape)

    random_actions = jrd.randint(key, greedy_actions.shape, 0, qvalues.shape[-1])
    return jnp.where(eps <= exp_coef, random_actions, greedy_actions)


def explore_unbatched(
    key: jax.Array,
    params: Params,
    apply_fn: Callable[..., Any],
    exp_coef: float,
    observation: jax.Array,
    n_actions: int,
) -> jax.Array:
    observations = jnp.expand_dims(observation, axis=0)
    actions = explore(key, params, apply_fn, exp_coef, observations, n_actions)
    return jnp.squeeze(actions, axis=0)


def compute_qvalues(
    params: Params, apply_fn: Callable[..., Any], observations: jax.Array
):
    return apply_fn({"params": params}, observations)


def compute_max_qvalues(
    params: Params, apply_fn: Callable[..., Any], observations: jax.Array
):
    return jnp.max(
        compute_q_values(params, apply_fn, observations), axis=-1, keepdims=True
    )


def process_experience(
    params: Params,
    apply_fn: Callable,
    gamma: float,
    sample: list[OffPolicyExp],
):
    from rl.buffer import array_of_name

    next_observations = array_of_name(sample, "next_observation")
    next_qvalues = jax.jit(compute_max_qvalues, static_argnums=1)(
        params, apply_fn, next_observations
    )

    not_dones = 1.0 - array_of_name(sample, "done")[..., None]
    rewards = array_of_name(sample, "reward")[..., None]

    observations = np.array([exp.observation for exp in sample])
    returns = rewards + gamma * not_dones * next_qvalues
    return observations, returns


def update_step(
    state: train_state.TrainState,
    observations: jax.Array,
    returns: jax.Array,
):
    loss, grads = jax.value_and_grad(loss_fn)(
        state.params, state.apply_fn, observations, returns
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


class DQN(Base):
    def __init__(
        self,
        seed: int,
        env: gym.Env,
        config: ml_collections.ConfigDict,
        *,
        create_module: Callable[..., nn.Module] = create_module,
        create_params: Callable[..., Params] = create_params,
    ):
        Base.__init__(self, seed)
        self.config = config

        module = create_module(env.observation_space, env.action_space)
        params = create_params(self.nextkey(), module, env.observation_space)
        self.state = create_train_state(module, params, self.config)

    def select_action(self, observation: jax.Array) -> jax.Array:
        """unbatched"""
        action = jax.jit(explore_unbatched, static_argnums=(2))(
            self.nextkey(),
            self.state.params,
            self.state.apply_fn,
            NO_EXPLORATION,
            observation,
            self.config.action_space.n,
        )
        return ensure_int(action)

    def explore(self, observation: jax.Array):
        """unbatched"""
        action = jax.jit(explore_unbatched, static_argnums=(2))(
            self.nextkey(),
            self.state.params,
            self.state.apply_fn,
            self.config.exploration_coef,
            observation,
            self.config.action_space.n,
        )
        return ensure_int(action)

    def update(self, buffer: OffPolicyBuffer):
        sample = buffer.sample(self.config.batch_size)
        observations, returns = process_experience(
            self.state.params, self.state.apply_fn, self.config.gamma, sample
        )
        self.state, loss = jax.jit(update_step)(self.state, observations, returns)


def train(seed: int, dqn: DQN, env: gym.Env, n_env_steps: int, skip_steps: int):
    buffer = OffPolicyBuffer(seed, 10**4)

    observation, info = env.reset(seed=seed + 1)
    episode_return = 0.0

    for step in range(1, n_env_steps + 1):
        action = dqn.explore(observation)
        next_observation, reward, done, trunc, info = env.step(action)
        episode_return += reward

        buffer.add(
            OffPolicyExp(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                next_observation=next_observation,
            )
        )

        if done or trunc:
            print(step, " > ", episode_return)
            episode_return = 0.0
            next_observation, info = env.reset()

        if len(buffer) > dqn.config.batch_size and step % skip_steps == 0:
            dqn.update(buffer)

        observation = next_observation
