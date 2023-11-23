from typing import Callable, TypeVar

import chex
import distrax as dx
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
from rl.buffer import OffPolicyBuffer, OffPolicyExp
from rl.modules import modules_factory, create_params

NO_EXPLORATION = 0.0


def create_module(
    observation_space: spaces.Space,
    action_space: spaces.Discrete,
) -> tuple[nn.Module, nn.Module]:
    modules = modules_factory(observation_space, action_space, False)
    return modules["policy"]


def create_train_state(
    module: nn.Module, params: Params, config: ml_collections.ConfigDict
) -> train_state.TrainState:
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)


def loss_fn(params: Params, qvalue_fn: Callable, states: jax.Array, returns: jax.Array):
    q_values = qvalue_fn({"params": params}, states)
    loss = jnp.mean(jnp.square(q_values - returns))
    return loss, {"loss": loss}


def compute_q_values(
    params: Params, qvalue_fn: Callable, observations: jax.Array
) -> jax.Array:
    return qvalue_fn({"params": params}, observations)


def compute_max_qvalues(
    params: Params, apply_fn: Callable, observations: jax.Array
) -> jax.Array:
    return jnp.max(
        compute_q_values(params, apply_fn, observations), axis=-1, keepdims=True
    )


def explore(
    key: jax.Array,
    params: Params,
    apply_fn: Callable,
    exp_coef: float,
    observations: jax.Array,
) -> jax.Array:
    qvalues = apply_fn({"params": params}, observations)
    greedy_actions = jnp.argmax(qvalues, axis=-1, keepdims=False)

    eps = jrd.uniform(key, greedy_actions.shape)

    random_actions = jrd.randint(key, greedy_actions.shape, 0, qvalues.shape[-1])
    return jnp.where(eps <= exp_coef, random_actions, greedy_actions)


def explore_unbatched(
    key: jax.Array,
    params: Params,
    apply_fn: Callable,
    exp_coef: float,
    observation: jax.Array,
) -> jax.Array:
    observations = jnp.expand_dims(observation, axis=0)
    actions = explore(key, params, apply_fn, exp_coef, observations)
    return jnp.squeeze(actions, axis=0)


def process_experience(
    params: Params,
    qvalue_fn: Callable,
    gamma: float,
    sample: list[OffPolicyExp],
):
    from rl.buffer import array_of_name

    next_observations = array_of_name(sample, "next_observation")
    next_qvalues = jax.jit(compute_max_qvalues, static_argnums=1)(
        params, qvalue_fn, next_observations
    )

    not_dones = 1.0 - array_of_name(sample, "done")[..., None]
    rewards = array_of_name(sample, "reward")[..., None]

    observations = np.array([exp.observation for exp in sample])
    returns = rewards + gamma * not_dones * next_qvalues
    return observations, returns


def process_experience_vectorized(
    params: Params,
    qvalue_fn: Callable,
    gamma: float,
    sample: list[OffPolicyExp],
):
    from rl.buffer import array_of_name

    next_observations = array_of_name(sample, "next_observation")
    next_qvalues = jax.vmap(
        jax.jit(compute_max_qvalues, static_argnums=1),
        in_axes=(None, None, 1),
        out_axes=1,
    )(params, qvalue_fn, next_observations)

    not_dones = 1.0 - array_of_name(sample, "done")[..., None]
    rewards = array_of_name(sample, "reward")[..., None]

    observations = np.array([exp.observation for exp in sample])
    returns = rewards + gamma * not_dones * next_qvalues

    observations = jnp.reshape(observations, (-1, *observations.shape[2:]))
    returns = jnp.reshape(returns, (-1, *returns.shape[2:]))
    return observations, returns


def update_step(
    state: train_state.TrainState,
    observations: jax.Array,
    returns: jax.Array,
):
    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.apply_fn, observations, returns
    )
    state = state.apply_gradients(grads=grads)
    return state, loss, info


class DQN(Base):
    def __init__(
        self,
        seed: int,
        env: gym.Env,
        config: ml_collections.ConfigDict,
        *,
        create_module: Callable[..., nn.Module] = create_module,
        create_params: Callable[..., Params] = create_params,
        n_envs: int = 1,
    ):
        Base.__init__(self, seed)
        self.config = config

        qvalue = create_module(env.observation_space, env.action_space)
        params = create_params(self.nextkey(), qvalue, env.observation_space.shape)
        self.state = create_train_state(qvalue, params, self.config)

        self.explore_fn = explore if n_envs > 1 else explore_unbatched
        self.explore_fn = jax.jit(self.explore_fn, static_argnums=(2))

        self.process_experience_fn = (
            process_experience_vectorized if n_envs > 1 else process_experience
        )

        self.n_envs = n_envs

    def select_action(self, observation: jax.Array) -> jax.Array:
        action = self.explore_fn(
            self.nextkey(),
            self.state.params,
            self.state.apply_fn,
            NO_EXPLORATION,
            observation,
        )
        # action = jax.jit(explore_unbatched, static_argnums=(2))(
        #     self.nextkey(),
        #     self.state.params,
        #     self.state.apply_fn,
        #     NO_EXPLORATION,
        #     observation,
        # )
        return action

    def explore(self, observation: jax.Array):
        action = self.explore_fn(
            self.nextkey(),
            self.state.params,
            self.state.apply_fn,
            self.config.exploration_coef,
            observation,
        )
        # action = jax.jit(explore_unbatched, static_argnums=(2))(
        #     self.nextkey(),
        #     self.state.params,
        #     self.state.apply_fn,
        #     self.config.exploration_coef,
        #     observation,
        # )
        return action

    def update(self, buffer: OffPolicyBuffer):
        sample = buffer.sample(self.config.batch_size)
        observations, returns = self.process_experience_fn(
            self.state.params, self.state.apply_fn, self.config.gamma, sample
        )
        self.state, loss, info = jax.jit(update_step)(self.state, observations, returns)

        return loss


def train(seed: int, dqn: DQN, env: gym.Env, n_env_steps: int):
    assert dqn.n_envs == 1

    buffer = OffPolicyBuffer(seed, dqn.config.max_buffer_size)

    observation, info = env.reset(seed=seed + 1)
    episode_return = 0.0

    for step in range(1, n_env_steps + 1):
        action = dqn.explore(np.array(observation))
        next_observation, reward, done, trunc, info = env.step(int(action))
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

        if len(buffer) > dqn.config.batch_size and step % dqn.config.skip_steps == 0:
            dqn.update(buffer)

        observation = next_observation


def train_vectorized(seed: int, dqn: DQN, env: gym.Env, n_env_steps: int):
    assert dqn.n_envs > 1

    buffer = OffPolicyBuffer(seed, dqn.config.max_buffer_size)

    observation, info = env.reset()
    episode_return = np.zeros((observation.shape[0],))

    for step in range(1, n_env_steps + 1):
        action = dqn.explore(observation)
        next_observation, reward, done, trunc, info = env.step(np.array(action))
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

        for i, (d, t) in enumerate(zip(done, trunc)):
            if d or t:
                if i == 0:
                    print(step, " > ", episode_return[i])
                episode_return[i] = 0.0

        if len(buffer) > dqn.config.batch_size and step % dqn.config.skip_steps == 0:
            dqn.update(buffer)

        observation = next_observation
