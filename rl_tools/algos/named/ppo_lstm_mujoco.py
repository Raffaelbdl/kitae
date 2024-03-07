"""Proximal Policy Optimization (PPO)"""

from dataclasses import dataclass
from typing import Callable

import chex
import distrax as dx
import flax.linen as nn
from gymnasium import spaces
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jrd_extensions import PRNGSequence

from rl_tools.base import OnPolicyAgent
from rl_tools.config import AlgoConfig, AlgoParams

from rl_tools.buffer import Experience, batchify_and_randomize
from rl_tools.distribution import get_log_probs
from rl_tools.loss import loss_policy_ppo, loss_value_clip
from rl_tools.timesteps import calculate_gaes_targets


from rl_tools.algos.factory import AlgoFactory
from rl_tools.modules.modules import PassThrough, init_params
from rl_tools.modules.optimizer import linear_learning_rate_schedule
from rl_tools.modules.policy import policy_output_factory
from rl_tools.modules.train_state import PolicyValueTrainState, TrainState
from rl_tools.modules.value import ValueOutput


@chex.dataclass
class PPOTrainState:
    policy_state: TrainState
    value_state: TrainState


@dataclass
class PPOParams(AlgoParams):
    """
    Proximal Policy Optimization parameters.

    Parameters:
        gamma: The discount factor.
        _lambda: The factor for Generalized Advantage Estimator.
        clip_eps: The clipping range for update.
        entropy_coef: The loss coefficient of the entropy loss.
        value_coef: The loss coefficient of the value loss.
        normalize:  If true, advantages are normalized.
    """

    gamma: float
    _lambda: float
    clip_eps: float
    entropy_coef: float
    value_coef: float
    normalize: bool


class PolicyLSTM(nn.Module):
    action_space: spaces.Box
    n_actions: int
    rng_collection: str = "lstm"

    def setup(self):
        self.dense1 = nn.Dense(256)
        self.dense2 = nn.Dense(256)
        self.lstm = nn.OptimizedLSTMCell(256)
        self.policy_output = policy_output_factory(self.action_space)(self.n_actions)

    def __call__(
        self,
        inputs: jax.Array,  # n_E, S
        dones: jax.Array,  # n_E, 1
        last_state: jax.Array | None = None,
    ):
        x = nn.relu(self.dense1(inputs))
        x = nn.relu(self.dense2(x))

        if last_state is None:
            rng = self.make_rng(self.rng_collection)
            last_state, _ = self.lstm.initialize_carry(rng, x.shape)

        # re-init where done
        last_carry = (
            (1.0 - dones) * last_state,
            (1.0 - dones) * x,
        )

        (state, _), outputs = self.lstm(last_carry, x)
        dist = self.policy_output(outputs)

        return dist, state


class ValueLSTM(nn.Module):
    rng_collection: str = "lstm"

    def setup(self):
        self.dense1 = nn.Dense(256)
        self.dense2 = nn.Dense(256)
        self.lstm = nn.OptimizedLSTMCell(256)
        self.value_output = ValueOutput()

    def __call__(
        self,
        inputs: jax.Array,  # n_E, S
        dones: jax.Array,  # n_E, 1
        last_state: jax.Array | None = None,
    ):
        x = nn.relu(self.dense1(inputs))
        x = nn.relu(self.dense2(x))

        if last_state is None:
            rng = self.make_rng(self.rng_collection)
            last_state, _ = self.lstm.initialize_carry(rng, x.shape)

        # re-init where done
        last_carry = (
            (1.0 - dones) * last_state,
            (1.0 - dones) * x,
        )

        (state, _), outputs = self.lstm(last_carry, x)
        value = self.value_output(outputs)

        return value, state


def train_state_ppo_factory(
    key: jax.Array, config: AlgoConfig, *, tabulate: bool = False, **kwargs
) -> PPOTrainState:

    rng = PRNGSequence(key)
    observation_shape = config.env_cfg.observation_space.shape
    action_shape = config.env_cfg.action_space.shape

    learning_rate = config.update_cfg.learning_rate
    if config.update_cfg.learning_rate_annealing:
        learning_rate = linear_learning_rate_schedule(
            learning_rate,
            0.0,
            n_envs=config.env_cfg.n_envs,
            n_env_steps=config.train_cfg.n_env_steps,
            max_buffer_size=config.update_cfg.max_buffer_size,
            batch_size=config.update_cfg.batch_size,
            num_epochs=config.update_cfg.n_epochs,
        )
    tx = optax.chain(
        optax.clip_by_global_norm(config.update_cfg.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    def initialize_lstm(module: nn.Module) -> TrainState:
        params = init_params(
            {"params": next(rng), "lstm": next(rng)},
            module,
            [observation_shape, (1, 1)],
            tabulate,
        )
        return TrainState.create(apply_fn=jax.jit(module.apply), params=params, tx=tx)

    n_actions = (
        config.env_cfg.action_space.n
        if isinstance(config.env_cfg.action_space, spaces.Discrete)
        else action_shape[-1]
    )
    return PPOTrainState(
        policy_state=initialize_lstm(
            PolicyLSTM(config.env_cfg.action_space, n_actions)
        ),
        value_state=initialize_lstm(ValueLSTM()),
    )


def explore_factory(config: AlgoConfig) -> Callable:
    @jax.jit
    def explore_fn(
        policy_state: TrainState,
        key: jax.Array,
        observations: jax.Array,
        dones: jax.Array,
        state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        key1, key2 = jax.random.split(key, 2)
        dists, state = policy_state.apply_fn(
            policy_state.params,
            observations,
            dones,
            state,
            rngs={"lstm": key1},
        )
        return dists.sample_and_log_prob(seed=key2), state

    return explore_fn


def process_experience_factory(config: AlgoConfig) -> Callable:
    algo_params = config.algo_params

    @jax.jit
    def fn(ppo_state: PPOTrainState, key: jax.Array, experience: Experience):
        all_obs = jnp.concatenate(
            [experience.observation, experience.next_observation[-1:]], axis=0
        )
        all_dones = jnp.concatenate([experience.done, experience.done[-1:]], axis=0)

        def compute_value(carry, inputs: tuple[jax.Array, jax.Array]):
            state, k = carry
            observation, done = inputs
            k, _key = jax.random.split(k, 2)
            value, state = ppo_state.value_state.apply_fn(
                ppo_state.value_state.params,
                observation,
                done,
                state,
                rngs={"lstm": _key},
            )
            return (state, k), value[0]  # remove batch

        # init state, find how to skip this line
        (state, key), _ = compute_value((None, key), (all_obs[:1], all_dones[:1]))
        _, all_values = jax.lax.scan(compute_value, (state, key), (all_obs, all_dones))

        values = all_values[:-1]
        next_values = all_values[1:]

        dones = experience.done[..., None]
        not_dones = 1.0 - dones
        discounts = algo_params.gamma * not_dones

        rewards = experience.reward[..., None]
        gaes, targets = calculate_gaes_targets(
            values,
            next_values,
            discounts,
            rewards,
            algo_params._lambda,
            algo_params.normalize,
        )

        return (
            experience.observation,
            experience.action,
            dones,
            experience.log_prob,
            gaes,
            targets,
            values,
        )

    return fn


def update_step_factory(config: AlgoConfig) -> Callable:
    algo_params = config.algo_params

    def update_policy_value_fn(
        ppo_state: PolicyValueTrainState, key: jax.Array, batch: tuple[jax.Array, ...]
    ):

        observations, actions, dones, log_probs_old, gaes, targets, values_old = batch
        key, key_policy, key_value = jax.random.split(key, 3)

        def loss_fn(params):

            def compute_policy(carry, inputs: tuple[jax.Array, jax.Array]):
                state, k = carry
                observation, done = inputs
                k, _key = jax.random.split(key, 2)
                dist, state = ppo_state.policy_state.apply_fn(
                    params["policy"], observation, done, state, rngs={"lstm": _key}
                )
                return (state, k), dist

            # init state, find how to skip this line
            (state, _), _ = compute_policy(
                (None, key_policy), (observations[:1], dones[:1])
            )
            _, dists = jax.lax.scan(
                compute_policy, (state, key_policy), (observations, dones)
            )

            if isinstance(config.env_cfg.action_space, spaces.Discrete):
                dists = dx.Categorical(jnp.squeeze(dists.logits, axis=1))
            elif isinstance(config.env_cfg.action_space, spaces.Box):
                dists = dx.Normal(
                    jnp.squeeze(dists.loc, axis=1), jnp.squeeze(dists.scale, axis=1)
                )
            log_probs, _log_probs_old = get_log_probs(dists, actions, log_probs_old)
            loss_policy, info_policy = loss_policy_ppo(
                dists,
                log_probs,
                _log_probs_old,
                gaes,
                algo_params.clip_eps,
                algo_params.entropy_coef,
            )

            def compute_value(carry, inputs: tuple[jax.Array, jax.Array]):
                state, k = carry
                observation, done = inputs
                k, _key = jax.random.split(key, 2)
                value, state = ppo_state.value_state.apply_fn(
                    params["value"], observation, done, state, rngs={"lstm": _key}
                )
                return (state, k), value[0]

            # init state, find how to skip this line
            (state, _), _ = compute_value(
                (None, key_value), (observations[:1], dones[:1])
            )
            _, values = jax.lax.scan(
                compute_value, (state, key_value), (observations, dones)
            )

            loss_value, info_value = loss_value_clip(
                values, targets, values_old, algo_params.clip_eps
            )

            loss = loss_policy + algo_params.value_coef * loss_value
            info = info_policy | info_value
            info["total_loss"] = loss

            return loss, info

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            {
                "policy": ppo_state.policy_state.params,
                "value": ppo_state.value_state.params,
            },
        )

        ppo_state.policy_state = ppo_state.policy_state.apply_gradients(
            grads=grads["policy"]
        )
        ppo_state.value_state = ppo_state.value_state.apply_gradients(
            grads=grads["value"]
        )

        return ppo_state, loss, info

    @jax.jit
    def update_step_fn(
        ppo_state: PolicyValueTrainState,
        key: jax.Array,
        experiences: tuple[jax.Array, ...],
    ):
        batches = batchify_and_randomize(key, experiences, config.update_cfg.batch_size)

        loss = 0.0
        for batch in zip(*batches):
            key, _key = jax.random.split(key, 2)
            ppo_state, l, info = update_policy_value_fn(ppo_state, _key, batch)
            loss += l
        loss /= len(batches[0])

        return ppo_state, info

    return update_step_fn


class PPO(OnPolicyAgent):
    """
    Proximal Policy Optimization (PPO)
    Paper : https://arxiv.org/abs/1707.06347
    Implementation details : https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """

    def __init__(
        self,
        config: AlgoConfig,
        *,
        rearrange_pattern: str = "b h w c -> b h w c",
        preprocess_fn: Callable = None,
        run_name: str = None,
        tabulate: bool = False,
    ):
        AlgoFactory.intialize(
            self,
            config,
            train_state_ppo_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            run_name=run_name,
            tabulate=tabulate,
            experience_type=Experience,
        )
        self.lstm_state: jax.Array = None

    def select_action(
        self, observation: jax.Array, done: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return self.explore(observation, done)

    def explore(
        self, observation: jax.Array, done: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)
        (action, log_prob), self.lstm_state = self.explore_fn(
            self.state.policy_state, keys, observation, done[..., None], self.lstm_state
        )

        return np.array(action), log_prob
