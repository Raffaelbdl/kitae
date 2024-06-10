"""Proximal Policy Optimization (PPO)"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import distrax as dx
import flax.linen as nn
from gymnasium import spaces
import jax
import jax.numpy as jnp
import optax

from kitae.agent import OnPolicyAgent
from kitae.buffers.buffer import Experience, batchify_and_randomize
from kitae.config import AlgoConfig, AlgoParams

from kitae.operations.loss import loss_policy_ppo, loss_value_clip
from kitae.operations.timesteps import calculate_gaes_targets
from kitae.loops.update import update_epoch

from kitae.modules.encoder import encoder_factory
from kitae.modules.modules import PassThrough, init_params
from kitae.modules.optimizer import linear_learning_rate_schedule
from kitae.modules.policy import (
    policy_output_factory,
    sample_and_log_prob,
    get_log_prob,
)
from kitae.modules.pytree import AgentPyTree, TrainState
from kitae.modules.value import ValueOutput

from kitae.types import Params, PRNGKeyArray, LossDict
from kitae.types import ExploreFn, ProcessExperienceFn, UpdateFn

PPO_tuple = namedtuple(
    "PPO_tuple",
    ["observation", "action", "log_prob", "gae", "target", "value"],
)


class PPOState(AgentPyTree):
    encoder_state: TrainState
    policy_state: TrainState
    value_state: TrainState


@dataclass
class PPOParams(AlgoParams):
    """Proximal Policy Optimization parameters."""

    gamma: float = 0.99
    _lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.1
    value_coef: float = 0.5
    normalize: bool = True


def train_state_ppo_factory(
    key: PRNGKeyArray,
    config: AlgoConfig,
    *,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> PPOState:

    key1, key2, key3 = jax.random.split(key, 3)
    observation_shape = config.env_cfg.observation_space.shape
    hidden_shape = (512,) if len(observation_shape) == 3 else (256,)
    action_shape = config.env_cfg.action_space.shape

    encoder = encoder_factory(
        config.env_cfg.observation_space,
        preprocess_fn=preprocess_fn,
    )
    policy_output = policy_output_factory(config.env_cfg.action_space)
    n_actions = (
        config.env_cfg.action_space.n
        if isinstance(config.env_cfg.action_space, spaces.Discrete)
        else action_shape[-1]
    )

    if config.update_cfg.shared_encoder:
        policy = policy_output(n_actions)
        value = ValueOutput()
        encoder = encoder()
        input_shape = hidden_shape
    else:
        policy = nn.Sequential([encoder(), policy_output(n_actions)])
        value = nn.Sequential([encoder(), ValueOutput()])
        encoder = PassThrough()
        input_shape = observation_shape

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

    policy_state = TrainState.create(
        apply_fn=jax.jit(policy.apply),
        params=init_params(key1, policy, [input_shape], tabulate),
        tx=tx,
    )
    value_state = TrainState.create(
        apply_fn=jax.jit(value.apply),
        params=init_params(key2, value, [input_shape], tabulate),
        tx=tx,
    )
    encoder_state = TrainState.create(
        apply_fn=jax.jit(encoder.apply),
        params=init_params(key3, encoder, [observation_shape], tabulate),
        tx=tx,
    )

    return PPOState(
        policy_state=policy_state, value_state=value_state, encoder_state=encoder_state
    )


def explore_factory(config: AlgoConfig) -> ExploreFn:

    def explore_fn(
        state: PPOState, key: PRNGKeyArray, observations: jax.Array
    ) -> tuple[jax.Array, jax.Array]:

        hiddens = state.encoder_state.apply_fn(state.encoder_state.params, observations)
        if not isinstance(hiddens, tuple):
            hiddens = (hiddens,)
        dists = state.policy_state.apply_fn(state.policy_state.params, *hiddens)
        actions, log_probs = sample_and_log_prob(dists, key)

        return actions, log_probs

    return jax.jit(explore_fn)


def process_experience_factory(config: AlgoConfig) -> ProcessExperienceFn:
    """Process experience PPO-style."""
    algo_params: PPOParams = config.algo_params

    def process_experience_fn(
        state: PPOState, key: PRNGKeyArray, experience: Experience
    ) -> PPOState:

        all_obs = jnp.concatenate(
            [experience.observation, experience.next_observation[-1:]], axis=0
        )
        all_hiddens = state.encoder_state.apply_fn(state.encoder_state.params, all_obs)
        if not isinstance(all_hiddens, tuple):
            all_hiddens = (all_hiddens,)
        all_values = state.value_state.apply_fn(state.value_state.params, *all_hiddens)

        values = all_values[:-1]
        next_values = all_values[1:]

        not_dones = 1.0 - experience.done[..., None]
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

        return PPO_tuple(
            observation=experience.observation,
            action=experience.action,
            log_prob=experience.log_prob,
            gae=gaes,
            target=targets,
            value=values,
        )

    return jax.jit(process_experience_fn)


def update_policy_value_factory(config: AlgoConfig) -> UpdateFn:
    algo_params: PPOParams = config.algo_params

    def update_policy_value_fn(
        state: PPOState, key: PRNGKeyArray, batch: PPO_tuple
    ) -> tuple[PPOState, LossDict]:

        def loss_fn(params):
            hiddens = state.encoder_state.apply_fn(params["encoder"], batch.observation)
            if not isinstance(hiddens, tuple):
                hiddens = (hiddens,)

            dists = state.policy_state.apply_fn(params["policy"], *hiddens)
            log_probs = get_log_prob(dists, batch.action)

            loss_policy, info_policy = loss_policy_ppo(
                dists,
                log_probs,
                batch.log_prob,
                batch.gae,
                algo_params.clip_eps,
                algo_params.entropy_coef,
            )

            values = state.value_state.apply_fn(params["value"], *hiddens)
            loss_value, info_value = loss_value_clip(
                values, batch.target, batch.value, algo_params.clip_eps
            )

            loss = loss_policy + algo_params.value_coef * loss_value
            info = info_policy | info_value
            info["total_loss"] = loss

            return loss, info

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            {
                "policy": state.policy_state.params,
                "value": state.value_state.params,
                "encoder": state.encoder_state.params,
            },
        )

        state.policy_state = state.policy_state.apply_gradients(grads=grads["policy"])
        state.value_state = state.value_state.apply_gradients(grads=grads["value"])
        state.encoder_state = state.encoder_state.apply_gradients(
            grads=grads["encoder"]
        )

        return state, info

    return jax.jit(update_policy_value_fn)


def update_step_factory(config: AlgoConfig) -> UpdateFn:

    update_policy_value_fn = update_policy_value_factory(config)

    def update_step_fn(
        state: PPOState, key: PRNGKeyArray, experience: PPO_tuple
    ) -> tuple[PPOState, LossDict]:

        return update_epoch(
            key,
            state,
            experience,
            batchify_and_randomize,
            update_policy_value_fn,
            experience_type=PPO_tuple,
            batch_size=config.update_cfg.batch_size,
        )

    return jax.jit(update_step_fn)


class PPO(OnPolicyAgent):
    """
    Proximal Policy Optimization (PPO)
    Paper : https://arxiv.org/abs/1707.06347
    Implementation details : https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """

    def __init__(
        self,
        run_name: str,
        config: AlgoConfig,
        *,
        preprocess_fn: Callable = None,
        tabulate: bool = False,
    ):
        super().__init__(
            run_name,
            config,
            train_state_ppo_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
            experience_type=Experience,
        )
