"""Soft Actor Critic (SAC)"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jrd_extensions import PRNGSequence

from kitae.agent import OffPolicyAgent
from kitae.buffers.buffer import Experience
from kitae.config import AlgoConfig, AlgoParams

from kitae.operations.timesteps import compute_td_targets
from kitae.operations.transformation import action_clip

from kitae.modules.encoder import encoder_factory
from kitae.modules.modules import init_params, IndependentVariable
from kitae.modules.policy import make_policy, PolicyTanhNormal
from kitae.modules.pytree import AgentPyTree, TrainState
from kitae.modules.qvalue import make_double_q_value, qvalue_factory

from kitae.types import Params, PRNGKeyArray, LossDict
from kitae.types import ExploreFn, ProcessExperienceFn, UpdateFn

SAC_tuple = namedtuple("SAC_tuple", ["observation", "action", "target"])


class SACState(AgentPyTree):
    policy_state: TrainState
    qvalue_state: TrainState
    alpha_state: TrainState


@dataclass
class SACParams(AlgoParams):
    """Soft Actor Critic parameters."""

    gamma: float = 0.99
    tau: float = 0.005

    log_std_min: float = -20
    log_std_max: float = 5
    initial_alpha: float = 0.1

    skip_steps: int = 1
    start_step: int = -1


def train_state_sac_factory(
    key: PRNGKeyArray,
    config: AlgoConfig,
    *,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> SACState:

    key1, key2, key3 = jax.random.split(key, 3)
    observation_shape = config.env_cfg.observation_space.shape
    action_shape = config.env_cfg.action_space.shape

    policy = make_policy(
        encoder_factory(config.env_cfg.observation_space)(),
        PolicyTanhNormal(
            action_shape[-1],
            config.algo_params.log_std_min,
            config.algo_params.log_std_max,
        ),
    )
    policy_state = TrainState.create(
        apply_fn=jax.jit(policy.apply),
        params=init_params(key1, policy, [observation_shape], tabulate),
        target_params=init_params(key1, policy, [observation_shape], False),
        tx=optax.adam(config.update_cfg.learning_rate),
    )

    qvalue = make_double_q_value(
        qvalue_factory(config.env_cfg.observation_space, config.env_cfg.action_space)(),
        qvalue_factory(config.env_cfg.observation_space, config.env_cfg.action_space)(),
    )
    qvalue_state = TrainState.create(
        apply_fn=jax.jit(qvalue.apply),
        params=init_params(key2, qvalue, [observation_shape, action_shape], tabulate),
        target_params=init_params(
            key2, qvalue, [observation_shape, action_shape], False
        ),
        tx=optax.adam(config.update_cfg.learning_rate),
    )

    alpha = IndependentVariable(
        name="log_alpha",
        init_fn=nn.initializers.constant(jnp.log(config.algo_params.initial_alpha)),
        shape=(),
    )
    alpha_state = TrainState.create(
        apply_fn=jax.jit(alpha.apply),
        params=init_params(key3, alpha, [], tabulate),
        tx=optax.adam(config.update_cfg.learning_rate),
    )

    return SACState(
        policy_state=policy_state,
        qvalue_state=qvalue_state,
        alpha_state=alpha_state,
    )


def explore_factory(config: AlgoConfig) -> ExploreFn:

    def explore_fn(
        state: SACState, key: PRNGKeyArray, observations: jax.Array
    ) -> tuple[jax.Array, jax.Array]:

        dists = state.policy_state.apply_fn(state.policy_state.params, observations)
        actions, log_probs = dists.sample_and_log_prob(seed=key)
        actions = action_clip(actions, config.env_cfg.action_space)

        return actions, log_probs

    return jax.jit(explore_fn)


def process_experience_factory(config: AlgoConfig) -> ProcessExperienceFn:
    algo_params: SACParams = config.algo_params

    def process_experience_fn(
        state: SACState, key: PRNGKeyArray, experience: Experience
    ) -> SAC_tuple:

        next_dists = state.policy_state.apply_fn(
            state.policy_state.target_params, experience.next_observation
        )
        next_actions, next_log_probs = next_dists.sample_and_log_prob(seed=key)
        next_log_probs = jnp.sum(next_log_probs, axis=-1, keepdims=True)

        next_q1, next_q2 = state.qvalue_state.apply_fn(
            state.qvalue_state.target_params,
            experience.next_observation,
            next_actions,
        )
        next_q_min = jnp.minimum(next_q1, next_q2)

        discounts = algo_params.gamma * (1.0 - experience.done[..., None])
        targets = compute_td_targets(
            experience.reward[..., None], discounts, next_q_min
        )

        alpha = jnp.exp(state.alpha_state.apply_fn(state.alpha_state.params))
        targets -= discounts * alpha * next_log_probs

        return SAC_tuple(experience.observation, experience.action, targets)

    return jax.jit(process_experience_fn)


def update_qvalue_factory(config: AlgoConfig) -> UpdateFn:

    def update_qvalue_fn(
        state: SACState, key: PRNGKeyArray, batch: SAC_tuple
    ) -> tuple[SACState, LossDict]:

        def loss_fn(params: Params):
            q1, q2 = state.qvalue_state.apply_fn(
                params, batch.observation, batch.action
            )
            loss_q1 = jnp.mean(optax.l2_loss(q1, batch.target))
            loss_q2 = jnp.mean(optax.l2_loss(q2, batch.target))

            return loss_q1 + loss_q2, {"loss_qvalue": loss_q1 + loss_q2}

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.qvalue_state.params
        )
        state.qvalue_state = state.qvalue_state.apply_gradients(grads=grads)

        return state, info

    return jax.jit(update_qvalue_fn)


def update_policy_factory(config: AlgoConfig) -> UpdateFn:
    algo_params: SACParams = config.algo_params

    def update_policy_fn(
        state: SACState, key: PRNGKeyArray, batch: SAC_tuple
    ) -> tuple[SACState, LossDict]:

        def loss_fn(params: Params):
            dists = state.policy_state.apply_fn(params, batch.observation)
            actions, log_probs = dists.sample_and_log_prob(seed=key)
            log_probs = jnp.sum(log_probs, axis=-1)

            qvalues, _ = state.qvalue_state.apply_fn(
                state.qvalue_state.params, batch.observation, actions
            )
            alpha = jnp.exp(state.alpha_state.apply_fn(state.alpha_state.params))
            loss = jnp.mean(alpha * log_probs - qvalues)

            return loss, {"loss_policy": loss, "entropy": -jnp.mean(log_probs)}

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.policy_state.params
        )
        state.policy_state = state.policy_state.apply_gradients(grads=grads)

        state.policy_state = state.policy_state.replace(
            target_params=optax.incremental_update(
                state.policy_state.params,
                state.policy_state.target_params,
                algo_params.tau,
            )
        )
        state.qvalue_state = state.qvalue_state.replace(
            target_params=optax.incremental_update(
                state.qvalue_state.params,
                state.qvalue_state.target_params,
                algo_params.tau,
            )
        )

        return state, info

    return jax.jit(update_policy_fn)


def update_alpha_factory(config: AlgoConfig) -> UpdateFn:
    algo_params: SACParams = config.algo_params

    def update_apha_fn(
        state: SACState, key: PRNGKeyArray, batch: SAC_tuple
    ) -> tuple[SACState, LossDict]:

        dists = state.policy_state.apply_fn(
            state.policy_state.params, batch.observation
        )
        _, log_probs = dists.sample_and_log_prob(seed=key)

        def loss_fn(params: Params):
            alpha = jnp.exp(state.alpha_state.apply_fn(params))
            loss = alpha * -jnp.mean(log_probs + algo_params.target_entropy)
            return loss, {"loss_alpha": loss, "alpha": alpha}

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.alpha_state.params
        )
        state.alpha_state = state.alpha_state.apply_gradients(grads=grads)
        return state, info

    return jax.jit(update_apha_fn)


def update_step_factory(config: AlgoConfig) -> UpdateFn:

    update_qvalue_fn = update_qvalue_factory(config)
    update_policy_fn = update_policy_factory(config)
    update_alpha_fn = update_alpha_factory(config)

    def update_step_fn(
        state: SACState, key: PRNGKeyArray, batch: SAC_tuple
    ) -> tuple[SACState, LossDict]:
        rng = PRNGSequence(key)

        state, info_qvalue = update_qvalue_fn(state, next(rng), batch)
        state, info_policy = update_policy_fn(state, next(rng), batch)
        state, info_alpha = update_alpha_fn(state, next(rng), batch)

        info = info_qvalue | info_policy | info_alpha
        return state, info

    return jax.jit(update_step_fn)


class SAC(OffPolicyAgent):
    """Soft Actor Crtic (SAC)
    Paper: https://arxiv.org/abs/1812.05905
    """

    def __init__(
        self,
        run_name: str,
        config: AlgoConfig,
        *,
        preprocess_fn: Callable = None,
        tabulate: bool = False,
    ):
        config.algo_params.target_entropy = -config.env_cfg.action_space.shape[-1] / 2
        self.algo_params: SACParams = config.algo_params

        super().__init__(
            run_name,
            config,
            train_state_sac_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
            experience_type=Experience,
        )

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)
        action, log_prob = self.explore_fn(self.state, keys, observation)
        return action, log_prob

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)
        action, log_prob = self.explore_fn(self.state, keys, observation)

        if self.step < self.algo_params.start_step:
            action = jax.random.uniform(
                self.nextkey(),
                action.shape,
                minval=self.config.env_cfg.action_space.low,
                maxval=self.config.env_cfg.action_space.high,
            )
            log_prob = jnp.zeros_like(action)

        return action, log_prob
