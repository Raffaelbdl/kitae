"""Deep Deterministic Policy Gradient (TD3)"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jrd_extensions import PRNGSequence

from kitae.agent import OffPolicyAgent
from kitae.algos.experience import ExperiencePipeline
from kitae.buffer import OffPolicyBuffer, Experience, numpy_stack_experiences
from kitae.config import AlgoConfig, AlgoParams

from kitae.operations.timesteps import compute_td_targets
from kitae.operations.transformation import action_clip

from kitae.modules.encoder import encoder_factory
from kitae.modules.modules import init_params
from kitae.modules.policy import PolicyNormalExternalStd
from kitae.modules.pytree import AgentPyTree, TrainState
from kitae.modules.qvalue import make_double_q_value, qvalue_factory

from kitae.types import Params, PRNGKeyArray, LossDict
from kitae.types import ExploreFn, ProcessExperienceFn, UpdateFn

TD3_tuple = namedtuple("TD3_tuple", ["observation", "action", "target"])


class TD3State(AgentPyTree):
    policy_state: TrainState
    qvalue_state: TrainState


@dataclass
class TD3Params(AlgoParams):
    """TD3 parameters."""

    gamma: float = 0.99
    tau: float = 0.005
    action_noise: float = 0.1

    policy_update_frequency: int = 1
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5

    skip_steps: int = 1
    start_step: int = -1


def train_state_ddpg_factory(
    key: PRNGKeyArray,
    config: AlgoConfig,
    *,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> TD3State:

    key1, key2 = jax.random.split(key)
    observation_shape = config.env_cfg.observation_space.shape
    action_shape = config.env_cfg.action_space.shape
    action_scale = (
        config.env_cfg.action_space.high - config.env_cfg.action_space.low
    ) / 2.0
    action_bias = (
        config.env_cfg.action_space.high + config.env_cfg.action_space.low
    ) / 2.0

    class Policy(nn.Module):
        def setup(self) -> None:
            self.encoder = encoder_factory(config.env_cfg.observation_space)()
            self.output = PolicyNormalExternalStd(
                action_shape[-1], action_scale, action_bias
            )

        def __call__(self, x: jax.Array, policy_noise: float):
            return self.output(self.encoder(x), policy_noise)

    policy = Policy()
    policy_state = TrainState.create(
        apply_fn=jax.jit(policy.apply),
        params=init_params(key1, policy, [observation_shape, ()], tabulate),
        target_params=init_params(key1, policy, [observation_shape, ()], False),
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

    return TD3State(policy_state=policy_state, qvalue_state=qvalue_state)


def explore_factory(config: AlgoConfig) -> ExploreFn:

    def explore_fn(
        state: TD3State,
        key: PRNGKeyArray,
        observations: jax.Array,
        *,
        action_noise: float,
    ):
        dists = state.policy_state.apply_fn(
            state.policy_state.params, observations, action_noise
        )
        actions, log_probs = dists.sample_and_log_prob(seed=key)
        actions = action_clip(actions, config.env_cfg.action_space)

        return actions, log_probs

    return jax.jit(explore_fn)


def process_experience_factory(config: AlgoConfig) -> ProcessExperienceFn:
    algo_params: TD3Params = config.algo_params

    def process_experience_fn(
        state: TD3State, key: PRNGKeyArray, experience: Experience
    ) -> TD3_tuple:
        dists = state.policy_state.apply_fn(
            state.policy_state.target_params,
            experience.next_observation,
            0.0,
        )
        next_actions = dists.sample(seed=key)

        noise = jnp.clip(
            jax.random.normal(key, next_actions.shape) * algo_params.target_noise_std,
            -algo_params.target_noise_clip,
            algo_params.target_noise_clip,
        )
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

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

        return TD3_tuple(experience.observation, experience.action, targets)

    return jax.jit(process_experience_fn)


def update_qvalue_factory(config: AlgoConfig) -> UpdateFn:

    def update_qvalue_fn(
        state: TD3State, key: PRNGKeyArray, batch: TD3_tuple
    ) -> tuple[TD3State, LossDict]:

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
    algo_params: TD3Params = config.algo_params

    def update_policy_fn(
        state: TD3State, key: PRNGKeyArray, batch: TD3_tuple
    ) -> tuple[TD3State, LossDict]:

        def loss_fn(params: Params):
            dists = state.policy_state.apply_fn(
                params, batch.observation, jnp.zeros(())
            )
            actions = dists.sample(seed=key)

            qvalues, _ = state.qvalue_state.apply_fn(
                state.qvalue_state.params, batch.observation, actions
            )
            loss = -jnp.mean(qvalues)

            return loss, {"loss_policy": loss}

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


def update_step_factory(config: AlgoConfig) -> UpdateFn:

    update_qvalue_fn = update_qvalue_factory(config)
    update_policy_fn = update_policy_factory(config)

    def update_step_fn(
        state: TD3State,
        key: PRNGKeyArray,
        batch: TD3_tuple,
        *,
        should_update_policy: bool = True,
    ) -> tuple[TD3State, LossDict]:
        rng = PRNGSequence(key)

        state, info = update_qvalue_fn(state, next(rng), batch)

        if should_update_policy:
            state, info_policy = update_policy_fn(state, next(rng), batch)
            info |= info_policy

        return state, info

    return jax.jit(update_step_fn, static_argnames=("should_update_policy"))


def process_and_update_factory(
    config: AlgoConfig,
    update_step_factory: Callable,
    experience_pipeline: ExperiencePipeline,
) -> Callable:
    update_step_fn = update_step_factory(config)

    def process_and_update(
        state: AgentPyTree,
        key: PRNGKeyArray,
        sample: Experience,
        *,
        should_update_policy: bool,
    ) -> tuple[AgentPyTree, dict]:
        rng = PRNGSequence(key)

        experience = experience_pipeline.run(state, next(rng), sample)

        for _ in range(config.update_cfg.n_epochs):
            state, info = update_step_fn(
                state, next(rng), experience, should_update_policy=should_update_policy
            )

        return state, info

    return jax.jit(process_and_update, static_argnames=("should_update_policy"))


class TD3(OffPolicyAgent):
    """
    Deep Deterministic Policy Gradient (TD3)
    Paper: https://arxiv.org/abs/1509.02971
    """

    def __init__(
        self,
        run_name: str,
        config: AlgoConfig,
        *,
        preprocess_fn: Callable = None,
        tabulate: bool = False,
    ):
        self.algo_params: TD3Params = config.algo_params

        super().__init__(
            run_name,
            config,
            train_state_ddpg_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
            experience_type=Experience,
        )

        self.process_and_update_fn = process_and_update_factory(
            self.config, update_step_factory, self.experience_pipeline
        )

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)
        action, log_prob = self.explore_fn(self.state, keys, observation)
        return action, log_prob

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)
        action, log_prob = self.explore_fn(
            self.state, keys, observation, action_noise=self.algo_params.action_noise
        )

        if self.step < self.algo_params.start_step:
            action = jax.random.uniform(
                self.nextkey(),
                action.shape,
                minval=self.config.env_cfg.action_space.low,
                maxval=self.config.env_cfg.action_space.high,
            )
            log_prob = jnp.zeros_like(action)

        return action, log_prob

    def update(self, buffer: OffPolicyBuffer) -> dict:
        sample = buffer.sample(self.config.update_cfg.batch_size)
        sample = numpy_stack_experiences(sample)
        update_policy = self.step % self.config.algo_params.policy_update_frequency == 0
        self.state, info = self.process_and_update_fn(
            self.state, self.nextkey(), sample, should_update_policy=update_policy
        )
        return info

        experience = jax.jit(self.experience_pipeline.run)(
            self.state, self.nextkey(), sample
        )

        for _ in range(self.config.update_cfg.n_epochs):
            self.state, info = self.update_step_fn(
                self.state,
                self.nextkey(),
                experience,
                should_update_policy=update_policy,
            )

        return info
