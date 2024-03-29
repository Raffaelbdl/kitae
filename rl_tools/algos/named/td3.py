"""Deep Deterministic Policy Gradient (TD3)"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from rl_tools.base import OffPolicyAgent, ExperienceTransform
from rl_tools.config import AlgoConfig, AlgoParams
from rl_tools.types import Params

from rl_tools.buffer import OffPolicyBuffer, Experience
from rl_tools.loss import loss_mean_squared_error

from rl_tools.timesteps import compute_td_targets


from rl_tools.modules.encoder import encoder_factory
from rl_tools.modules.modules import init_params
from rl_tools.modules.train_state import PolicyQValueTrainState, TrainState
from rl_tools.modules.policy import PolicyNormalExternalStd
from rl_tools.modules.qvalue import make_double_q_value, qvalue_factory

TD3_tuple = namedtuple("TD3_tuple", ["observation", "action", "target"])


@dataclass
class TD3Params(AlgoParams):
    """TD3 parameters"""

    gamma: float
    skip_steps: int
    tau: float
    action_noise: float
    policy_update_frequency: int
    target_noise_std: float
    target_noise_clip: float
    start_step: int


def train_state_ddpg_factory(
    key: jax.Array,
    config: AlgoConfig,
    *,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> PolicyQValueTrainState:

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

    return PolicyQValueTrainState(policy_state=policy_state, qvalue_state=qvalue_state)


def explore_factory(config: AlgoConfig) -> Callable:
    @jax.jit
    def explore_fn(
        policy_state: TrainState,
        key: jax.Array,
        observations: jax.Array,
        action_noise: float,
    ):
        dists = policy_state.apply_fn(policy_state.params, observations, action_noise)
        actions, log_probs = dists.sample_and_log_prob(seed=key)
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions, log_probs

    return explore_fn


def process_experience_factory(config: AlgoConfig) -> Callable:
    algo_params = config.algo_params

    @jax.jit
    def process_experience_fn(
        td3_state: PolicyQValueTrainState,
        key: jax.Array,
        experience: Experience,
    ) -> tuple[jax.Array, ...]:
        next_actions = td3_state.policy_state.apply_fn(
            td3_state.policy_state.target_params,
            experience.next_observation,
            0.0,
        ).sample(seed=0)
        noise = jnp.clip(
            jax.random.normal(key, next_actions.shape) * algo_params.target_noise_std,
            -algo_params.target_noise_clip,
            algo_params.target_noise_clip,
        )
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        next_q1, next_q2 = td3_state.qvalue_state.apply_fn(
            td3_state.qvalue_state.target_params,
            experience.next_observation,
            next_actions,
        )
        next_q_min = jnp.minimum(next_q1, next_q2)

        discounts = algo_params.gamma * (1.0 - experience.done[..., None])
        targets = compute_td_targets(
            experience.reward[..., None], discounts, next_q_min
        )

        return (experience.observation, experience.action, targets)

    return process_experience_fn


def update_step_factory(config: AlgoConfig) -> Callable:
    @jax.jit
    def update_qvalue_fn(
        qvalue_state: TrainState, batch: TD3_tuple
    ) -> tuple[TrainState, dict]:

        def loss_fn(params: Params):
            q1, q2 = qvalue_state.apply_fn(params, batch.observation, batch.action)
            loss_q1 = loss_mean_squared_error(q1, batch.target)
            loss_q2 = loss_mean_squared_error(q2, batch.target)

            return loss_q1 + loss_q2, {"loss_qvalue": loss_q1 + loss_q2}

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            qvalue_state.params
        )
        qvalue_state = qvalue_state.apply_gradients(grads=grads)

        return qvalue_state, info

    @jax.jit
    def update_policy_fn(
        key: jax.Array,
        policy_state: TrainState,
        qvalue_state: TrainState,
        batch: TD3_tuple,
    ) -> tuple[TrainState, dict]:

        def loss_fn(params: Params):
            dists = policy_state.apply_fn(params, batch.observation, jnp.zeros(()))
            actions = dists.sample(seed=key)
            qvalues, _ = qvalue_state.apply_fn(
                qvalue_state.params, batch.observation, actions
            )
            loss = -jnp.mean(qvalues)
            return loss, {"loss_policy": loss}

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            policy_state.params
        )
        policy_state = policy_state.apply_gradients(grads=grads)

        policy_state = policy_state.replace(
            target_params=optax.incremental_update(
                policy_state.params,
                policy_state.target_params,
                config.algo_params.tau,
            )
        )
        qvalue_state = qvalue_state.replace(
            target_params=optax.incremental_update(
                qvalue_state.params,
                qvalue_state.target_params,
                config.algo_params.tau,
            )
        )

        return (policy_state, qvalue_state), info

    def update_step_fn(
        state: PolicyQValueTrainState,
        key: jax.Array,
        experiences: tuple[jax.Array, ...],
        should_update_policy: bool = True,
    ) -> tuple[PolicyQValueTrainState, dict]:
        batch = TD3_tuple(*experiences)

        state.qvalue_state, info = update_qvalue_fn(state.qvalue_state, batch)
        if should_update_policy:
            (state.policy_state, state.qvalue_state), info_policy = update_policy_fn(
                key, state.policy_state, state.qvalue_state, batch
            )
            info |= info_policy

        return state, info

    return update_step_fn


class TD3(OffPolicyAgent):
    """
    Deep Deterministic Policy Gradient (TD3)
    Paper : https://arxiv.org/abs/1509.02971
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
            train_state_ddpg_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
            experience_type=Experience,
        )

        self.algo_params = self.config.algo_params

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)

        action, zeros = self.explore_fn(self.state.policy_state, keys, observation, 0.0)
        return action, zeros

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)

        action, log_prob = self.explore_fn(
            self.state.policy_state,
            keys,
            observation,
            self.algo_params.action_noise,
        )
        if self.step < self.algo_params.start_step:
            action = jax.random.uniform(
                self.nextkey(), action.shape, minval=-1.0, maxval=1.0
            )
            log_prob = jnp.zeros_like(action)
        return action, log_prob

    def update(self, buffer: OffPolicyBuffer) -> dict:
        sample = buffer.sample(self.config.update_cfg.batch_size)

        experiences = self.process_experience_pipeline(
            [ExperienceTransform(self.process_experience_fn, self.state)],
            key=self.nextkey(),
            experiences=sample,
        )
        update_policy = self.step % self.config.algo_params.policy_update_frequency == 0

        for _ in range(self.config.update_cfg.n_epochs):
            self.state, info = self.update_step_fn(
                self.state, self.nextkey(), experiences, update_policy
            )

        return info
