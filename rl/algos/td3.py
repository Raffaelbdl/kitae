"""Deep Deterministic Policy Gradient (TD3)"""

from dataclasses import dataclass
import functools
from typing import Callable

import chex
import distrax as dx
import flax.struct as struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.algos.general_fns import fn_parallel

from rl.base import Base, EnvType, EnvProcs, AlgoType
from rl.callbacks.callback import Callback
from rl.config import AlgoConfig, AlgoParams
from rl.types import Array, Params, GymEnv, EnvPoolEnv

from rl.buffer import Buffer, OffPolicyBuffer, OffPolicyExp, stack_experiences
from rl.loss import loss_mean_squared_error
from rl.modules.qvalue import (
    TrainStatePolicyQvalue,
    ParamsPolicyQValue,
)
from rl.train import train

from dx_tabulate import add_distrax_representers

add_distrax_representers()


class TrainState(TrainState):
    target_params: Params


@dataclass
class TD3Params(AlgoParams):
    """
    Deep Deterministic Policy Gradient parameters
    """

    gamma: float
    skip_steps: int
    tau: float
    action_noise: float
    policy_update_frequency: int
    target_noise_std: float
    target_noise_clip: float
    start_step: int


@chex.dataclass
class TD3TrainState:
    policy_state: TrainState
    qvalue_state: TrainState


def train_state_ddpg_factory(
    key: jax.Array,
    config: AlgoConfig,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> TD3TrainState:
    import flax.linen as nn
    from rl.modules.modules import init_params

    key1, key2 = jax.random.split(key)
    observation_shape = config.env_cfg.observation_space.shape
    action_shape = config.env_cfg.action_space.shape

    class Policy(nn.Module):
        num_outputs: int

        @nn.compact
        def __call__(self, x: jax.Array):
            x = nn.relu(nn.Dense(256)(x))
            x = nn.relu(nn.Dense(256)(x))
            return nn.tanh(nn.Dense(self.num_outputs)(x))

    module_policy = Policy(action_shape[-1])
    policy_state = TrainState.create(
        apply_fn=module_policy.apply,
        params=init_params(key1, module_policy, observation_shape, tabulate),
        target_params=init_params(key1, module_policy, observation_shape, False),
        tx=optax.adam(config.update_cfg.learning_rate),
    )

    class QValue(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array, a: jax.Array):
            def qvalue_fn(x, a):
                x = jnp.concatenate([x, a], axis=-1)
                x = nn.relu(nn.Dense(256)(x))
                x = nn.relu(nn.Dense(256)(x))
                return nn.Dense(1)(x)

            return qvalue_fn(x, a), qvalue_fn(x, a)

    module_qvalue = QValue()
    qvalue_state = TrainState.create(
        apply_fn=module_qvalue.apply,
        params=init_params(
            key2, module_qvalue, [observation_shape, action_shape], tabulate
        ),
        target_params=init_params(
            key2, module_qvalue, [observation_shape, action_shape], False
        ),
        tx=optax.adam(config.update_cfg.learning_rate),
    )

    return TD3TrainState(policy_state=policy_state, qvalue_state=qvalue_state)


def explore_factory(train_state: TD3TrainState, algo_params: TD3Params) -> Callable:
    # @jax.jit
    def fn(
        policy_state: TrainState,
        key: jax.Array,
        observations: jax.Array,
        action_noise: float,
    ):
        actions = jax.jit(policy_state.apply_fn)(
            {"params": policy_state.params}, observations
        )
        actions += jax.random.normal(key, actions.shape) * action_noise
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions, jnp.zeros_like(actions)

    return fn


def process_experience_factory(
    train_state: TD3TrainState,
    algo_params: TD3Params,
    vectorized: bool,
    parallel: bool,
) -> Callable:
    @jax.jit
    def fn(
        policy_state: TrainState,
        qvalue_state: TrainState,
        key: jax.Array,
        sample: list[OffPolicyExp],
    ):
        stacked = stack_experiences(sample)
        # stacked = sample

        return (
            stacked.observation,
            stacked.action,
            stacked.reward,
            stacked.done,
            stacked.next_observation,
        )

    return fn


def update_step_factory(train_state: TD3TrainState, config: AlgoConfig) -> Callable:
    qvalue_apply = train_state.qvalue_state.apply_fn
    policy_apply = train_state.policy_state.apply_fn

    @jax.jit
    def update_qvalue_fn(
        policy_state: TrainState,
        qvalue_state: TrainState,
        key: jax.Array,
        batch: tuple[jax.Array],
    ):
        observations, actions, rewards, dones, next_observations = batch

        next_actions = policy_apply(
            {"params": policy_state.target_params}, next_observations
        )
        noise = jnp.clip(
            jax.random.normal(key, next_actions.shape)
            * config.algo_params.target_noise_std,
            -config.algo_params.target_noise_clip,
            config.algo_params.target_noise_clip,
        )
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        next_q1, next_q2 = qvalue_apply(
            {"params": qvalue_state.target_params}, next_observations, next_actions
        )
        next_q_min = jnp.minimum(next_q1, next_q2)

        discounts = config.algo_params.gamma * (1.0 - dones[..., None])
        targets = rewards[..., None] + discounts * next_q_min

        def loss_fn(params: Params, observations, actions, targets):
            q1, q2 = qvalue_state.apply_fn({"params": params}, observations, actions)
            loss_q1 = loss_mean_squared_error(q1, targets)
            loss_q2 = loss_mean_squared_error(q2, targets)

            return loss_q1 + loss_q2, {"loss_qvalue": loss_q1 + loss_q2}

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            qvalue_state.params, observations, actions, targets
        )
        qvalue_state = qvalue_state.apply_gradients(grads=grads)

        return qvalue_state, loss, info

    @jax.jit
    def update_policy_fn(
        policy_state: TrainState, qvalue_state: TrainState, batch: tuple[jax.Array]
    ):
        def loss_fn(params: Params, observations):
            actions = policy_apply({"params": params}, observations)
            qvalues, _ = qvalue_apply(
                {"params": qvalue_state.params}, observations, actions
            )
            loss = -jnp.mean(qvalues)
            return loss, {"loss_policy": loss}

        observations, _, _, _, _ = batch
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            policy_state.params, observations
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

        return (policy_state, qvalue_state), loss, info

    def update_step_fn(
        policy_state: TrainState,
        qvalue_state: TrainState,
        key: jax.Array,
        batch: tuple[jax.Array],
        step: int,
    ):
        qvalue_state, loss_qvalue, info_qvalue = update_qvalue_fn(
            policy_state, qvalue_state, key, batch
        )

        if step % config.algo_params.policy_update_frequency == 0:
            (policy_state, qvalue_state), loss_policy, info_policy = update_policy_fn(
                policy_state, qvalue_state, batch
            )
        else:
            loss_policy = 0.0
            info_policy = {}

        info = info_qvalue | info_policy
        info["total_loss"] = loss_qvalue + loss_policy

        return qvalue_state, policy_state, info

    return update_step_fn


class TD3(Base):
    """
    Deep Deterministic Policy Gradient (TD3)
    Paper : https://arxiv.org/abs/1509.02971
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
        super().__init__(
            config,
            train_state_ddpg_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            run_name=run_name,
            tabulate=tabulate,
        )
        self.step = 0

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

        action, zeros = self.explore_fn(self.state.policy_state, keys, observation, 0.0)
        return action, zeros

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

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

    def should_update(self, step: int, buffer: Buffer) -> bool:
        self.step = step
        return (
            len(buffer) >= self.config.update_cfg.batch_size
            and step % self.algo_params.skip_steps == 0
            and step >= self.algo_params.start_step
        )

    def update(self, buffer: OffPolicyBuffer) -> dict:
        def fn(state: TD3TrainState, key: jax.Array, sample: tuple):
            key1, key2 = jax.random.split(key)
            experiences = self.process_experience_fn(
                state.policy_state, state.qvalue_state, key1, sample
            )
            state.qvalue_state, state.policy_state, info = self.update_step_fn(
                state.policy_state, state.qvalue_state, key2, experiences, self.step
            )
            return state, info

        sample = buffer.sample(self.config.update_cfg.batch_size)
        self.state, info = fn(self.state, self.nextkey(), sample)
        return info

    def train(
        self, env: GymEnv | EnvPoolEnv, n_env_steps: int, callbacks: list[Callback]
    ) -> None:
        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE if self.config.train_cfg.n_agents == 1 else EnvType.PARALLEL,
            EnvProcs.ONE if self.config.train_cfg.n_envs == 1 else EnvProcs.MANY,
            AlgoType.OFF_POLICY,
            saver=self.saver,
            callbacks=callbacks,
        )

    def resume(
        self, env: GymEnv | EnvPoolEnv, n_env_steps: int, callbacks: list[Callback]
    ) -> None:
        step, self.state = self.saver.restore_latest_step(self.state)

        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE if self.config.train_cfg.n_agents == 1 else EnvType.PARALLEL,
            EnvProcs.ONE if self.config.train_cfg.n_envs == 1 else EnvProcs.MANY,
            AlgoType.OFF_POLICY,
            start_step=step,
            saver=self.saver,
            callbacks=callbacks,
        )
