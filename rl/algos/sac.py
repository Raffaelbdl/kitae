"""Soft Actor Critic (SAC)"""

from dataclasses import dataclass
import functools
from typing import Callable

import chex
import distrax as dx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.base import Base, EnvType, EnvProcs, AlgoType
from rl.callbacks.callback import Callback
from rl.config import AlgoConfig, AlgoParams
from rl.types import Array, Params, GymEnv, EnvPoolEnv

from rl.buffer import Buffer, OffPolicyBuffer, Experience
from rl.loss import loss_mean_squared_error

from rl.train import train
from rl.timesteps import compute_td_targets


from rl.modules.modules import TrainState
from rl.modules.policy import PolicyNormalExternalStd, PolicyTanhNormal
from rl.modules.policy_qvalue import TrainStatePolicyQValueTemperature


@dataclass
class SACParams(AlgoParams):
    """SAC parameters"""

    gamma: float
    tau: float

    log_std_min: float
    log_std_max: float

    initial_temperature: float  # log
    start_step: int
    skip_steps: int


def train_state_sac_factory(
    key: jax.Array,
    config: AlgoConfig,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> TrainStatePolicyQValueTemperature:
    import flax.linen as nn
    from rl.modules.encoder import encoder_factory
    from rl.modules.modules import init_params

    key1, key2, key3 = jax.random.split(key, 3)
    observation_shape = config.env_cfg.observation_space.shape
    action_shape = config.env_cfg.action_space.shape

    class Policy(nn.Module):
        def setup(self) -> None:
            self.encoder = encoder_factory(config.env_cfg.observation_space)()
            self.output = PolicyTanhNormal(
                action_shape[-1],
                config.algo_params.log_std_min,
                config.algo_params.log_std_max,
            )

        def __call__(self, x: jax.Array) -> dx.Distribution:
            return self.output(self.encoder(x))

    module_policy = Policy()
    policy_state = TrainState.create(
        apply_fn=module_policy.apply,
        params=init_params(key1, module_policy, [observation_shape], tabulate),
        target_params=init_params(key1, module_policy, [observation_shape], False),
        tx=optax.adam(config.update_cfg.learning_rate),
    )

    from rl.modules.qvalue import qvalue_factory

    class QValue(nn.Module):
        def setup(self) -> None:
            self.q1 = qvalue_factory(
                config.env_cfg.observation_space, config.env_cfg.action_space
            )()
            self.q2 = qvalue_factory(
                config.env_cfg.observation_space, config.env_cfg.action_space
            )()

        def __call__(self, x: jax.Array, a: jax.Array):
            return self.q1(x, a), self.q2(x, a)

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

    class Temperature(nn.Module):
        initial_temperature: float

        @nn.compact
        def __call__(self) -> jax.Array:
            log_temp = self.param(
                "log_temp",
                nn.initializers.constant(jnp.log(self.initial_temperature)),
                (),
            )
            return jnp.exp(log_temp)

    module_temperature = Temperature(config.algo_params.initial_temperature)
    temperature_params = module_temperature.init(key3)["params"]
    temperature_state = TrainState.create(
        apply_fn=module_temperature.apply,
        params=temperature_params,
        target_params=temperature_params,
        tx=optax.adam(config.update_cfg.learning_rate),
    )

    return TrainStatePolicyQValueTemperature(
        policy_state=policy_state,
        qvalue_state=qvalue_state,
        temperature_state=temperature_state,
    )


def explore_factory(
    train_state: TrainStatePolicyQValueTemperature, algo_params: SACParams
) -> Callable:
    policy_apply = train_state.policy_state.apply_fn

    @jax.jit
    def fn(
        policy_state: TrainState,
        key: jax.Array,
        observations: jax.Array,
    ):
        actions, log_probs = policy_apply(
            {"params": policy_state.params}, observations
        ).sample_and_log_prob(seed=key)
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions, log_probs

    return fn


def process_experience_factory(
    train_state: TrainStatePolicyQValueTemperature, algo_params: SACParams
) -> Callable:
    policy_apply = train_state.policy_state.apply_fn
    qvalue_apply = train_state.qvalue_state.apply_fn

    @jax.jit
    def fn(
        sac_state: TrainStatePolicyQValueTemperature,
        key: jax.Array,
        experience: Experience,
    ):
        next_actions, next_log_probs = policy_apply(
            {"params": sac_state.policy_state.target_params},
            experience.next_observation,
        ).sample_and_log_prob(seed=0)
        next_log_probs = jnp.sum(next_log_probs, axis=-1, keepdims=True)

        next_q1, next_q2 = qvalue_apply(
            {"params": sac_state.qvalue_state.target_params},
            experience.next_observation,
            next_actions,
        )
        next_q_min = jnp.minimum(next_q1, next_q2)

        discounts = algo_params.gamma * (1.0 - experience.done[..., None])
        targets = compute_td_targets(
            experience.reward[..., None], discounts, next_q_min
        )

        temps = sac_state.temperature_state.apply_fn(
            {"params": sac_state.temperature_state.params}
        )
        targets -= discounts * temps * next_log_probs

        return (experience.observation, experience.action, targets)

    return fn


def update_step_factory(
    train_state: TrainStatePolicyQValueTemperature, config: AlgoConfig
) -> Callable:
    qvalue_apply = train_state.qvalue_state.apply_fn
    policy_apply = train_state.policy_state.apply_fn
    temperature_apply = train_state.temperature_state.apply_fn

    @jax.jit
    def update_qvalue_fn(qvalue_state: TrainState, batch: tuple[jax.Array]):
        observations, actions, targets = batch

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
        policy_state: TrainState,
        qvalue_state: TrainState,
        temperature_state: TrainState,
        batch: tuple[jax.Array],
    ):
        def loss_fn(params: Params, observations):
            actions, log_probs = policy_apply(
                {"params": params}, observations
            ).sample_and_log_prob(seed=0)
            log_probs = jnp.sum(log_probs, axis=-1)
            qvalues, _ = qvalue_apply(
                {"params": qvalue_state.params}, observations, actions
            )
            temp = temperature_apply({"params": temperature_state.params})
            loss = jnp.mean(temp * log_probs - qvalues)
            return loss, {"loss_policy": loss, "entropy": -jnp.mean(log_probs)}

        observations, _, _ = batch
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

    @jax.jit
    def update_temperature_fn(temperature_state: TrainState, entropy: float):
        def loss_fn(params: Params):
            temperature = temperature_apply({"params": params})
            loss = temperature * jnp.mean(entropy - config.algo_params.target_entropy)
            return loss, {"temperature_loss": loss, "temperature": temperature}

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            temperature_state.params
        )
        temperature_state = temperature_state.apply_gradients(grads=grads)
        return temperature_state, loss, info

    def update_step_fn(
        policy_state: TrainState,
        qvalue_state: TrainState,
        temperature_state: TrainState,
        batch: tuple[jax.Array],
    ):
        qvalue_state, loss_qvalue, info_qvalue = update_qvalue_fn(qvalue_state, batch)

        (policy_state, qvalue_state), loss_policy, info_policy = update_policy_fn(
            policy_state, qvalue_state, temperature_state, batch
        )

        temperature_state, loss_temperature, info_temperature = update_temperature_fn(
            temperature_state, info_policy["entropy"]
        )

        info = info_qvalue | info_policy | info_temperature
        info["total_loss"] = loss_qvalue + loss_policy + loss_temperature

        return qvalue_state, policy_state, temperature_state, info

    return update_step_fn


class SAC(Base):
    """Sof Actor Crtic (SAC)"""

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
            train_state_sac_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            run_name=run_name,
            tabulate=tabulate,
        )
        self.step = 0
        self.config.algo_params.target_entropy = (
            -self.config.env_cfg.action_space.shape[-1] / 2
        )

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.explore(observation)

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )
        action, log_prob = self.explore_fn(self.state.policy_state, keys, observation)

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
        def fn(state: TrainStatePolicyQValueTemperature, key: jax.Array, sample: tuple):
            experiences = self.process_experience_fn(state, key, sample)
            (
                state.qvalue_state,
                state.policy_state,
                state.temperature_state,
                info,
            ) = self.update_step_fn(
                state.policy_state,
                state.qvalue_state,
                state.temperature_state,
                experiences,
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