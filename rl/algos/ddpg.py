"""Deep Deterministic Policy Gradient (DDPG)"""

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


@dataclass
class DDPGParams(AlgoParams):
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
class ParamsDDPG:
    params_policy: Params
    params_qvalue: Params
    params_target_qvalue: Params


class DDPGTrainState(TrainState):
    policy_fn: Callable = struct.field(pytree_node=False)
    qvalue_fn: Callable = struct.field(pytree_node=False)


def train_state_ddpg_factory(
    key: jax.Array,
    config: AlgoConfig,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> DDPGTrainState:
    import flax.linen as nn
    from optax import adam
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
            locs = nn.tanh(nn.Dense(self.num_outputs)(x))
            return dx.Normal(locs, jnp.ones_like(locs))

    module_policy = Policy(action_shape[-1])
    params_policy = init_params(key1, module_policy, observation_shape, tabulate)

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
    params_qvalue = init_params(
        key2, module_qvalue, [observation_shape, action_shape], tabulate
    )

    return DDPGTrainState.create(
        apply_fn=None,
        params=ParamsDDPG(
            params_policy=params_policy,
            params_qvalue=params_qvalue,
            params_target_qvalue=params_qvalue,
        ),
        tx=adam(config.update_cfg.learning_rate),
        policy_fn=module_policy.apply,
        qvalue_fn=module_qvalue.apply,
    )


def explore_factory(train_state: DDPGTrainState, algo_params: DDPGParams) -> Callable:
    @jax.jit
    def fn(
        params: Params,
        key: jax.Array,
        observations: jax.Array,
        action_noise: float,
    ):
        locs = train_state.policy_fn({"params": params}, observations).loc
        dists = dx.Normal(locs, action_noise * jnp.ones_like(locs))
        outputs = dists.sample_and_log_prob(seed=key)
        return outputs

    return fn


def process_experience_factory(
    train_state: DDPGTrainState,
    algo_params: DDPGParams,
    vectorized: bool,
    parallel: bool,
) -> Callable:
    def compute_returns(
        params: ParamsDDPG,
        key: jax.Array,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
    ) -> jax.Array:
        next_actions = train_state.policy_fn(
            {"params": params.params_policy}, next_observations
        ).loc
        noise = jnp.clip(
            dx.Normal(0, algo_params.target_noise_std).sample(
                seed=key, sample_shape=next_actions.shape
            ),
            -algo_params.target_noise_clip,
            algo_params.target_noise_clip,
        )
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        next_q1, next_q2 = train_state.qvalue_fn(
            {"params": params.params_target_qvalue}, next_observations, next_actions
        )
        next_q_min = jnp.minimum(next_q1, next_q2)

        discounts = algo_params.gamma * (1.0 - dones[..., None])
        return (rewards[..., None] + discounts * next_q_min,)

    # TODO make this part decorator ?
    returns_fn = compute_returns
    if vectorized:
        returns_fn = jax.vmap(returns_fn, in_axes=(None, 1, 1, 1), out_axes=1)
    if parallel:
        returns_fn = fn_parallel(returns_fn)

    @jax.jit
    def fn(
        params: ParamsDDPG,
        key: jax.Array,
        sample: list[OffPolicyExp],
    ):
        stacked = stack_experiences(sample)

        observations = stacked.observation
        actions = stacked.action
        (returns,) = returns_fn(
            params,
            key,
            stacked.next_observation,
            stacked.reward,
            stacked.done,
        )

        return observations, actions, returns

    return fn


def update_step_factory(
    train_state: TrainStatePolicyQvalue, config: AlgoConfig
) -> Callable:
    @jax.jit
    def update_qvalue_fn(state: DDPGTrainState, batch: tuple[jax.Array]):
        def loss_fn(params: ParamsDDPG, observations, actions, targets):
            q1, q2 = train_state.qvalue_fn(
                {"params": params.params_qvalue}, observations, actions
            )
            loss_q1 = loss_mean_squared_error(q1, targets)
            loss_q2 = loss_mean_squared_error(q2, targets)

            return loss_q1 + loss_q2, {"loss_qvalue": loss_q1 + loss_q2}

        observations, actions, targets = batch
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, observations, actions, targets
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, info

    @jax.jit
    def update_policy_fn(state: DDPGTrainState, batch: tuple[jax.Array]):
        def loss_fn(params: ParamsDDPG, observations, actions):
            actions = train_state.policy_fn(
                {"params": params.params_policy}, observations
            ).loc
            qvalues, _ = train_state.qvalue_fn(
                {"params": state.params.params_qvalue}, observations, actions
            )
            loss = -jnp.mean(qvalues)
            return loss, {"loss_policy": loss}

        observations, actions, _ = batch
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, observations, actions
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, info

    def update_step_fn(
        state: DDPGTrainState,
        key: jax.Array,
        batch: tuple[jax.Array],
        step: int,
    ):
        state, loss_qvalue, info_qvalue = update_qvalue_fn(state, batch)

        if step % config.algo_params.policy_update_frequency == 0:
            state, loss_policy, info_policy = update_policy_fn(state, batch)
        else:
            loss_policy = 0.0
            info_policy = {}

        info = info_qvalue | info_policy
        info["total_loss"] = loss_qvalue + loss_policy

        state.params.params_target_qvalue = jax.tree_map(
            lambda p, t: (1 - config.algo_params.tau) * t + config.algo_params.tau * p,
            state.params.params_qvalue,
            state.params.params_target_qvalue,
        )

        return state, info

    return update_step_fn


class DDPG(Base):
    """
    Deep Deterministic Policy Gradient (DDPG)
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

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

        action, zeros = self.explore_fn(
            self.state.params.params_policy, keys, observation, 0.0
        )
        return action, zeros

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

        action, log_prob = self.explore_fn(
            self.state.params.params_policy,
            keys,
            observation,
            self.algo_params.action_noise,
        )
        return action, log_prob

    def should_update(self, step: int, buffer: Buffer) -> bool:
        self.step = step
        return (
            len(buffer) >= self.config.update_cfg.batch_size
            and step % self.algo_params.skip_steps == 0
            and step >= self.algo_params.start_step
        )

    def update(self, buffer: OffPolicyBuffer) -> dict:
        def fn(state: TrainStatePolicyQvalue, key: jax.Array, sample: tuple):
            key1, key2 = jax.random.split(key)
            experiences = self.process_experience_fn(state.params, key1, sample)
            state, info = self.update_step_fn(state, key2, experiences, self.step)
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
