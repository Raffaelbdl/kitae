"""Deep Q-Network (DQN)"""

from dataclasses import dataclass
from typing import Callable

import distrax as dx
import jax
import jax.numpy as jnp
import numpy as np
import optax


from rl.base import OffPolicyAgent, EnvType, EnvProcs, AlgoType
from rl.callbacks.callback import Callback
from rl.config import AlgoConfig, AlgoParams
from rl.types import Params, GymEnv, EnvPoolEnv

from rl.buffer import Experience
from rl.loss import loss_mean_squared_error
from rl.train import train
from rl.timesteps import compute_td_targets

from rl.algos.factory import AlgoFactory
from rl.modules.modules import init_params
from rl.modules.train_state import TrainState
from rl.modules.qvalue import qvalue_factory

NO_EXPLORATION = 0.0


@dataclass
class DQNParams(AlgoParams):
    """
    Deep Q-Network parameters

    Parameters:
        exploration: The exploration coefficient of the epsilon-greedy policy.
        gamma: The discount factor.
        skip_step: The numbers of steps skipped when training.
    """

    exploration: float
    gamma: float
    skip_steps: int
    start_step: int = -1


def train_state_dqn_factory(
    key: jax.Array,
    config: AlgoConfig,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> TrainState:
    observation_shape = config.env_cfg.observation_space.shape

    qvalue = qvalue_factory(
        config.env_cfg.observation_space, config.env_cfg.action_space
    )()
    return TrainState.create(
        apply_fn=qvalue.apply,
        params=init_params(key, qvalue, [observation_shape], tabulate),
        target_params=init_params(key, qvalue, [observation_shape], False),
        tx=optax.adam(config.update_cfg.learning_rate),
    )


def explore_factory(config: AlgoConfig) -> Callable:
    @jax.jit
    def fn(
        qvalue_state: TrainState,
        key: jax.Array,
        observations: jax.Array,
        exploration: float,
    ) -> jax.Array:
        all_qvalues = qvalue_state.apply_fn(
            {"params": qvalue_state.params}, observations
        )
        actions, log_probs = dx.EpsilonGreedy(
            all_qvalues, exploration
        ).sample_and_log_prob(seed=key)

        return actions, log_probs

    return fn


def process_experience_factory(config: AlgoConfig) -> Callable:
    algo_params = config.algo_params

    @jax.jit
    def fn(dqn_state: TrainState, key: jax.Array, experience: Experience):
        all_next_qvalues = dqn_state.apply_fn(
            {"params": dqn_state.params}, experience.next_observation
        )
        next_qvalues = jnp.max(all_next_qvalues, axis=-1, keepdims=True)

        discounts = algo_params.gamma * (1.0 - experience.done[..., None])
        returns = compute_td_targets(
            experience.reward[..., None], discounts, next_qvalues
        )
        actions = experience.action[..., None]

        return experience.observation, actions, returns

    return fn


def update_step_factory(config: AlgoConfig) -> Callable:

    @jax.jit
    def update_qvalue_fn(qvalue_state: TrainState, batch: tuple[jax.Array]):

        def loss_fn(params: Params, observations, actions, returns):
            all_qvalues = qvalue_state.apply_fn({"params": params}, observations)
            qvalues = jnp.take_along_axis(all_qvalues, actions, axis=-1)
            loss = loss_mean_squared_error(qvalues, returns)
            return loss, {"loss_qvalue": loss}

        observations, actions, returns = batch

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            qvalue_state.params,
            observations=observations,
            actions=actions,
            returns=returns,
        )
        qvalue_state = qvalue_state.apply_gradients(grads=grads)

        return qvalue_state, loss, info

    def update_step_fn(state: TrainState, batch: tuple[jax.Array]):
        state, loss_qvalue, info_qvalue = update_qvalue_fn(state, batch)
        info = info_qvalue
        info["total_loss"] = loss_qvalue

        return state, info

    return update_step_fn


class DQN(OffPolicyAgent):
    """
    Deep Q-Network (DQN)
    Paper : https://arxiv.org/abs/1312.5602
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
            train_state_dqn_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            run_name=run_name,
            tabulate=tabulate,
            experience_type=Experience,
        )
        self.algo_params = self.config.algo_params

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

        action, zeros = self.explore_fn(
            self.state.params, keys, observation, exploration=NO_EXPLORATION
        )
        return action, zeros

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

        action, zeros = self.explore_fn(
            self.state.params,
            keys,
            observation,
            exploration=self.algo_params.exploration,
        )
        return jax.device_get(action), zeros

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
