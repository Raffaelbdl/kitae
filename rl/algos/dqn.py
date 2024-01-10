"""Deep Q-Network (DQN)"""

from dataclasses import dataclass
import functools
from typing import Callable

import chex
import distrax as dx
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np

from rl.algos.general_fns import fn_parallel

from rl.base import Base, EnvType, EnvProcs, AlgoType
from rl.callbacks.callback import Callback
from rl.config import AlgoConfig, AlgoParams
from rl.types import Params, GymEnv, EnvPoolEnv

from rl.buffer import OffPolicyBuffer, Experience, stack_experiences
from rl.loss import loss_mean_squared_error
from rl.modules.qvalue import train_state_qvalue_factory
from rl.train import train
from rl.timesteps import compute_td_targets


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


def loss_factory(train_state: TrainState) -> Callable:
    @jax.jit
    def fn(params: Params, batch: tuple[jax.Array]) -> tuple[float, dict]:
        observations, actions, returns = batch
        all_qvalues = train_state.apply_fn({"params": params}, observations)
        qvalues = jnp.take_along_axis(all_qvalues, actions, axis=-1)

        loss = loss_mean_squared_error(qvalues, returns)
        return loss, {"loss_qvalue": loss}

    return fn


def explore_factory(train_state: TrainState, algo_params: DQNParams) -> Callable:
    @jax.jit
    def fn(
        params: Params, key: jax.Array, observations: jax.Array, exploration: float
    ) -> jax.Array:
        all_qvalues = train_state.apply_fn({"params": params}, observations)
        actions, log_probs = dx.EpsilonGreedy(
            all_qvalues, exploration
        ).sample_and_log_prob(seed=key)

        return actions, log_probs

    return fn


def process_experience_factory(
    train_state: TrainState, algo_params: DQNParams
) -> Callable:
    qvalue_apply = train_state.apply_fn

    @jax.jit
    def fn(dqn_state: TrainState, key: jax.Array, experience: Experience):
        all_next_qvalues = qvalue_apply(
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


def update_step_factory(train_state: TrainState, config: AlgoConfig) -> Callable:
    loss_fn = loss_factory(train_state)

    @jax.jit
    def fn(state: TrainState, key: jax.Array, batch: tuple[jax.Array]):
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch=batch
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, info

    return fn


class DQN(Base):
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
        Base.__init__(
            self,
            config=config,
            train_state_factory=train_state_qvalue_factory,
            explore_factory=explore_factory,
            process_experience_factory=process_experience_factory,
            update_step_factory=update_step_factory,
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

    def should_update(self, step: int, buffer: OffPolicyBuffer) -> bool:
        return (
            len(buffer) >= self.config.update_cfg.batch_size
            and step % self.algo_params.skip_steps == 0
        )

    def update(self, buffer: OffPolicyBuffer) -> dict:
        def fn(state: TrainState, key: jax.Array, sample: tuple):
            key1, key2 = jax.random.split(key, 2)
            experiences = self.process_experience_fn(state, key1, sample)
            state, loss, info = self.update_step_fn(state, key2, experiences)
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
