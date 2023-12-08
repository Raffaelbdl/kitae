from typing import Callable

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from rl.base import Base, EnvType, EnvProcs, AlgoType
from rl.buffer import OffPolicyBuffer, OffPolicyExp
from rl.loss import loss_mean_squared_error
from rl.train import train

from rl.types import Params, GymEnv, EnvPoolEnv

from rl.modules.qvalue import train_state_qvalue_factory
from rl.algos.general_fns import fn_parallel
from rl.buffer import stack_experiences


NO_EXPLORATION = 0.0


def loss_factory(train_state: TrainState) -> Callable:
    @jax.jit
    def fn(params: Params, batch: tuple[jax.Array]):
        observations, actions, returns = batch
        all_qvalues = train_state.apply_fn({"params": params}, observations)
        qvalues = jnp.take_along_axis(all_qvalues, actions, axis=-1)

        loss = loss_mean_squared_error(qvalues, returns)
        return loss, {"loss_qvalue": loss}

    return fn


def explore_factory(
    train_state: TrainState,
    config: ml_collections.ConfigDict,
) -> Callable:
    @jax.jit
    def fn(
        params: Params, key: jax.Array, observations: jax.Array, exploration: float
    ) -> jax.Array:
        all_qvalues = train_state.apply_fn({"params": params}, observations)
        greedy_action = jnp.argmax(all_qvalues, axis=-1)
        key1, key2 = jax.random.split(key)
        eps = jax.random.uniform(key1, greedy_action.shape)
        random_action = jax.random.randint(
            key2, greedy_action.shape, 0, all_qvalues.shape[-1]
        )

        actions = jnp.where(eps <= exploration, random_action, greedy_action)

        return actions, jnp.zeros_like(actions)

    return fn


def process_experience_factory(
    train_state: TrainState,
    config: ml_collections.ConfigDict,
    vectorized: bool,
    parallel: bool,
) -> Callable:
    def compute_returns(
        params: Params,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
    ) -> jax.Array:
        all_next_qvalues = train_state.apply_fn({"params": params}, next_observations)
        next_qvalues = jnp.max(all_next_qvalues, axis=-1, keepdims=True)

        discounts = config.gamma * (1.0 - dones[..., None])
        return (rewards[..., None] + discounts * next_qvalues,)

    returns_fn = compute_returns
    if vectorized:
        returns_fn = jax.vmap(returns_fn, in_axes=(None, 1, 1, 1), out_axes=1)
    if parallel:
        returns_fn = fn_parallel(returns_fn)

    @jax.jit
    def fn(params: Params, sample: list[OffPolicyExp]):
        stacked = stack_experiences(sample)

        observations = stacked.observation
        actions = jax.tree_map(lambda x: x[..., None], stacked.action)
        (returns,) = returns_fn(
            params, stacked.next_observation, stacked.reward, stacked.done
        )

        return observations, actions, returns

    return fn


def update_step_factory(
    train_state: TrainState, config: ml_collections.ConfigDict
) -> Callable:
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
    def __init__(
        self,
        seed: int,
        config: ml_collections.ConfigDict,
        *,
        rearrange_pattern: str = "b h w c -> b h w c",
        preprocess_fn: Callable = None,
        run_name: str = None,
        tabulate: bool = False,
    ):
        Base.__init__(
            self,
            seed,
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

    def select_action(self, observation: jax.Array) -> jax.Array:
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

        action, zeros = self.explore_fn(
            self.state.params, keys, observation, exploration=NO_EXPLORATION
        )
        return action, zeros

    def explore(self, observation: jax.Array):
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

        action, zeros = self.explore_fn(
            self.state.params, keys, observation, exploration=self.config.exploration
        )
        return action, zeros

    def should_update(self, step: int, buffer: OffPolicyBuffer) -> None:
        return (
            len(buffer) >= self.config.batch_size and step % self.config.skip_steps == 0
        )

    def update(self, buffer: OffPolicyBuffer):
        def fn(state: TrainState, key: jax.Array, sample: tuple):
            experiences = self.process_experience_fn(state.params, sample)
            state, loss, info = self.update_step_fn(state, key, experiences)
            return state, info

        sample = buffer.sample(self.config.batch_size)
        self.state, info = fn(self.state, self.nextkey(), sample)
        return info

    def train(self, env: GymEnv | EnvPoolEnv, n_env_steps: int, callbacks: list):
        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE
            if self.config.env_config.n_agents == 1
            else EnvType.PARALLEL,
            EnvProcs.ONE if self.config.env_config.n_envs == 1 else EnvProcs.MANY,
            AlgoType.OFF_POLICY,
            saver=self.saver,
            callbacks=callbacks,
        )

    def resume(self, env: GymEnv | EnvPoolEnv, n_env_steps: int, callbacks: list):
        step, self.state = self.saver.restore_latest_step(self.state)

        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE
            if self.config.env_config.n_agents == 1
            else EnvType.PARALLEL,
            EnvProcs.ONE if self.config.env_config.n_envs == 1 else EnvProcs.MANY,
            AlgoType.OFF_POLICY,
            start_step=step,
            saver=self.saver,
            callbacks=callbacks,
        )
