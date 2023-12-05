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


def explore_factory(train_state: TrainState, batched: bool) -> Callable:
    @jax.jit
    def fn(
        params: Params, key: jax.Array, observations: jax.Array, exploration: float
    ) -> jax.Array:
        if not batched:
            observations = jnp.expand_dims(observations, axis=0)

        all_qvalues = train_state.apply_fn({"params": params}, observations)
        greedy_action = jnp.argmax(all_qvalues, axis=-1)
        key1, key2 = jax.random.split(key)
        eps = jax.random.uniform(key1, greedy_action.shape)
        random_action = jax.random.randint(
            key2, greedy_action.shape, 0, all_qvalues.shape[-1]
        )

        outputs = jnp.where(eps <= exploration, random_action, greedy_action)
        if not batched:
            return jax.tree_map(lambda x: jnp.squeeze(x, axis=0), outputs)
        return outputs

    return fn


def process_experience_factory(
    train_state: TrainState, gamma: float, vectorized: bool
) -> Callable:
    from rl.buffer import stack_experiences

    def compute_returns(
        params: Params,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
    ) -> jax.Array:
        all_next_qvalues = train_state.apply_fn({"params": params}, next_observations)
        next_qvalues = jnp.max(all_next_qvalues, axis=-1, keepdims=True)

        discounts = gamma * (1.0 - dones[..., None])
        return rewards[..., None] + discounts * next_qvalues

    returns_fn = compute_returns
    if vectorized:
        returns_fn = jax.vmap(returns_fn, in_axes=(None, 1, 1, 1), out_axes=1)

    @jax.jit
    def fn(params: Params, sample: list[OffPolicyExp]):
        stacked = stack_experiences(sample)

        observations = stacked.observation
        actions = stacked.action[..., None]
        returns = compute_returns(
            params, stacked.next_observation, stacked.reward, stacked.done
        )

        if vectorized:
            observations = jnp.reshape(observations, (-1, *observations.shape[2:]))
            actions = jnp.reshape(actions, (-1, *actions.shape[2:]))
            returns = jnp.reshape(returns, (-1, *returns.shape[2:]))

        return observations, actions, returns

    return fn


def update_step_factory(train_state: TrainState) -> Callable:
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
        n_envs: int = 1,
        run_name: str = None,
        tabulate: bool = False,
    ):
        Base.__init__(self, seed, run_name=run_name)
        self.config = config

        self.state = train_state_qvalue_factory(
            self.nextkey(),
            self.config,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            n_envs=n_envs,
            tabulate=tabulate,
        )
        self.explore_fn = explore_factory(self.state, n_envs > 1)
        self.process_experience_fn = process_experience_factory(
            self.state, self.config.gamma, n_envs > 1
        )
        self.update_step_fn = update_step_factory(self.state)

        self.n_envs = n_envs

    def select_action(self, observation: jax.Array) -> jax.Array:
        action = self.explore_fn(
            self.state.params, self.nextkey(), observation, NO_EXPLORATION
        )
        return action, jnp.zeros_like(action)

    def explore(self, observation: jax.Array):
        action = self.explore_fn(
            self.state.params, self.nextkey(), observation, self.config.exploration
        )
        return action, jnp.zeros_like(action)

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
            EnvType.SINGLE,
            EnvProcs.ONE if self.n_envs == 1 else EnvProcs.MANY,
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
            EnvType.SINGLE,
            EnvProcs.ONE if self.n_envs == 1 else EnvProcs.MANY,
            AlgoType.OFF_POLICY,
            start_step=step,
            saver=self.saver,
            callbacks=callbacks,
        )
