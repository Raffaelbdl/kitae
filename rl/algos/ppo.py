from typing import Callable

import distrax as dx
from envpool.python.api import EnvPool
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from rl.base import Base, EnvType, EnvProcs
from rl.buffer import OnPolicyBuffer, OnPolicyExp
from rl.loss import loss_policy_ppo_discrete, loss_value_clip
from rl.timesteps import calculate_gaes_targets
from rl.train import train
from rl.modules.policy_value import (
    train_state_policy_value_factory,
    TrainStatePolicyValue,
    ParamsPolicyValue,
)

GymEnv = gym.Env
EnvPoolEnv = EnvPool


def loss_factory(
    train_state: TrainStatePolicyValue,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
) -> Callable:
    @jax.jit
    def fn(params: ParamsPolicyValue, batch: tuple[jax.Array]):
        observations, actions, log_probs_old, gaes, targets, values_old = batch
        hiddens = train_state.encoder_fn(
            {"params": params.params_encoder}, observations
        )

        logits = train_state.policy_fn({"params": params.params_policy}, hiddens)
        all_log_probs = nn.log_softmax(logits)
        log_probs = jnp.take_along_axis(all_log_probs, actions, axis=-1)

        loss_policy, info_policy = loss_policy_ppo_discrete(
            logits, log_probs, log_probs_old, gaes, clip_eps, entropy_coef
        )

        values = train_state.value_fn({"params": params.params_value}, hiddens)
        loss_value, info_value = loss_value_clip(values, targets, values_old, clip_eps)

        loss = loss_policy + value_coef * loss_value
        info = info_policy | info_value
        return loss, info

    return fn


def explore_factory(
    train_state: TrainStatePolicyValue,
    batched: bool,
) -> Callable:
    @jax.jit
    def fn(
        params: ParamsPolicyValue, key: jax.Array, observations: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if not batched:
            observations = jnp.expand_dims(observations, axis=0)

        hiddens = train_state.encoder_fn(
            {"params": params.params_encoder}, observations
        )
        logits = train_state.policy_fn({"params": params.params_policy}, hiddens)
        outputs = dx.Categorical(logits=logits).sample_and_log_prob(seed=key)

        if not batched:
            return jax.tree_map(lambda x: jnp.squeeze(x, axis=0), outputs)
        return outputs

    return fn


def process_experience_factory(
    train_state: TrainStatePolicyValue,
    gamma: float,
    _lambda: float,
    normalize: bool,
    vectorized: bool,
):
    from rl.buffer import stack_experiences

    def compute_values_gaes(
        params: ParamsPolicyValue,
        observations: jax.Array,
        next_observations: jax.Array,
        dones: jax.Array,
        rewards: jax.Array,
    ):
        all_obs = jnp.concatenate([observations, next_observations[-1:]], axis=0)
        all_hiddens = train_state.encoder_fn({"params": params.params_encoder}, all_obs)
        all_values = train_state.value_fn({"params": params.params_value}, all_hiddens)

        values = all_values[:-1]
        next_values = all_values[1:]

        not_dones = 1.0 - dones[..., None]
        discounts = gamma * not_dones

        rewards = rewards[..., None]
        gaes, targets = calculate_gaes_targets(
            values, next_values, discounts, rewards, _lambda, normalize
        )

        return gaes, targets, values

    gaes_fn = compute_values_gaes
    if vectorized:
        gaes_fn = jax.vmap(gaes_fn, in_axes=(None, 1, 1, 1, 1), out_axes=1)

    @jax.jit
    def fn(params: ParamsPolicyValue, sample: list[OnPolicyExp]):
        stacked = stack_experiences(sample)

        observations = stacked.observation

        gaes, targets, values = gaes_fn(
            params,
            observations,
            stacked.next_observation,
            stacked.done,
            stacked.reward,
        )

        actions = stacked.action[..., None]
        log_probs = stacked.log_prob[..., None]

        if vectorized:
            observations = jnp.reshape(observations, (-1, *observations.shape[2:]))
            actions = jnp.reshape(actions, (-1, *actions.shape[2:]))
            log_probs = jnp.reshape(log_probs, (-1, *log_probs.shape[2:]))
            gaes = jnp.reshape(gaes, (-1, *gaes.shape[2:]))
            targets = jnp.reshape(targets, (-1, *targets.shape[2:]))
            values = jnp.reshape(values, (-1, *values.shape[2:]))

        return observations, actions, log_probs, gaes, targets, values

    return fn


def update_step_factory(
    train_state: TrainStatePolicyValue,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    batch_size: int,
):
    loss_fn = loss_factory(train_state, clip_eps, entropy_coef, value_coef)

    @jax.jit
    def fn(state: TrainStatePolicyValue, key: jax.Array, experiences: tuple[jax.Array]):
        num_elems = experiences[0].shape[0]
        iterations = num_elems // batch_size
        inds = jax.random.permutation(key, num_elems)[: iterations * batch_size]

        experiences = jax.tree_util.tree_map(
            lambda x: x[inds].reshape((iterations, batch_size) + x.shape[1:]),
            experiences,
        )

        loss = 0.0
        for batch in zip(*experiences):
            (l, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, batch=batch
            )
            loss += l

            state = state.apply_gradients(grads=grads)
        return state, loss, info

    return fn


class PPO(Base):
    def __init__(
        self,
        seed: int,
        config: ml_collections.ConfigDict,
        *,
        rearrange_pattern: str = "b h w c -> b h w c",
        n_envs: int = 1,
        run_name: str = None,
        tabulate: bool = False,
    ):
        Base.__init__(self, seed, run_name=run_name)
        self.config = config

        self.state = train_state_policy_value_factory(
            self.nextkey(),
            self.config,
            rearrange_pattern=rearrange_pattern,
            n_envs=n_envs,
            tabulate=tabulate,
        )
        self.explore_fn = explore_factory(self.state, n_envs > 1)
        self.process_experience_fn = process_experience_factory(
            self.state,
            self.config.gamma,
            self.config._lambda,
            self.config.normalize,
            n_envs > 1,
        )
        self.update_step_fn = update_step_factory(
            self.state,
            self.config.clip_eps,
            self.config.entropy_coef,
            self.config.value_coef,
            self.config.batch_size,
        )

        self.n_envs = n_envs

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.explore(observation)

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        action, log_prob = self.explore_fn(
            self.state.params, self.nextkey(), observation
        )
        return action, log_prob

    def update(self, buffer: OnPolicyBuffer):
        def fn(
            state: TrainStatePolicyValue, key: jax.random.PRNGKeyArray, sample: tuple
        ):
            experiences = self.process_experience_fn(state.params, sample)

            loss = 0.0
            for epoch in range(self.config.num_epochs):
                key, _k = jax.random.split(key)
                state, l, info = self.update_step_fn(state, _k, experiences)
                loss += l

            loss /= self.config.num_epochs
            return state, info

        sample = buffer.sample()
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
            saver=self.saver,
            callbacks=callbacks,
        )

    def resume(self, env: GymEnv | EnvPoolEnv, n_env_steps: int):
        step, self.state = self.saver.restore_latest_step(self.state)

        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE,
            EnvProcs.ONE if self.n_envs == 1 else EnvProcs.MANY,
            start_step=step,
            saver=self.saver,
        )
