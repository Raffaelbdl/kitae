import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import pettingzoo
import vec_parallel_env

from rl.base import Base, EnvType, EnvProcs
from rl.buffer import OnPolicyBuffer, OnPolicyExp
from rl.timesteps import calculate_gaes_targets
from rl.modules.policy_value import TrainStatePolicyValue, ParamsPolicyValue
from rl.train import train

ParallelEnv = pettingzoo.ParallelEnv
SubProcVecParallelEnv = vec_parallel_env.SubProcVecParallelEnv

DictArray = dict[str, jax.Array]

from rl.algos.ppo import (
    train_state_policy_value_factory,
    explore_factory,
    update_step_factory,
)


def process_experience_factory(
    train_state: TrainStatePolicyValue,
    gamma: float,
    _lambda: float,
    normalize: float,
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
        gaes, targets, values = {}, {}, {}
        for agent in observations.keys():
            g, t, v = gaes_fn(
                params,
                observations[agent],
                stacked.next_observation[agent],
                stacked.done[agent],
                stacked.reward[agent],
            )
            gaes[agent] = g
            targets[agent] = t
            values[agent] = v

        actions = jax.tree_map(lambda x: x[..., None], stacked.action)
        log_probs = jax.tree_map(lambda x: x[..., None], stacked.log_prob)

        observations = jnp.concatenate(list(observations.values()), axis=0)
        actions = jnp.concatenate(list(actions.values()), axis=0)
        log_probs = jnp.concatenate(list(log_probs.values()), axis=0)
        gaes = jnp.concatenate(list(gaes.values()), axis=0)
        targets = jnp.concatenate(list(targets.values()), axis=0)
        values = jnp.concatenate(list(values.values()), axis=0)

        if vectorized:
            observations = jnp.reshape(observations, (-1, *observations.shape[2:]))
            actions = jnp.reshape(actions, (-1, *actions.shape[2:]))
            log_probs = jnp.reshape(log_probs, (-1, *log_probs.shape[2:]))
            gaes = jnp.reshape(gaes, (-1, *gaes.shape[2:]))
            targets = jnp.reshape(targets, (-1, *targets.shape[2:]))
            values = jnp.reshape(values, (-1, *values.shape[2:]))

        return observations, actions, log_probs, gaes, targets, values

    return fn


class PPO(Base):
    def __init__(
        self,
        seed: int,
        config: ml_collections.ConfigDict,
        *,
        rearrange_pattern: str = "b h w c -> b h w c",
        n_envs: int = 1,
        n_agents: int = 2,
        run_name: str = None,
        tabulate: bool = False,
    ):
        Base.__init__(self, seed, run_name=run_name)
        self.config = config

        self.state = train_state_policy_value_factory(
            self.nextkey(),
            self.config,
            rearrange_pattern=rearrange_pattern,
            n_envs=n_envs * n_agents,
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

    def select_action(self, observation: DictArray) -> tuple[DictArray, DictArray]:
        return self.explore(observation)

    def explore(self, observation: DictArray) -> tuple[DictArray, DictArray]:
        def fn(
            params: ParamsPolicyValue,
            key: jax.random.PRNGKeyArray,
            observation: DictArray,
        ):
            action, log_prob = {}, {}
            for agent, obs in observation.items():
                key, _k = jax.random.split(key)
                a, lp = self.explore_fn(params, _k, obs)
                action[agent] = a
                log_prob[agent] = lp
            return action, log_prob

        return fn(self.state.params, self.nextkey(), observation)

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

    def train(
        self,
        env: ParallelEnv | SubProcVecParallelEnv,
        n_env_steps: int,
        callbacks: list,
    ):
        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.PARALLEL,
            EnvProcs.ONE if self.n_envs == 1 else EnvProcs.MANY,
            saver=self.saver,
            callbacks=callbacks,
        )

    def resume(self, env: ParallelEnv | SubProcVecParallelEnv, n_env_steps: int):
        step, self.state = self.saver.restore_latest_step(self.state)

        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.PARALLEL,
            EnvProcs.ONE if self.n_envs == 1 else EnvProcs.MANY,
            start_step=step,
            saver=self.saver,
        )
