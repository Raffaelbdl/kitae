from typing import Callable

import jax
import ml_collections
import numpy as np

from rl.base import Base, EnvType, EnvProcs
from rl.buffer import OnPolicyBuffer
from rl.train import train_population

from rl.types import ParallelEnv, SubProcVecParallelEnv, DictArray

from rl.algos.ppo import explore_factory
from rl.algos.ippo import process_experience_factory
from rl.modules.policy_value import ParamsPolicyValue, TrainStatePolicyValue
from rl.modules.policy_value import train_state_policy_value_population_factory
from rl.populations.pop_ppo import update_step_factory


class PopulationPPO(Base):
    def __init__(
        self,
        seed: int,
        population_size: int,
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

        self.state = train_state_policy_value_population_factory(
            self.nextkey(),
            self.config,
            population_size,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
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
            self.config.jsd_coef,
            self.config.batch_size,
        )

        self.population_size = population_size
        self.n_envs = n_envs

    def select_action(
        self, observations: list[DictArray]
    ) -> tuple[list[DictArray], list[DictArray]]:
        return self.explore(observations)

    def explore(
        self, observations: list[DictArray]
    ) -> tuple[list[DictArray], list[DictArray]]:
        def fn(
            params: list[ParamsPolicyValue],
            key: jax.Array,
            observations: list[DictArray],
        ):
            actions, log_probs = [], []
            for i, obs in enumerate(observations):
                _actions, _log_probs = {}, {}
                for agent, o in obs.items():
                    key, _k = jax.random.split(key)
                    a, lp = self.explore_fn(params[i], _k, o)
                    _actions[agent] = a
                    _log_probs[agent] = lp
                actions.append(_actions)
                log_probs.append(_log_probs)
            return actions, log_probs

        return fn(self.state.params, self.nextkey(), observations)

    def update(self, buffers: list[OnPolicyBuffer]):
        def fn(state: TrainStatePolicyValue, key: jax.Array, samples: list[tuple]):
            experiences = [
                self.process_experience_fn(state.params[i], samples[i])
                for i in range(len(buffers))
            ]

            loss = 0.0
            for epoch in range(self.config.num_epochs):
                key, _k = jax.random.split(key)
                state, l, info = self.update_step_fn(state, _k, experiences)
                loss += l

            loss /= self.config.num_epochs
            info["total_loss"] = loss
            return state, info

        samples = [b.sample() for b in buffers]
        self.state, info = fn(self.state, self.nextkey(), samples)
        return info

    def train(
        self,
        env: list[ParallelEnv | SubProcVecParallelEnv],
        n_env_steps: int,
        callbacks: list,
    ):
        return train_population(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.PARALLEL,
            EnvProcs.ONE if self.n_envs == 1 else EnvProcs.MANY,
            saver=self.saver,
            callbacks=callbacks,
        )

    def resume(
        self,
        env: list[ParallelEnv | SubProcVecParallelEnv],
        n_env_steps: int,
        callbacks: list,
    ):
        step, self.state = self.saver.restore_latest_step(self.state)

        return train_population(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.PARALLEL,
            EnvProcs.ONE if self.n_envs == 1 else EnvProcs.MANY,
            start_step=step,
            saver=self.saver,
            callbacks=callbacks,
        )
