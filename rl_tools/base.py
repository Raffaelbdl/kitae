"""Contains the self classes for reinforcement learning."""

from pathlib import Path
from typing import Any, Callable

import cloudpickle
import jax
import yaml

from shaberax.logger import GeneralLogger

from jrd_extensions import Seeded

from rl_tools.interface import IAgent, IBuffer, AlgoType

from rl_tools.train import vectorized_train

from rl_tools.save import Saver
from rl_tools.types import ActionType, ObsType

from ml_collections import ConfigDict
from rl_tools.config import AlgoConfig


from rl_tools.algos.factory import ExperienceTransform
from rl_tools.algos.factory import explore_general_factory
from rl_tools.algos.factory import process_experience_pipeline_factory

from rl_tools.buffer import Experience

import numpy as np


class BaseAgent(IAgent, Seeded):
    def __init__(
        self,
        run_name: str,
        config: AlgoConfig,
        train_state_factory: Callable,
        explore_factory: Callable,
        process_experience_factory: Callable,
        update_step_factory: Callable,
        *,
        preprocess_fn: Callable,
        tabulate: bool = False,
        experience_type: bool = Experience,
    ):
        Seeded.__init__(self, config.seed)

        self.run_name = run_name
        self.config = config

        self.preprocess_fn = preprocess_fn
        self.vectorized = True
        self.parallel = config.env_cfg.n_agents > 1

        self.state = train_state_factory(
            self.nextkey(),
            config,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
        )

        self.explore_fn = explore_general_factory(
            explore_factory(config), self.vectorized, self.parallel
        )
        self.explore_fn = jax.jit(self.explore_fn)

        self.process_experience_fn = process_experience_factory(config)
        self.process_experience_pipeline = process_experience_pipeline_factory(
            self.vectorized, self.parallel, experience_type
        )
        self.process_experience_pipeline = jax.jit(self.process_experience_pipeline)

        self.update_step_fn = update_step_factory(config)

        self.saver = create_saver(self, run_name)

        GeneralLogger.debug("Finished initialization.")

    def explore(self, observation: ObsType) -> tuple[ActionType, np.ndarray]:
        keys = self.interact_keys(observation)
        action, log_prob = self.explore_fn(self.state, keys, observation)
        return np.array(action), log_prob

    def select_action(self, observation: ObsType) -> tuple[ActionType, np.ndarray]:
        return self.explore(observation)

    def update(self, buffer: IBuffer) -> dict:
        sample = buffer.sample(self.config.update_cfg.batch_size)
        GeneralLogger.debug("Sampled")

        experiences = self.process_experience_pipeline(
            [ExperienceTransform(self.process_experience_fn, self.state)],
            key=self.nextkey(),
            experiences=sample,
        )
        GeneralLogger.debug("Processed")

        for _ in range(self.config.update_cfg.n_epochs):
            self.state, info = jax.jit(self.update_step_fn)(
                self.state, self.nextkey(), experiences
            )
        GeneralLogger.debug("Updated")

        return info

    def restore(self, step: int = -1) -> int:
        """Restores the agent's states from the given step."""
        if step < 0:
            latest_step, self.state = self.saver.restore_latest_step(self.state)
            return latest_step

        # TODO Handle specific steps
        raise NotImplementedError("Saving a specific step is not yet implemented.")

    def train(self, env, n_env_steps):
        return vectorized_train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            self.algo_type,
            saver=self.saver,
        )

    def resume(self, env, n_env_steps):
        step, self.state = self.saver.restore_latest_step(self.state)
        return vectorized_train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            self.algo_type,
            saver=self.saver,
            start_step=step,
        )

    def interact_keys(self, observation: ObsType) -> jax.Array | dict[str : jax.Array]:
        if self.parallel:
            return {a: self.nextkey() for a in observation.keys()}
        return self.nextkey()

    @classmethod
    def unserialize(cls, data_dir: str | Path):
        """Creates a new instance of the agent given the save directory.

        Args:
            data_dir: A string or Path to the save directory.
        Returns:
            An instance of the chosen agent.
        """
        path = Path(data_dir)

        config_path = path.joinpath("config")
        with open(config_path, "r") as f:
            config_dict = yaml.load(f, yaml.SafeLoader)
        config = ConfigDict(config_dict)

        extra_path = path.joinpath("extra")
        with open(extra_path, "rb") as f:
            extra = cloudpickle.load(f)

        config.env_cfg = extra.pop("env_config")
        extra["run_name"] = path.parts[-1]

        return cls(config=config, **extra)


def create_saver(self: BaseAgent, run_name: str) -> Saver:
    return Saver(Path("./results").joinpath(run_name).absolute(), self)


class OffPolicyAgent(BaseAgent):
    algo_type = AlgoType.OFF_POLICY
    step = 0

    def should_update(self, step: int, buffer: IBuffer) -> bool:
        self.step = step
        return (
            len(buffer) >= self.config.update_cfg.batch_size
            and step % self.config.algo_params.skip_steps == 0
            and step >= self.config.algo_params.start_step
        )


class OnPolicyAgent(BaseAgent):
    algo_type = AlgoType.ON_POLICY

    def should_update(self, step: int, buffer: IBuffer) -> bool:
        return len(buffer) >= self.config.update_cfg.max_buffer_size
