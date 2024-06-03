"""Contains the self classes for reinforcement learning."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

import jax
import numpy as np
from tensorboardX import SummaryWriter

from shaberax.logger import GeneralLogger
from jrd_extensions import Seeded

from kitae.algos.factory import explore_general_factory
from kitae.algos.experience import ExperiencePipeline
from kitae.buffer import Experience, numpy_stack_experiences
from kitae.config import AlgoConfig, ConfigSerializable
from kitae.interface import IAgent, IBuffer, AlgoType
from kitae.train import vectorized_train
from kitae.types import ActionType, ObsType

from save.checkpoint import PyTreeNodeTrainStateFlaxCheckpointer
from save.serializable import Serializable, SerializableDict, SerializableObject
from save.serializable import save_file, CloudPickleSerializable


@dataclass
class AgentInfo:
    config: AlgoConfig
    extra_info: dict


class AgentSerializable(Serializable):
    @staticmethod
    def serialize(agent_info: AgentInfo, path: Path):
        # path : runs/env_id/env_id__timems/config
        config_path = Path(path).resolve().joinpath("config")
        os.makedirs(config_path, exist_ok=True)

        ConfigSerializable.serialize(agent_info.config, config_path)
        CloudPickleSerializable.serialize(
            agent_info.extra_info, config_path.joinpath("extra")
        )

    @staticmethod
    def unserialize(path: Path) -> AgentInfo:
        # path : runs/env_id/env_id__timems/config
        config_path = Path(path).resolve().joinpath("config")

        config = ConfigSerializable.unserialize(config_path)
        extra_info = CloudPickleSerializable.unserialize(config_path.joinpath("extra"))

        return AgentInfo(config=config, extra_info=extra_info)


class BaseAgent(IAgent, SerializableObject, Seeded):
    serializable_dict = SerializableDict({"run_config": AgentSerializable})

    def __init__(
        self,
        run_name: str,
        config: AlgoConfig,
        train_state_factory: Callable,
        explore_factory: Callable,
        process_experience_factory: Callable,
        update_step_factory: Callable,
        *,
        preprocess_fn: Callable = None,
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
        self.experience_pipeline = ExperiencePipeline(
            [self.process_experience_fn], self.vectorized, self.parallel
        )

        self.update_step_fn = update_step_factory(config)

        path = Path(run_name).resolve()
        self.run_config = AgentInfo(
            self.config, {"run_name": run_name, "preprocess_fn": preprocess_fn}
        )
        save_file(self, path)
        self.checkpointer = PyTreeNodeTrainStateFlaxCheckpointer(
            path.joinpath("checkpoints")
        )
        self.writer = SummaryWriter(path.joinpath("logs"))

        GeneralLogger.debug("Finished initialization.")

    def explore(self, observation: ObsType) -> tuple[ActionType, np.ndarray]:
        keys = self.interact_keys(observation)
        action, log_prob = self.explore_fn(self.state, keys, observation)
        return np.array(action), log_prob

    def select_action(self, observation: ObsType) -> tuple[ActionType, np.ndarray]:
        return self.explore(observation)

    def update(self, buffer: IBuffer) -> dict:
        sample = buffer.sample(self.config.update_cfg.batch_size)
        sample = numpy_stack_experiences(sample)
        GeneralLogger.debug("Buffer Sampled")

        experience = self.experience_pipeline.run(self.state, self.nextkey(), sample)
        GeneralLogger.debug("Experience Processed")

        for _ in range(self.config.update_cfg.n_epochs):
            self.state, info = self.update_step_fn(
                self.state, self.nextkey(), experience
            )
        GeneralLogger.debug("State Updated")

        return info

    def restore(self, step: int = -1) -> int:
        """Restores the agent's states from the given step."""
        if step < 0:
            latest_step, self.state = self.checkpointer.restore_last(self.state)
            return latest_step

        # can raise FileNotFoundError
        self.state = self.checkpointer.restore(self.state, step)
        return step

    def train(self, env, n_env_steps):
        return vectorized_train(
            int(jax.random.key_data(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            self.algo_type,
            checkpointer=self.checkpointer,
            writer=self.writer,
        )

    def resume(self, env, n_env_steps):
        step, self.state = self.checkpointer.restore_last(self.state)
        return vectorized_train(
            int(jax.random.key_data(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            self.algo_type,
            checkpointer=self.checkpointer,
            writer=self.writer,
            start_step=step,
        )

    def interact_keys(self, observation: ObsType) -> jax.Array | dict[str : jax.Array]:
        if self.parallel:
            return {a: self.nextkey() for a in observation.keys()}
        return self.nextkey()

    @classmethod
    def unserialize(cls, path: str | Path):
        """Creates a new instance of the agent given the save directory.

        Args:
            data_dir: A string or Path to the save directory.
        Returns:
            An instance of the chosen agent.
        """
        path = Path(path).resolve()
        agent_info = AgentSerializable.unserialize(path.joinpath("run_config"))
        return cls(config=agent_info.config, **agent_info.extra_info)


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
