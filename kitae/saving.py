from contextlib import AbstractContextManager
from pathlib import Path
import time
from typing import Any

import cloudpickle
from flax.training import orbax_utils, train_state
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter

from kitae.interface import IAgent
from kitae.config import dump_algo_config

from save.checkpoint import FlaxCheckpointer


def default_run_name(env_id: str) -> str:
    """Generates a default name for a run."""
    return f"{env_id}/{env_id}__{int(time.time())}"


class Saver:
    """Saver class for agents.

    Handles saving during training and restoring from checkpoints.

    Args:
        dir (Path): The path where the agent is saved.
        writer (SummaryWriter): A SummaryWriter to log metrics.
    """

    def __init__(self, dir: str | Path, agent: IAgent) -> None:
        """Initializes a Saver instance for an agent.

        Args:
            dir: A string or path-like path to the saving directory.
            agent: The parent agent.
        """
        self.dir = dir = Path(dir).absolute()
        self.ckptr = FlaxCheckpointer(self.dir.joinpath("checkpoints"))

        self.save_base_data(dir, agent)

        self.writer = SummaryWriter(dir.joinpath("logs"))

    def save_base_data(self, dir_path: Path, agent: IAgent) -> None:
        """Saves the original information.

        Args:
            dir_path (Path): The path where the agent must be saved.
            agent (IAgent): The agent whose information must be saved.
        """
        config_dir = dir_path.joinpath("config")
        dump_algo_config(agent.config, config_dir)

        extra_info = {
            "run_name": agent.run_name,
            "preprocess_fn": agent.preprocess_fn,
        }
        with config_dir.joinpath("extra").open("wb") as f:
            cloudpickle.dump(extra_info, f)

    def save(self, step: int, ckpt: dict[str, Any]) -> None:
        """Saves a checkpoint.

        Args:
            step (int): The current step of the environment.
            ckpt (dict): A checkpoint to save.
        """
        self.ckptr.save(ckpt, step)

    def restore_latest_step(self) -> tuple[int, dict[str, Any]]:
        """Restores the state of an agent from the latest step.

        Args:
            base_state_dict (dict): The state of the agent before restoring it.

        Returns:
            The latest step as an int and the restored state of the agent.
        """
        return self.ckptr.restore_last()


class SaverContext(AbstractContextManager):
    """A context to ensures that the agent state is saved when the training is interrupted.

    Tip:
        Typical usage::

            with SaverContext(saver, save_frequency) as s:
                for step in range(n_env_steps):
                    ...

                    s.update(step, agent.state)

    Attributes:
        saver (Saver): A saver instance.
    """

    def __init__(self, saver: Saver, save_frequency: int) -> None:
        """Initializes a SaverContext instance.

        Args:
            saver (Saver): A Saver instance.
            save_frequency (int): The frequency at which the agent's state must be saved.
        """
        super().__init__()
        self.saver = saver

        self.save_frequency = save_frequency
        self.cur_step = 0
        self.cur_state = None

    def update(self, step: int, state: train_state.TrainState) -> None:
        """Informs the Saver of a new state, and saves it when necessary.

        Args:
            step (int): The current step of the environment.
            state (TrainState): The state of the agent.
        """
        self.cur_step = step
        self.cur_state = state

        if self.save_frequency < 0:
            return

        if step % self.save_frequency != 0:
            return

        self.saver.save(step, state)

    def __exit__(self, *args, **kwargs) -> bool | None:
        """Saves the agent's state when the context is exited."""
        if self.cur_state is None:
            return

        self.saver.save(self.cur_step, self.cur_state)
