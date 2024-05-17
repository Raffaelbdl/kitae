from contextlib import AbstractContextManager
import time
from typing import Any

from save.checkpoint import Checkpointer


def default_run_name(env_id: str) -> str:
    """Generates a default name for a run."""
    return f"{env_id}/{env_id}__{int(time.time())}"


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

    def __init__(self, checkpointer: Checkpointer, save_frequency: int) -> None:
        """Initializes a SaverContext instance.

        Args:
            saver (Saver): A Saver instance.
            save_frequency (int): The frequency at which the agent's state must be saved.
        """
        super().__init__()
        self.checkpointer = checkpointer

        self.save_frequency = save_frequency
        self.cur_step = 0
        self.cur_state = None

    def update(self, step: int, state: Any) -> None:
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

        self.checkpointer.save(state, step)

    def __exit__(self, *args, **kwargs) -> bool | None:
        """Saves the agent's state when the context is exited."""
        if self.cur_state is None:
            return

        self.checkpointer.save(self.cur_state, self.cur_step)
