from contextlib import AbstractContextManager
from types import TracebackType

from flax.training import orbax_utils, train_state
import orbax.checkpoint


class Saver:
    def __init__(self, dir: str) -> None:
        self.ckptr = orbax.checkpoint.PyTreeCheckpointer()
        self.options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=None, create=True
        )
        self.ckpt_manager = orbax.checkpoint.CheckpointManager(
            dir, self.ckptr, self.options
        )

    def save(self, step: int, state: train_state.TrainState):
        ckpt = {"model": state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.ckpt_manager.save(step, ckpt, save_kwargs={"save_args": save_args})

    def restore_latest_step(self, base_train_state: train_state.TrainState):
        step = self.ckpt_manager.latest_step()
        return (
            step,
            self.ckpt_manager.restore(step, items={"model": base_train_state})["model"],
        )


class SaverContext(AbstractContextManager):
    def __init__(self, saver: Saver, save_frequency: int) -> None:
        super().__init__()
        self.saver = saver

        self.save_frequency = save_frequency
        self.cur_step = 0
        self.cur_state = None

    def update(self, step: int, state: train_state.TrainState):
        self.cur_step = step
        self.cur_state = state
        if step % self.save_frequency != 0:
            return

        self.saver.save(step, state)

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        if self.cur_state is None:
            return

        self.saver.save(self.cur_step, self.cur_state)
