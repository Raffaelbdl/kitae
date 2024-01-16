"""Contains the episode return callback."""

from collections import deque

from rl.callbacks.callback import Callback, CallbackData


class EpisodeReturnCallback(Callback):
    """EpisodeReturnCallback class

    This callbacks keep tracks of the previous episodes returns during training.
    It logs an average episode return.
    """

    def __init__(self, *, population_size: int = 1, buffer_length: int = 20) -> None:
        super().__init__()

        self.population_size = population_size
        self.episode_return_buffer = [
            deque(maxlen=buffer_length) for _ in range(population_size)
        ]

    def on_episode_end(self, callback_data: CallbackData) -> None:
        data = callback_data.episode_end_data
        self.episode_return_buffer[data.agent_id].append(data.episode_return)

    def get_logs(self) -> dict:
        logs = {}
        for i in range(len(self.episode_return_buffer)):
            buffer = self.episode_return_buffer[i]
            if len(buffer) > 0:
                if self.population_size > 1:
                    logs[f"agent_{i}/mean_episode_return"] = sum(buffer) / len(buffer)
                else:
                    logs["mean_episode_return"] = sum(buffer) / len(buffer)
        return logs
