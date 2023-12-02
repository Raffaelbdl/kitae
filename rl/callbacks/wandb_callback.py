from collections import deque

import ml_collections

from rl.callbacks.callback import Callback, CallbackData


class WandbCallback(Callback):
    def __init__(
        self, project: str, entity: str, config: ml_collections.ConfigDict
    ) -> None:
        Callback.__init__(self)

        import wandb

        self.wandb = wandb
        self.project = project
        self.entity = entity
        self.config = config

        self.wandb.init(project=project, config=config, entity=entity)
        self.episode_returns = deque(maxlen=10)

    def on_episode_end(self, callback_data: CallbackData):
        self.episode_returns.append(callback_data.episode_return)

    def on_update_end(self, callback_data: CallbackData):
        logs = callback_data.logs
        logs["episode_return"] = sum(self.episode_returns) / len(self.episode_returns)
        self.wandb.log(callback_data.logs)
