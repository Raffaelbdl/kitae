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

    def on_update_end(self, callback_data: CallbackData):
        self.wandb.log(callback_data.logs)
