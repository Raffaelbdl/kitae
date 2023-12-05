from typing import Any, Optional, Union, Dict, Sequence, List

import flatdict
from wandb.sdk.lib.paths import StrPath
from wandb.sdk.wandb_settings import Settings

from rl.callbacks.callback import Callback, CallbackData


class WandbCallback(Callback):
    def __init__(
        self,
        job_type: Optional[str] = None,
        dir: Optional[StrPath] = None,
        config: Union[Dict, str, None] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        reinit: Optional[bool] = None,
        tags: Optional[Sequence] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        magic: Optional[Union[dict, str, bool]] = None,
        config_exclude_keys: Optional[List[str]] = None,
        config_include_keys: Optional[List[str]] = None,
        anonymous: Optional[str] = None,
        mode: Optional[str] = None,
        allow_val_change: Optional[bool] = None,
        resume: Optional[Union[bool, str]] = None,
        force: Optional[bool] = None,
        tensorboard: Optional[bool] = None,  # alias for sync_tensorboard
        sync_tensorboard: Optional[bool] = None,
        monitor_gym: Optional[bool] = None,
        save_code: Optional[bool] = None,
        id: Optional[str] = None,
        settings: Union[Settings, Dict[str, Any], None] = None,
    ) -> None:
        Callback.__init__(self)

        import wandb

        self.wandb = wandb
        self.wandb.init(
            job_type=job_type,
            dir=dir,
            config=config,
            project=project,
            entity=entity,
            reinit=reinit,
            tags=tags,
            group=group,
            name=name,
            notes=notes,
            magic=magic,
            config_exclude_keys=config_exclude_keys,
            config_include_keys=config_include_keys,
            anonymous=anonymous,
            mode=mode,
            allow_val_change=allow_val_change,
            resume=resume,
            force=force,
            tensorboard=tensorboard,
            sync_tensorboard=sync_tensorboard,
            monitor_gym=monitor_gym,
            save_code=save_code,
            id=id,
            settings=settings,
        )

    def on_update_end(self, callback_data: CallbackData):
        self.wandb.log(dict(flatdict.FlatDict(callback_data.logs, delimiter="/")))

    def on_train_end(self, callback_data: CallbackData):
        self.wandb.finish()
