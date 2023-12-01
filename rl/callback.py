from dataclasses import dataclass, field


@dataclass
class CallbackData:
    episode_return: float = field(default=0.0)


class Callback:
    def on_train_start(self, callback_data: CallbackData):
        pass

    def on_train_end(self, callback_data: CallbackData):
        pass

    def on_episode_start(self, callback_data: CallbackData):
        pass

    def on_episode_end(self, callback_data: CallbackData):
        pass

    def on_update_start(self, callback_data: CallbackData):
        pass

    def on_update_end(self, callback_data: CallbackData):
        pass


def on_train_start(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_train_start(callback_data)


def on_train_end(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_train_end(callback_data)


def on_episode_start(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_episode_start(callback_data)


def on_episode_end(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_episode_end(callback_data)


def on_update_start(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_update_start(callback_data)


def on_update_end(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_update_end(callback_data)
