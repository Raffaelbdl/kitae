from dataclasses import dataclass, field


@dataclass
class OnEpisodeEndData:
    agent_id: int = field(default=0)
    episode_return: float = field(default=0.0)


@dataclass
class CallbackData:
    logs: dict = field(default_factory=lambda: {})

    episode_end_data: OnEpisodeEndData = field(default=OnEpisodeEndData())

    @staticmethod
    def on_episode_end(agent_id: int, episode_return: float):
        return CallbackData(episode_end_data=OnEpisodeEndData(agent_id, episode_return))


class Callback:
    def on_train_start(self, callback_data: CallbackData):
        pass

    def on_train_end(self, callback_data: CallbackData):
        pass

    def on_episode_end(self, callback_data: CallbackData):
        pass

    def on_update_start(self, callback_data: CallbackData):
        pass

    def on_update_end(self, callback_data: CallbackData):
        pass

    def get_logs(self) -> dict:
        return {}


def on_train_start(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_train_start(callback_data)


def on_train_end(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_train_end(callback_data)


def on_episode_end(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_episode_end(callback_data)


def on_update_start(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_update_start(callback_data)


def on_update_end(callbacks: list[Callback], callback_data: CallbackData):
    for c in callbacks:
        c.on_update_end(callback_data)
