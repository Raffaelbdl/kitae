"""Contains the base classes for callbacks."""

from dataclasses import dataclass, field


@dataclass
class OnEpisodeEndData:
    agent_id: int = 0
    episode_return: float = 0


@dataclass
class CallbackData:
    logs: dict = field(default_factory=lambda: {})
    episode_end_data: OnEpisodeEndData = field(
        default_factory=lambda: OnEpisodeEndData()
    )

    @classmethod
    def on_episode_end(cls, agent_id: int, episode_return: float):
        return cls(episode_end_data=OnEpisodeEndData(agent_id, episode_return))


class Callback:
    """Base class for callbacks."""

    def on_train_start(self, callback_data: CallbackData):
        """Called when the training starts."""
        pass

    def on_train_end(self, callback_data: CallbackData):
        """Called when the training ends."""
        pass

    def on_episode_end(self, callback_data: CallbackData):
        """Called when any environment is terminated."""
        pass

    def on_update_start(self, callback_data: CallbackData):
        """Called when the update starts."""
        pass

    def on_update_end(self, callback_data: CallbackData):
        """Called when the update ends."""
        pass

    def get_logs(self) -> dict:
        """Returns information to log."""
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
