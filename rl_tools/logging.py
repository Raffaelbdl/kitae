import jax
import numpy as np

from rl_tools.callbacks.callback import Callback


def init_episode_return(
    first_observation: jax.Array | dict[str, jax.Array],
    parallel: bool,
    vectorized: bool,
):
    if not parallel and not vectorized:
        return 0.0
    elif not parallel and vectorized:
        return np.zeros((first_observation.shape[0],))
    elif parallel and not vectorized:
        return 0.0
    elif parallel and vectorized:
        return np.zeros((np.array(list(first_observation.values())[0]).shape[0],))
    else:
        raise NotImplementedError


class Logger:
    def __init__(
        self, callbacks: list[Callback], *, parallel: bool, vectorized: bool
    ) -> None:
        self.callbacks = callbacks

        self.parallel = parallel
        self.vectorized = vectorized

        self.logs: dict = {"step": 0}

    def init_logs(self, first_observation: jax.Array | dict[str, jax.Array]):
        if isinstance(first_observation, tuple):
            self.logs["episode_return"] = [
                init_episode_return(obs, self.parallel, self.vectorized)
                for obs in first_observation
            ]
        else:
            self.logs["episode_return"] = init_episode_return(
                first_observation, self.parallel, self.vectorized
            )

    def __getitem__(self, key):
        return self.logs[key]

    def __setitem__(self, key, value):
        self.logs[key] = value

    def update(self, value: dict):
        self.logs |= value

    def get_logs(self) -> dict:
        for callback in self.callbacks:
            self.logs |= callback.get_logs()
        return self.logs
