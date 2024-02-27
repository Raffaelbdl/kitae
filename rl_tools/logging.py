import jax
import numpy as np

from rl_tools.base import EnvType, EnvProcs
from rl_tools.callbacks.callback import Callback


def init_episode_return(
    first_observation: jax.Array | dict[str, jax.Array],
    env_type: EnvType,
    env_procs: EnvProcs,
):
    if env_type == EnvType.SINGLE and env_procs == EnvProcs.ONE:
        return 0.0
    elif env_type == EnvType.SINGLE and env_procs == EnvProcs.MANY:
        return np.zeros((first_observation.shape[0],))
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.ONE:
        return 0.0
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.MANY:
        return np.zeros((np.array(list(first_observation.values())[0]).shape[0],))
    else:
        raise NotImplementedError


class Logger:
    def __init__(
        self, callbacks: list[Callback], *, env_type: EnvType, env_procs: EnvProcs
    ) -> None:
        self.callbacks = callbacks

        self.env_type = env_type
        self.env_procs = env_procs

        self.logs: dict = {"step": 0}

    def init_logs(self, first_observation: jax.Array | dict[str, jax.Array]):
        if isinstance(first_observation, tuple):
            self.logs["episode_return"] = [
                init_episode_return(obs, self.env_type, self.env_procs)
                for obs in first_observation
            ]
        else:
            self.logs["episode_return"] = init_episode_return(
                first_observation, self.env_type, self.env_procs
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
