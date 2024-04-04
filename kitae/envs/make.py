from typing import Callable

import gymnasium as gym
import pettingzoo


def make_env(
    env_id: str, idx: int, *, capture_video: bool, run_name: str, **env_kwargs
) -> Callable:
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **env_kwargs)
        max_episode_steps = env_kwargs.pop("max_episode_steps", 0)
        if max_episode_steps:
            env = gym.wrappers.TimeLimit(env, max_episode_steps)

        return env

    return thunk


def make_single_env(
    env_id: str, *, capture_video: bool, run_name: str, **env_kwargs
) -> gym.vector.SyncVectorEnv:
    env = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id, 0, capture_video=capture_video, run_name=run_name, **env_kwargs
            )
        ]
    )
    return gym.wrappers.RecordEpisodeStatistics(env)


def wrap_single_env(env: gym.Env) -> gym.vector.SyncVectorEnv:
    env = gym.vector.SyncVectorEnv([lambda: env])
    return gym.wrappers.RecordEpisodeStatistics(env)


from kitae.wrapper import SubProcVecParallelEnvCompatibility
from kitae.envs.wrappers.vector import SubProcVecParallelEnv
from kitae.envs.wrappers.record_episode_statistics import (
    ParallelRecordEpisodeStatistics,
)


def wrap_single_parallel_env(env: pettingzoo.ParallelEnv) -> SubProcVecParallelEnv:
    env = SubProcVecParallelEnv([lambda: env])
    env = SubProcVecParallelEnvCompatibility(env)
    return ParallelRecordEpisodeStatistics(env)


def wrap_multiple_parallel_env(
    env: pettingzoo.ParallelEnv, n_envs: int
) -> SubProcVecParallelEnv:
    env = SubProcVecParallelEnv([lambda: env for _ in range(n_envs)])
    env = SubProcVecParallelEnvCompatibility(env)
    return ParallelRecordEpisodeStatistics(env)


def make_vec_env(
    env_id: str, n_envs: int, *, capture_video: bool, run_name: str, **env_kwargs
) -> gym.vector.AsyncVectorEnv:
    env = gym.vector.AsyncVectorEnv(
        [
            make_env(
                env_id, i, capture_video=capture_video, run_name=run_name, **env_kwargs
            )
            for i in range(n_envs)
        ]
    )
    return gym.wrappers.RecordEpisodeStatistics(env)
