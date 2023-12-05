from typing import Any, TypeVar

from envpool.python.api import EnvPool
import gymnasium as gym
import pettingzoo
import vec_parallel_env

GymEnv = gym.Env
EnvPoolEnv = EnvPool
ParallelEnv = pettingzoo.ParallelEnv
SubProcVecParallelEnv = vec_parallel_env.SubProcVecParallelEnv

EnvLike = (
    gym.Env | EnvPool | pettingzoo.ParallelEnv | vec_parallel_env.SubProcVecParallelEnv
)

import flax
import jax

DictArray = dict[Any, jax.Array]

ActionType = TypeVar("ActionType")
ObsType = TypeVar("ObsType")
Params = flax.core.FrozenDict
