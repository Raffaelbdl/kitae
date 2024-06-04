from typing import Any, Callable, NamedTuple, TypeVar

from envpool.python.api import EnvPool
import flax.struct
import flax.traceback_util
from flax.training.train_state import TrainState
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

Array = jax.Array
DictArray = dict[Any, jax.Array]
PRNGKeyArray = jax.Array

ActionType = TypeVar("ActionType")
ObsType = TypeVar("ObsType")
Params = flax.core.FrozenDict
LossDict = dict[str, jax.Array]

ExperienceTuple = NamedTuple
ProcessedExperienceTuple = NamedTuple

ExploreFn = Callable[
    [flax.struct.PyTreeNode, PRNGKeyArray, jax.Array], tuple[jax.Array, jax.Array]
]
ProcessExperienceFn = Callable[
    [flax.struct.PyTreeNode, PRNGKeyArray, ExperienceTuple], ProcessedExperienceTuple
]
UpdateFn = Callable[
    [flax.struct.PyTreeNode, PRNGKeyArray, ProcessedExperienceTuple],
    tuple[flax.struct.PyTreeNode, LossDict],
]
