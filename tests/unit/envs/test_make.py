import pytest
import gymnasium as gym

from rl_tools.envs.make import make_env
from rl_tools.envs.make import make_single_env
from rl_tools.envs.make import wrap_single_env
from rl_tools.envs.make import make_vec_env


def test_make_env():
    env = make_env("Dummy", 0, capture_video=False, run_name="")()
    assert isinstance(env, gym.wrappers.RecordEpisodeStatistics)
    with pytest.raises(AttributeError):
        env.get_wrapper_attr("recording")

    env = make_env("Dummy", 0, capture_video=True, run_name="")()
    assert isinstance(env, gym.wrappers.RecordEpisodeStatistics)
    env.get_wrapper_attr("recording")  # won't raise error


def test_make_single_env():
    env = make_single_env("Dummy", capture_video=False, run_name="")
    assert isinstance(env, gym.vector.SyncVectorEnv)


def test_wrap_single_env():
    env = gym.make("Dummy")
    env = wrap_single_env(env)
    assert isinstance(env, gym.vector.SyncVectorEnv)
    assert isinstance(env.envs[0], gym.wrappers.RecordEpisodeStatistics)


def test_vec_env():
    n_envs = 6
    env = make_vec_env("Dummy", n_envs, capture_video=False, run_name="")
    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert env.num_envs == n_envs
