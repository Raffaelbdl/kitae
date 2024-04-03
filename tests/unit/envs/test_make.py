import pytest
import gymnasium as gym

from kitae.envs.make import make_env
from kitae.envs.make import make_single_env
from kitae.envs.make import wrap_single_env
from kitae.envs.make import make_vec_env


def test_make_env():
    env = make_env("Dummy", 0, capture_video=False, run_name="")()
    with pytest.raises(AttributeError):
        env.get_wrapper_attr("recording")

    env = make_env("Dummy", 0, capture_video=True, run_name="")()
    env.get_wrapper_attr("recording")  # won't raise error


def test_make_single_env():
    env = make_single_env("Dummy", capture_video=False, run_name="")
    assert env.get_wrapper_attr("num_envs") == 1
    assert isinstance(env, gym.wrappers.RecordEpisodeStatistics)


def test_wrap_single_env():
    env = gym.make("Dummy")
    env = wrap_single_env(env)
    assert env.get_wrapper_attr("num_envs") == 1
    assert isinstance(env, gym.wrappers.RecordEpisodeStatistics)


def test_vec_env():
    n_envs = 6
    env = make_vec_env("Dummy", n_envs, capture_video=False, run_name="")
    assert env.get_wrapper_attr("num_envs") == 6
    assert isinstance(env, gym.wrappers.RecordEpisodeStatistics)
