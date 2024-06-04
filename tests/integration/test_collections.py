import os
from pathlib import Path
import shutil

from kitae.algos.collections import dqn, ppo, sac, td3
from kitae.config import EnvConfig, TrainConfig, UpdateConfig, AlgoConfig
from kitae.envs.make import make_vec_env


def get_collections(name: str):
    if name == "dqn":
        return dqn.DQN, dqn.DQNParams
    if name == "ppo":
        return ppo.PPO, ppo.PPOParams
    if name == "sac":
        return sac.SAC, sac.SACParams
    if name == "td3":
        return td3.TD3, td3.TD3Params
    raise ValueError


def collections_loop(name: str, env_id: str):
    path = Path("./runs").joinpath(name).resolve()
    if os.path.isdir(path):
        shutil.rmtree(path)

    env = make_vec_env(env_id, 1, capture_video=False, run_name=None)
    env_cfg = EnvConfig(
        env_id, env.single_observation_space, env.single_action_space, 1, 1
    )

    _cls, _params = get_collections(name)
    agent = _cls(
        name,
        AlgoConfig(
            seed=0,
            algo_params=_params(),
            update_cfg=UpdateConfig(1e-3, False, 1.0, 128, 64, 1, False),
            train_cfg=TrainConfig(200, 100),
            env_cfg=env_cfg,
        ),
    )

    assert path.joinpath("agent_info", "config", "algo_config.yaml").exists()
    assert path.joinpath("agent_info", "config", "algo_params_type").exists()
    assert path.joinpath("agent_info", "config", "env_config").exists()
    assert path.joinpath("agent_info", "extra").exists()
    assert path.joinpath("checkpoints").exists()
    assert len(os.listdir(path.joinpath("checkpoints"))) == 0

    agent.train(env, agent.config.train_cfg.n_env_steps)
    assert path.joinpath("checkpoints", "100", "checkpoint.safetensors").exists()
    assert path.joinpath("checkpoints", "200", "checkpoint.safetensors").exists()

    env = make_vec_env(env_id, 1, capture_video=False, run_name=None)
    new_agent = _cls.unserialize(path)

    new_agent.resume(env, 300)
    assert path.joinpath("checkpoints", "300", "checkpoint.safetensors").exists()

    new_agent.restore(200)

    shutil.rmtree(path)

    return True


def test_dqn_loop():
    assert collections_loop("dqn", "CartPole-v1")


def test_ppo_loop():
    assert collections_loop("ppo", "CartPole-v1")
    assert collections_loop("ppo", "HalfCheetah-v4")


def test_sac_loop():
    assert collections_loop("sac", "HalfCheetah-v4")


def test_td3_loop():
    assert collections_loop("td3", "HalfCheetah-v4")
