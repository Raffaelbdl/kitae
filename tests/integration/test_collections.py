import os
import shutil

from kitae.algos.collections import dqn, ppo, sac, td3
from kitae.config import EnvConfig, TrainConfig, UpdateConfig, AlgoConfig
from kitae.envs.make import make_vec_env


def test_dqn_loop():
    env = make_vec_env("CartPole-v1", 1, capture_video=False, run_name=None)
    env_cfg = EnvConfig(
        "CartPole-v1", env.single_observation_space, env.single_action_space, 1, 1
    )

    agent = dqn.DQN(
        "test-dqn",
        AlgoConfig(
            seed=0,
            algo_params=dqn.DQNParams(),
            update_cfg=UpdateConfig(1e-3, False, 1.0, int(1e5), 64, 1, False),
            train_cfg=TrainConfig(200, 100),
            env_cfg=env_cfg,
        ),
    )
    assert os.path.isfile("./runs/test-dqn/agent_info/config/algo_config.yaml")
    assert os.path.isfile("./runs/test-dqn/agent_info/config/algo_params_type")
    assert os.path.isfile("./runs/test-dqn/agent_info/config/env_config")
    assert os.path.isfile("./runs/test-dqn/agent_info/extra")
    assert os.path.isdir("./runs/test-dqn/checkpoints")
    assert len(os.listdir("./runs/test-dqn/checkpoints")) == 0

    agent.train(env, agent.config.train_cfg.n_env_steps)
    assert os.path.isfile("./runs/test-dqn/checkpoints/100/checkpoint.safetensors")
    assert os.path.isfile("./runs/test-dqn/checkpoints/200/checkpoint.safetensors")

    shutil.rmtree("./runs/test-dqn")


def test_ppo_loop():
    env = make_vec_env("CartPole-v1", 1, capture_video=False, run_name=None)
    env_cfg = EnvConfig(
        "CartPole-v1", env.single_observation_space, env.single_action_space, 1, 1
    )

    agent = ppo.PPO(
        "test-ppo",
        AlgoConfig(
            seed=0,
            algo_params=ppo.PPOParams(),
            update_cfg=UpdateConfig(1e-3, False, 1.0, int(128), 64, 1, False),
            train_cfg=TrainConfig(200, 100),
            env_cfg=env_cfg,
        ),
    )
    assert os.path.isfile("./runs/test-ppo/agent_info/config/algo_config.yaml")
    assert os.path.isfile("./runs/test-ppo/agent_info/config/algo_params_type")
    assert os.path.isfile("./runs/test-ppo/agent_info/config/env_config")
    assert os.path.isfile("./runs/test-ppo/agent_info/extra")
    assert os.path.isdir("./runs/test-ppo/checkpoints")
    assert len(os.listdir("./runs/test-ppo/checkpoints")) == 0

    agent.train(env, agent.config.train_cfg.n_env_steps)
    assert os.path.isfile("./runs/test-ppo/checkpoints/100/checkpoint.safetensors")
    assert os.path.isfile("./runs/test-ppo/checkpoints/200/checkpoint.safetensors")

    shutil.rmtree("./runs/test-ppo")


def test_sac_loop():
    env = make_vec_env("HalfCheetah-v4", 1, capture_video=False, run_name=None)
    env_cfg = EnvConfig(
        "HalfCheetah-v4", env.single_observation_space, env.single_action_space, 1, 1
    )

    agent = sac.SAC(
        "test-sac",
        AlgoConfig(
            seed=0,
            algo_params=sac.SACParams(),
            update_cfg=UpdateConfig(1e-3, False, 1.0, int(1e-5), 64, 1, False),
            train_cfg=TrainConfig(200, 100),
            env_cfg=env_cfg,
        ),
    )
    assert os.path.isfile("./runs/test-sac/agent_info/config/algo_config.yaml")
    assert os.path.isfile("./runs/test-sac/agent_info/config/algo_params_type")
    assert os.path.isfile("./runs/test-sac/agent_info/config/env_config")
    assert os.path.isfile("./runs/test-sac/agent_info/extra")
    assert os.path.isdir("./runs/test-sac/checkpoints")
    assert len(os.listdir("./runs/test-sac/checkpoints")) == 0

    agent.train(env, agent.config.train_cfg.n_env_steps)
    assert os.path.isfile("./runs/test-sac/checkpoints/100/checkpoint.safetensors")
    assert os.path.isfile("./runs/test-sac/checkpoints/200/checkpoint.safetensors")

    shutil.rmtree("./runs/test-sac")


def test_td3_loop():
    env = make_vec_env("HalfCheetah-v4", 1, capture_video=False, run_name=None)
    env_cfg = EnvConfig(
        "HalfCheetah-v4", env.single_observation_space, env.single_action_space, 1, 1
    )

    agent = td3.TD3(
        "test-td3",
        AlgoConfig(
            seed=0,
            algo_params=td3.TD3Params(),
            update_cfg=UpdateConfig(1e-3, False, 1.0, int(1e-5), 64, 1, False),
            train_cfg=TrainConfig(200, 100),
            env_cfg=env_cfg,
        ),
    )
    assert os.path.isfile("./runs/test-td3/agent_info/config/algo_config.yaml")
    assert os.path.isfile("./runs/test-td3/agent_info/config/algo_params_type")
    assert os.path.isfile("./runs/test-td3/agent_info/config/env_config")
    assert os.path.isfile("./runs/test-td3/agent_info/extra")
    assert os.path.isdir("./runs/test-td3/checkpoints")
    assert len(os.listdir("./runs/test-td3/checkpoints")) == 0

    agent.train(env, agent.config.train_cfg.n_env_steps)
    assert os.path.isfile("./runs/test-td3/checkpoints/100/checkpoint.safetensors")
    assert os.path.isfile("./runs/test-td3/checkpoints/200/checkpoint.safetensors")

    shutil.rmtree("./runs/test-td3")
