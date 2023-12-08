import ml_collections

from evals.eval_callbacks import TimeCallback

from rl.algos import ppo
from rl.transformation import normalize_frames

from evals.eval_envs import (
    make_cartpole,
    make_cartpole_parallel,
    make_cartpole_parallel_vector,
    make_cartpole_vector,
    make_pong_vector,
)


def ppo_cartpole():
    TASK_ID = "CartPole-v1"
    env, env_config = make_cartpole()
    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_eps": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 64,
            "batch_size": 32,
            "num_epochs": 4,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 10**6,
            "shared_encoder": False,
            "save_frequency": -1,
        }
    )

    model = ppo.PPO(0, config, tabulate=True)

    callbacks = [TimeCallback(TASK_ID, env_config.n_envs)]
    model.train(env, config["n_env_steps"], callbacks)

    for c in callbacks:
        c.write_results("ppo_cartpole")


def ppo_cartpole_parallel():
    TASK_ID = "CartPole-Parallel"
    env, env_config = make_cartpole_parallel()
    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_eps": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 64,
            "batch_size": 32,
            "num_epochs": 4,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 10**6,
            "shared_encoder": False,
            "save_frequency": -1,
        }
    )

    model = ppo.PPO(0, config, tabulate=True)

    callbacks = [TimeCallback(TASK_ID, env_config.n_envs)]
    model.train(env, config["n_env_steps"], callbacks)

    for c in callbacks:
        c.write_results("ppo_cartpole_parallel")


def ppo_cartpole_vector():
    TASK_ID = "CartPole-v1"
    env, env_config = make_cartpole_vector(16)
    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_eps": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 64,
            "batch_size": 32,
            "num_epochs": 4,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 10**6,
            "shared_encoder": False,
            "save_frequency": -1,
        }
    )

    model = ppo.PPO(0, config, tabulate=True)

    callbacks = [TimeCallback(TASK_ID, env_config.n_envs)]
    model.train(env, config["n_env_steps"], callbacks)

    for c in callbacks:
        c.write_results("ppo_cartpole_vector")


def ppo_cartpole_parallel_vector():
    TASK_ID = "CartPole-Parallel"
    env, env_config = make_cartpole_parallel_vector(16)
    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_eps": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 64,
            "batch_size": 32,
            "num_epochs": 4,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 10**6,
            "shared_encoder": False,
            "save_frequency": -1,
        }
    )

    model = ppo.PPO(0, config, tabulate=True)

    callbacks = [TimeCallback(TASK_ID, env_config.n_envs)]
    model.train(env, config["n_env_steps"], callbacks)

    for c in callbacks:
        c.write_results("ppo_cartpole_parallel_vector")


def ppo_pong_vector():
    TASK_ID = "Pong-v5"
    env, env_config = make_pong_vector(32)
    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_eps": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 128,
            "batch_size": 256,
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 10**6,
            "shared_encoder": False,
            "save_frequency": -1,
        }
    )

    model = ppo.PPO(0, config, tabulate=True, preprocess_fn=normalize_frames)

    callbacks = [TimeCallback(TASK_ID, env_config.n_envs)]
    model.train(env, config["n_env_steps"], callbacks)

    for c in callbacks:
        c.write_results("ppo_cartpole_vector")
