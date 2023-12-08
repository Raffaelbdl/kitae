import ml_collections

from evals.eval_callbacks import TimeCallback

from rl.algos import dqn
from rl.transformation import normalize_frames

from evals.eval_envs import (
    make_cartpole,
    make_cartpole_parallel,
    make_cartpole_parallel_vector,
    make_cartpole_vector,
    make_pong_vector,
)


def dqn_cartpole():
    TASK_ID = "CartPole-v1"
    env, env_config = make_cartpole()
    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration": 0.1,
            "max_buffer_size": -1,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 10**6,
            "shared_encoder": False,
            "learning_rate_annealing": False,
            "max_grad_norm": 10,
            "save_frequency": -1,
        }
    )

    model = dqn.DQN(0, config, tabulate=True)

    callbacks = [TimeCallback(TASK_ID, env_config.n_envs)]
    model.train(env, config["n_env_steps"], callbacks)

    for c in callbacks:
        c.write_results("dqn_cartpole")


def dqn_cartpole_parallel():
    TASK_ID = "CartPole-Parallel"
    env, env_config = make_cartpole_parallel()
    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration": 0.1,
            "max_buffer_size": -1,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 10**6,
            "shared_encoder": False,
            "learning_rate_annealing": False,
            "max_grad_norm": 10,
            "save_frequency": -1,
        }
    )

    model = dqn.DQN(0, config, tabulate=True)

    callbacks = [TimeCallback(TASK_ID, env_config.n_envs)]
    model.train(env, config["n_env_steps"], callbacks)

    for c in callbacks:
        c.write_results("dqn_cartpole_parallel")


def dqn_cartpole_vector():
    TASK_ID = "CartPole-v1"
    env, env_config = make_cartpole_vector(32)
    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration": 0.1,
            "max_buffer_size": -1,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 10**6,
            "shared_encoder": False,
            "learning_rate_annealing": False,
            "max_grad_norm": 10,
            "save_frequency": -1,
        }
    )

    model = dqn.DQN(0, config, tabulate=True)

    callbacks = [TimeCallback(TASK_ID, env_config.n_envs)]
    model.train(env, config["n_env_steps"], callbacks)

    for c in callbacks:
        c.write_results("dqn_cartpole_vector")


def dqn_cartpole_parallel_vector():
    TASK_ID = "CartPole-Parallel"
    env, env_config = make_cartpole_parallel_vector(16)

    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration": 0.1,
            "max_buffer_size": -1,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 10**6,
            "shared_encoder": False,
            "learning_rate_annealing": False,
            "max_grad_norm": 10,
            "save_frequency": -1,
        }
    )

    model = dqn.DQN(0, config, tabulate=True)

    callbacks = [TimeCallback(TASK_ID, env_config.n_envs)]
    model.train(env, config["n_env_steps"], callbacks)

    for c in callbacks:
        c.write_results("dqn_cartpole_parallel_vector")


def dqn_pong_vector():
    TASK_ID = "Pong-v5"
    env, env_config = make_pong_vector(32)
    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration": 0.1,
            "max_buffer_size": -1,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 10**6,
            "shared_encoder": False,
            "learning_rate_annealing": False,
            "max_grad_norm": 10,
            "save_frequency": -1,
        }
    )

    model = dqn.DQN(0, config, tabulate=True, preprocess_fn=normalize_frames)

    callbacks = [TimeCallback(TASK_ID, env_config.n_envs)]
    model.train(env, config["n_env_steps"], callbacks)

    for c in callbacks:
        c.write_results("dqn_cartpole_vector")
