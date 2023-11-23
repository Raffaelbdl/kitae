import ml_collections

from rl.algos import ppo
from tests.integration.env_fn import (
    create_atari,
    create_cartpole,
    create_atari_envpool,
    create_cartpole_envpool,
)

# model did create, run 1000 steps and finish training without error


def test_no_error_ppo_run_visual_non_vectorized():
    env = create_atari()

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "clip_eps": 0.1,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 128,
            "batch_size": 64,
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": False,
        }
    )

    model = ppo.PPO(0, env, config)
    ppo.train(0, model, env, config.n_env_steps)

    assert True


def test_no_error_ppo_run_visual_non_vectorized_shared_encoder():
    env = create_atari()

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "clip_eps": 0.1,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 128,
            "batch_size": 64,
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": True,
        }
    )

    model = ppo.PPO(0, env, config)
    ppo.train(0, model, env, config.n_env_steps)

    assert True


def test_no_error_ppo_run_vector_non_vectorized():
    env = create_cartpole()

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "clip_eps": 0.1,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 128,
            "batch_size": 64,
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": False,
        }
    )

    model = ppo.PPO(0, env, config)
    ppo.train(0, model, env, config.n_env_steps)

    assert True


def test_no_error_ppo_run_vector_non_vectorized_shared_encoder():
    env = create_cartpole()

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "clip_eps": 0.1,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 128,
            "batch_size": 64,
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": True,
        }
    )

    model = ppo.PPO(0, env, config)
    ppo.train(0, model, env, config.n_env_steps)

    assert True


def test_no_error_ppo_run_visual_vectorized():
    N_ENVS = 16
    env = create_atari_envpool(N_ENVS)

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "clip_eps": 0.1,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 128,
            "batch_size": 64,
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": False,
        }
    )

    model = ppo.PPO(0, env, config, n_envs=N_ENVS)
    ppo.train_vectorized(0, model, env, config.n_env_steps)

    assert True


def test_no_error_ppo_run_visual_vectorized_shared_encoder():
    N_ENVS = 16
    env = create_atari_envpool(N_ENVS)

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "clip_eps": 0.1,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 128,
            "batch_size": 64,
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": True,
        }
    )

    model = ppo.PPO(0, env, config, n_envs=N_ENVS)
    ppo.train_vectorized(0, model, env, config.n_env_steps)

    assert True


def test_no_error_ppo_run_vector_vectorized():
    N_ENVS = 16
    env = create_cartpole_envpool(N_ENVS)

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "clip_eps": 0.1,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 128,
            "batch_size": 64,
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": False,
        }
    )

    model = ppo.PPO(0, env, config, n_envs=N_ENVS)
    ppo.train_vectorized(0, model, env, config.n_env_steps)

    assert True


def test_no_error_ppo_run_vector_vectorized_shared_encoder():
    N_ENVS = 16
    env = create_cartpole_envpool(N_ENVS)

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "clip_eps": 0.1,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 128,
            "batch_size": 64,
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": True,
        }
    )

    model = ppo.PPO(0, env, config, n_envs=N_ENVS)
    ppo.train_vectorized(0, model, env, config.n_env_steps)

    assert True
