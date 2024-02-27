import ml_collections

from rl_tools.algos import dqn
from tests.integration.env_fn import (
    create_atari,
    create_cartpole,
    create_atari_envpool,
    create_cartpole_envpool,
)

# model did create, run 1000 steps and finish training without error


def test_no_error_dqn_run_visual_non_vectorized():
    env = create_atari()

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration_coef": 0.1,
            "max_buffer_size": 0,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": False,
        }
    )

    model = dqn.DQN(0, env, config)
    dqn.train(0, model, env, config.n_env_steps)

    assert True


def test_no_error_dqn_run_visual_non_vectorized_shared_encoder():
    env = create_atari()

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration_coef": 0.1,
            "max_buffer_size": 0,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": True,
        }
    )

    model = dqn.DQN(0, env, config)
    dqn.train(0, model, env, config.n_env_steps)

    assert True


def test_no_error_dqn_run_vector_non_vectorized():
    env = create_cartpole()

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration_coef": 0.1,
            "max_buffer_size": 0,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": False,
        }
    )

    model = dqn.DQN(0, env, config)
    dqn.train(0, model, env, config.n_env_steps)

    assert True


def test_no_error_dqn_run_vector_non_vectorized_shared_encoder():
    env = create_cartpole()

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration_coef": 0.1,
            "max_buffer_size": 0,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": True,
        }
    )

    model = dqn.DQN(0, env, config)
    dqn.train(0, model, env, config.n_env_steps)

    assert True


def test_no_error_dqn_run_visual_vectorized():
    N_ENVS = 16
    env = create_atari_envpool(N_ENVS)

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration_coef": 0.1,
            "max_buffer_size": 0,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": False,
        }
    )

    model = dqn.DQN(
        0, env, config, n_envs=N_ENVS, rearrange_pattern="b c h w -> b h w c"
    )
    dqn.train_vectorized(0, model, env, config.n_env_steps)

    assert True


def test_no_error_dqn_run_visual_vectorized_shared_encoder():
    N_ENVS = 16
    env = create_atari_envpool(N_ENVS)

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration_coef": 0.1,
            "max_buffer_size": 0,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": True,
        }
    )

    model = dqn.DQN(
        0, env, config, n_envs=N_ENVS, rearrange_pattern="b c h w -> b h w c"
    )
    dqn.train_vectorized(0, model, env, config.n_env_steps)

    assert True


def test_no_error_dqn_run_vector_vectorized():
    N_ENVS = 16
    env = create_atari_envpool(N_ENVS)

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration_coef": 0.1,
            "max_buffer_size": 0,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": False,
        }
    )

    model = dqn.DQN(
        0, env, config, n_envs=N_ENVS, rearrange_pattern="b c h w -> b h w c"
    )
    dqn.train_vectorized(0, model, env, config.n_env_steps)

    assert True


def test_no_error_dqn_run_vector_vectorized_shared_encoder():
    N_ENVS = 16
    env = create_cartpole_envpool(N_ENVS)

    config = ml_collections.ConfigDict(
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "exploration_coef": 0.1,
            "max_buffer_size": 0,
            "batch_size": 64,
            "skip_steps": 4,
            "n_env_steps": 1000,
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "shared_encoder": True,
        }
    )

    model = dqn.DQN(
        0, env, config, n_envs=N_ENVS, rearrange_pattern="b c h w -> b h w c"
    )
    dqn.train_vectorized(0, model, env, config.n_env_steps)

    assert True
