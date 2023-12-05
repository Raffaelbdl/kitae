import ml_collections

from rl.populations import pop_ppo
from tests.integration.env_fn import (
    create_atari,
    create_cartpole,
    create_atari_envpool,
    create_cartpole_envpool,
)

# model did create, run 1000 steps and finish training without error


def test_no_error_ppo_run_visual_non_vectorized():
    POP_SIZE = 5
    envs = [create_atari() for _ in range(POP_SIZE)]

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
            "action_space": envs[0].action_space,
            "observation_space": envs[0].observation_space,
            "shared_encoder": False,
            "jsd_coef": 0.05,
            "save_frequency": -1,
        }
    )

    model = pop_ppo.PopulationPPO(0, POP_SIZE, config)
    model.train(envs, config.n_env_steps, [])

    assert True


def test_no_error_ppo_run_visual_non_vectorized_shared_encoder():
    POP_SIZE = 5
    envs = [create_atari() for _ in range(POP_SIZE)]

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
            "action_space": envs[0].action_space,
            "observation_space": envs[0].observation_space,
            "shared_encoder": True,
            "jsd_coef": 0.05,
            "save_frequency": -1,
        }
    )

    model = pop_ppo.PopulationPPO(0, POP_SIZE, config)
    model.train(envs, config.n_env_steps, [])

    assert True


def test_no_error_ppo_run_vector_non_vectorized():
    POP_SIZE = 5
    envs = [create_cartpole() for _ in range(POP_SIZE)]

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
            "action_space": envs[0].action_space,
            "observation_space": envs[0].observation_space,
            "shared_encoder": False,
            "jsd_coef": 0.05,
            "save_frequency": -1,
        }
    )

    model = pop_ppo.PopulationPPO(0, POP_SIZE, config)
    model.train(envs, config.n_env_steps, [])

    assert True


def test_no_error_ppo_run_vector_non_vectorized_shared_encoder():
    POP_SIZE = 5
    envs = [create_cartpole() for _ in range(POP_SIZE)]

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
            "action_space": envs[0].action_space,
            "observation_space": envs[0].observation_space,
            "shared_encoder": True,
            "jsd_coef": 0.05,
            "save_frequency": -1,
        }
    )

    model = pop_ppo.PopulationPPO(0, POP_SIZE, config)
    model.train(envs, config.n_env_steps, [])

    assert True


def test_no_error_ppo_run_visual_vectorized():
    POP_SIZE = 5
    N_ENVS = 4
    envs = [create_atari_envpool(N_ENVS) for _ in range(POP_SIZE)]

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
            "action_space": envs[0].action_space,
            "observation_space": envs[0].observation_space,
            "shared_encoder": False,
            "jsd_coef": 0.05,
            "save_frequency": -1,
        }
    )

    model = pop_ppo.PopulationPPO(
        0, POP_SIZE, config, rearrange_pattern="b c h w -> b h w c", n_envs=N_ENVS
    )
    model.train(envs, config.n_env_steps, [])

    assert True


def test_no_error_ppo_run_visual_vectorized_shared_encoder():
    POP_SIZE = 5
    N_ENVS = 4
    envs = [create_atari_envpool(N_ENVS) for _ in range(POP_SIZE)]

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
            "action_space": envs[0].action_space,
            "observation_space": envs[0].observation_space,
            "shared_encoder": True,
            "jsd_coef": 0.05,
            "save_frequency": -1,
        }
    )

    model = pop_ppo.PopulationPPO(
        0, POP_SIZE, config, rearrange_pattern="b c h w -> b h w c", n_envs=N_ENVS
    )
    model.train(envs, config.n_env_steps, [])

    assert True


def test_no_error_ppo_run_vector_vectorized():
    POP_SIZE = 5
    N_ENVS = 4
    envs = [create_cartpole_envpool(N_ENVS) for _ in range(POP_SIZE)]

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
            "action_space": envs[0].action_space,
            "observation_space": envs[0].observation_space,
            "shared_encoder": False,
            "jsd_coef": 0.05,
            "save_frequency": -1,
        }
    )

    model = pop_ppo.PopulationPPO(0, POP_SIZE, config, n_envs=N_ENVS)
    model.train(envs, config.n_env_steps, [])

    assert True


def test_no_error_ppo_run_vector_vectorized_shared_encoder():
    POP_SIZE = 5
    N_ENVS = 4
    envs = [create_cartpole_envpool(N_ENVS) for _ in range(POP_SIZE)]

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
            "action_space": envs[0].action_space,
            "observation_space": envs[0].observation_space,
            "shared_encoder": True,
            "jsd_coef": 0.05,
            "save_frequency": -1,
        }
    )

    model = pop_ppo.PopulationPPO(0, POP_SIZE, config, n_envs=N_ENVS)
    model.train(envs, config.n_env_steps, [])

    assert True
