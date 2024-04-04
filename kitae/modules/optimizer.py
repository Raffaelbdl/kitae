import optax


def linear_learning_rate_schedule(
    init_learning_rate: float,
    end_learning_rate: float,
    *,
    n_envs: int,
    n_env_steps: int,
    max_buffer_size: int,
    batch_size: int,
    num_epochs: int,
) -> optax.Schedule:
    total_buffer_size = n_envs * max_buffer_size
    total_env_steps = n_envs * n_env_steps
    num_batches = total_buffer_size // batch_size
    n_updates = total_env_steps // max_buffer_size * num_epochs * num_batches
    return optax.linear_schedule(init_learning_rate, end_learning_rate, n_updates, 0)
