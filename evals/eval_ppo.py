import envpool
import ml_collections

from rl.algos import ppo
from rl.wrapper import EnvpoolCompatibility

from evals.eval_callbacks import TimeCallback


N_ENVS = 16
N_ENV_STEPS = 5 * 10**5
CONFIG = ml_collections.ConfigDict(
    {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "_lambda": 0.95,
        "normalize": True,
        "max_buffer_size": 128,
        "batch_size": 256,
        "num_epochs": 3,
        "learning_rate_annealing": True,
        "max_grad_norm": 0.5,
        "n_env_steps": N_ENV_STEPS // N_ENVS,
        "shared_encoder": True,
        "save_frequency": -1,
    }
)


def measure_ppo_cnn_envpool_time():
    task_id = "Pong-v5"
    env = EnvpoolCompatibility(
        envpool.make(task_id, env_type="gymnasium", num_envs=N_ENVS)
    )
    CONFIG["action_space"] = env.action_space
    CONFIG["observation_space"] = env.observation_space

    model = ppo.PPO(0, CONFIG, rearrange_pattern="b c h w -> b h w c", n_envs=N_ENVS)

    time_callback = TimeCallback(task_id, n_envs=N_ENVS)
    model.train(env, CONFIG.n_env_steps, callbacks=[time_callback])
    time_callback.write_results("ppo_cnn_envpool")


if __name__ == "__main__":
    measure_ppo_cnn_envpool_time()
