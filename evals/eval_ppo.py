import envpool
import ml_collections

from rl.algos import ppo
from rl.wrapper import EnvpoolCompatibility
from rl.callbacks.callback import Callback
from rl.callbacks.wandb_callback import WandbCallback

from evals.eval_callbacks import EvalCallback, TimeCallback, ScoreCallback


def eval_ppo_cnn_envpool():
    TASK_ID = "Breakout-v5"
    N_ENVS = 32
    N_ENV_STEPS = 5 * 10**7
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
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": N_ENV_STEPS // N_ENVS,
            "shared_encoder": True,
            "save_frequency": -1,
        }
    )

    env = EnvpoolCompatibility(
        envpool.make(TASK_ID, env_type="gymnasium", num_envs=N_ENVS)
    )
    CONFIG["action_space"] = env.action_space
    CONFIG["observation_space"] = env.observation_space

    model = ppo.PPO(
        0, CONFIG, rearrange_pattern="b c h w -> b h w c", n_envs=N_ENVS, tabulate=True
    )

    callbacks: list[Callback] = [
        TimeCallback(TASK_ID, n_envs=N_ENVS),
        ScoreCallback(TASK_ID, n_envs=N_ENVS, maxlen=20),
        WandbCallback("cooperative_pong", entity="raffael", config=CONFIG),
    ]

    model.train(env, CONFIG.n_env_steps, callbacks=callbacks)

    for c in callbacks[:2]:
        c.write_results("ppo_cnn_envpool")


def eval_ppo_vector_envpool():
    TASK_ID = "CartPole-v1"
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
            "shared_encoder": False,
            "save_frequency": -1,
        }
    )

    env = EnvpoolCompatibility(
        envpool.make(TASK_ID, env_type="gymnasium", num_envs=N_ENVS)
    )
    CONFIG["action_space"] = env.action_space
    CONFIG["observation_space"] = env.observation_space

    model = ppo.PPO(0, CONFIG, n_envs=N_ENVS)

    callbacks: list[EvalCallback] = [
        TimeCallback(TASK_ID, n_envs=N_ENVS),
        ScoreCallback(TASK_ID, n_envs=N_ENVS, maxlen=20),
    ]

    model.train(env, CONFIG.n_env_steps, callbacks=callbacks)

    for c in callbacks:
        c.write_results("ppo_vec_envpool")
