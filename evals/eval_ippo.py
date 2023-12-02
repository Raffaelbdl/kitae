import pettingzoo
from pettingzoo.butterfly import cooperative_pong_v5
import ml_collections

from rl.algos import ippo
from rl.wrapper import SubProcVecParallelEnvCompatibility
from vec_parallel_env import SubProcVecParallelEnv
import numpy as np
from rl.callbacks.wandb_callback import WandbCallback
from evals.eval_callbacks import EvalCallback, TimeCallback, ScoreCallback


def create_cooperative_pong(render: bool = False) -> pettingzoo.ParallelEnv:
    from pettingzoo.butterfly import cooperative_pong_v5
    import supersuit as ss

    env = cooperative_pong_v5.parallel_env(
        render_mode="human" if render else "rgb_array"
    )
    env = ss.action_lambda_v1(
        env, lambda x, y: int(x) if x is not None else 0, lambda x: x
    )
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.frame_stack_v2(env, stack_size=4)
    env = ss.resize_v1(env, 84, 84)
    env = ss.observation_lambda_v0(
        env,
        lambda x, y: np.nan_to_num(x, posinf=0, neginf=0).astype(np.uint8),
    )
    env.reset()
    return env


def eval_ppo_cnn():
    TASK_ID = "CooperativePong"
    N_ENVS = 1
    N_ENV_STEPS = 10**3
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

    env = create_cooperative_pong(False)
    CONFIG["action_space"] = env.action_space(env.agents[0])
    CONFIG["observation_space"] = env.observation_space(env.agents[0])

    model = ippo.PPO(
        0, CONFIG, rearrange_pattern="b h w c -> b h w c", n_envs=N_ENVS, tabulate=True
    )

    callbacks: list[EvalCallback] = [
        TimeCallback(TASK_ID, n_envs=N_ENVS),
        ScoreCallback(TASK_ID, n_envs=N_ENVS, maxlen=20),
    ]

    model.train(env, CONFIG.n_env_steps, callbacks=callbacks)

    for c in callbacks:
        c.write_results("ippo_cnn")


def eval_ppo_cnn_subprocvenv():
    TASK_ID = "CooperativePong"
    N_ENVS = 24
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
            "num_epochs": 1,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": N_ENV_STEPS // N_ENVS,
            "shared_encoder": True,
            "save_frequency": -1,
        }
    )

    env = create_cooperative_pong(False)
    CONFIG["action_space"] = env.action_space(env.agents[0])
    CONFIG["observation_space"] = env.observation_space(env.agents[0])

    env = SubProcVecParallelEnv([lambda: env for _ in range(N_ENVS)])
    env = SubProcVecParallelEnvCompatibility(env)

    model = ippo.PPO(
        0, CONFIG, rearrange_pattern="b h w c -> b h w c", n_envs=N_ENVS, tabulate=True
    )

    callbacks: list[EvalCallback] = [
        WandbCallback("cooperative_pong", "raffael", CONFIG),
        TimeCallback(TASK_ID, n_envs=N_ENVS),
        ScoreCallback(TASK_ID, n_envs=N_ENVS, maxlen=20),
    ]

    model.train(env, CONFIG.n_env_steps, callbacks=callbacks)

    for c in callbacks:
        c.write_results("ippo_cnn_subprocvenv")
