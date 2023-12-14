from evals.eval_envs import make_pong_vector

from rl.algos import ppo
from rl import config as cfg
from rl.transformation import normalize_frames


def main():
    env, env_config = make_pong_vector(32)

    config = cfg.AlgoConfig.create_config_from_file(
        "./configs/ppo_pong.yaml", env_config
    )

    algo = ppo.PPO(
        config,
        rearrange_pattern="b c h w -> b h w c",
        preprocess_fn=normalize_frames,
    )

    algo.train(
        env,
        algo.config.train_cfg.n_env_steps,
        callbacks=[],
    )


if __name__ == "__main__":
    main()
