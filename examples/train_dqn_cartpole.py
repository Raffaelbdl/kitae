from evals.eval_envs import make_cartpole_vector

from rl.algos import dqn
from rl import config as cfg


def main():
    env, env_config = make_cartpole_vector(10)

    config = cfg.AlgoConfig.create_config_from_file(
        "./configs/dqn_cartpole.yaml", env_config
    )

    algo = dqn.DQN(config)

    algo.train(env, algo.config.train_cfg.n_env_steps, callbacks=[])


if __name__ == "__main__":
    main()
