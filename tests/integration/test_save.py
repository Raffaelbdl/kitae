import shutil

from evals.eval_envs import make_cartpole

from rl_tools.algos import dqn
import rl_tools.config as cfg


# should make this test algo-agnostic
def test_save_dqn():
    env, env_config = make_cartpole()
    config = cfg.AlgoConfig.create_config_from_file(
        "./configs/dqn_cartpole.yaml", env_config
    )
    config.n_env_steps = 1000
    config.train_cfg.save_frequency = 500
    algo = dqn.DQN(config, run_name="tmp_dqn")
    algo.train(env, algo.config.train_cfg.n_env_steps, callbacks=[])
    del algo

    new_algo = dqn.DQN.unserialize("./results/tmp_dqn")
    new_algo.restore()
    del new_algo

    shutil.move("./results/tmp_dqn", "./results/tmp_dqn2")
    new_algo = dqn.DQN.unserialize("./results/tmp_dqn2")
    new_algo.restore()
    del new_algo

    shutil.rmtree("./results/tmp_dqn2")
    assert True
