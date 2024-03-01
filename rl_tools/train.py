from absl import logging
import gymnasium as gym
import jax
import numpy as np

from rl_tools.interface import IAgent, AlgoType
from rl_tools.buffer import Experience, buffer_factory
from rl_tools.save import Saver, SaverContext

from rl_tools.envs.make import wrap_single_env
from rl_tools.types import EnvLike

from rl_tools.callbacks import callback
from rl_tools.callbacks.callback import Callback, CallbackData
from rl_tools.callbacks.episode_return_callback import EpisodeReturnCallback
from rl_tools.logging import Logger
from rl_tools.envs.wrappers.record_episode_statistics import (
    ParallelRecordEpisodeStatistics,
)

from tensorboardX import SummaryWriter

from rl_tools.envs.make import wrap_single_env, wrap_single_parallel_env


def check_env(env: EnvLike) -> EnvLike:
    def not_vec_fallback(env: EnvLike) -> EnvLike:
        if hasattr(env, "agents"):
            return wrap_single_parallel_env(env)
        return wrap_single_env(env)

    try:
        env.get_wrapper_attr("num_envs")
        env.get_wrapper_attr("is_vector_env")
        return env
    except AttributeError:
        return not_vec_fallback(env)


def process_termination(
    global_step: int,
    next_observations: np.ndarray,
    infos: dict,
    writer: SummaryWriter,
):
    if "final_info" in infos:
        real_next_observations = next_observations.copy()

        logged = False
        for idx, final in enumerate(infos["_final_info"]):
            if not final:
                continue

            episodic_return = infos["episode"]["r"][idx]
            episodic_length = infos["episode"]["l"][idx]

            if not logged:
                logged = True
                logging.info(
                    f"Global Step = {global_step} | Episodic Return = {episodic_return:.3f}"
                )
                # writer.add_scalar(
                #     "Charts/episodic_return", episodic_return, global_step
                # )
                # writer.add_scalar(
                #     "Charts/episodic_length", episodic_length, global_step
                # )

            real_next_observations[idx] = infos["final_observation"][idx]
        return real_next_observations

    return next_observations


def vectorized_train(
    seed: int,
    agent: IAgent,
    env: EnvLike,
    n_env_steps: int,
    parallel: bool,
    algo_type: AlgoType,
    *,
    start_step: int = 1,
    saver: Saver = None,
    callbacks: list[Callback] = None,
    writer: SummaryWriter = None,
):
    env = check_env(env)

    callbacks = callbacks if callbacks else []
    callbacks = [EpisodeReturnCallback(population_size=1)] + callbacks
    callback.on_train_start(callbacks, CallbackData())

    observations, infos = env.reset(seed=seed + 1)

    logger = Logger(callbacks, parallel=parallel, vectorized=True)
    logger.init_logs(observations)

    buffer = buffer_factory(seed, algo_type, agent.config.update_cfg.max_buffer_size)

    with SaverContext(saver, agent.config.train_cfg.save_frequency) as s:
        for step in range(start_step, n_env_steps + 1):
            global_step = step * agent.config.env_cfg.n_envs
            logger["step"] = step
            actions, log_probs = agent.explore(observations)
            (
                next_observations,
                rewards,
                terminateds,
                truncations,
                infos,
            ) = env.step(actions)

            real_next_observations = process_termination(
                global_step, next_observations, infos, writer
            )

            dones = np.logical_or(terminateds, truncations)

            buffer.add(
                Experience(
                    observation=observations,
                    action=actions,
                    reward=rewards,
                    done=dones,
                    next_observation=real_next_observations,
                    log_prob=log_probs,
                )
            )

            if agent.should_update(step, buffer):
                callback.on_update_start(callbacks, CallbackData())
                logger.update(agent.update(buffer))
                callback.on_update_end(callbacks, CallbackData(logs=logger.get_logs()))

            s.update(step, agent.state_dict)

            observations = next_observations

    env.close()
    callback.on_train_end(callbacks, CallbackData())
