import time
from typing import Callable

from absl import logging
import numpy as np
from tensorboardX import SummaryWriter

from rl_tools.interface import IAgent, AlgoType
from rl_tools.buffer import Experience, buffer_factory
from rl_tools.save import Saver, SaverContext

from rl_tools.envs.make import wrap_single_env, wrap_single_parallel_env
from rl_tools.types import EnvLike

from rl_tools.callbacks import callback
from rl_tools.callbacks.callback import Callback, CallbackData

logging.set_verbosity(logging.INFO)


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
    writer: SummaryWriter | None,
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
                if writer:
                    writer.add_scalar(
                        "Charts/episodic_return", episodic_return, global_step
                    )
                    writer.add_scalar(
                        "Charts/episodic_length", episodic_length, global_step
                    )

            real_next_observations[idx] = infos["final_observation"][idx]
        return real_next_observations

    return next_observations


def make_process_inputs(agent: IAgent, observations: np.ndarray, dones: np.ndarray):
    recurrent = False
    try:
        agent.explore(observations, dones)
        recurrent = True
        agent.lstm_state = None
    except TypeError:
        pass

    def process_inputs(
        observations: np.ndarray, dones: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        if recurrent:
            return (observations, dones)
        return observations

    return process_inputs


def vectorized_train(
    seed: int,
    agent: IAgent,
    env: EnvLike,
    n_env_steps: int,
    algo_type: AlgoType,
    *,
    start_step: int = 1,
    saver: Saver = None,
    callbacks: list[Callback] = None,
    writer: SummaryWriter | None = None,
):
    env = check_env(env)
    start_time = time.time()

    callbacks = callbacks if callbacks else []
    callback.on_train_start(callbacks, CallbackData())

    observations, infos = env.reset(seed=seed + 1)
    dones = np.zeros((len(observations)), np.bool_)
    process_inputs = make_process_inputs(agent, observations, dones)

    buffer = buffer_factory(seed, algo_type, agent.config.update_cfg.max_buffer_size)

    with SaverContext(saver, agent.config.train_cfg.save_frequency) as s:
        for step in range(start_step, n_env_steps + 1):
            global_step = step * agent.config.env_cfg.n_envs

            inputs = process_inputs(observations, dones)
            actions, log_probs = agent.explore(*inputs)

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
                update_dict = agent.update(buffer)
                if writer:
                    for key, value in update_dict.items():
                        writer.add_scalar(f"losses/{key}", value, global_step)
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                callback.on_update_end(callbacks, CallbackData(logs=update_dict))

            s.update(step, agent.state_dict)

            observations = next_observations

    env.close()
    writer.close()
    callback.on_train_end(callbacks, CallbackData())
