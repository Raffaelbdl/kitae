import time
import numpy as np
from tensorboardX import SummaryWriter

from kitae.interface import IAgent, AlgoType
from kitae.buffer import Experience, buffer_factory
from kitae.saving import SaverContext

from kitae.envs.make import wrap_single_env, wrap_single_parallel_env
from kitae.types import EnvLike

from save.checkpoint import Checkpointer


def check_env(env: EnvLike) -> EnvLike:
    """Checks if environment can be used for training.

    Args:
        env (EnvLike): An environment.

    Returns:
        The original environment or a compatible one.
    """

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
) -> np.ndarray:
    """Processes the termination part of the loop.

    Args:
        global_step (int): The current environment step.
        next_observations (Array): The observations resulting from the last actions.
        infos (dict): The info resulting from the last actions.
        writer (SummaryWriter): A writer to log metrics.

    Returns:
        The next observations to store in the buffer.
    """
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
                print(
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


def vectorized_train(
    seed: int,
    agent: IAgent,
    env: EnvLike,
    n_env_steps: int,
    algo_type: AlgoType,
    *,
    start_step: int = 1,
    checkpointer: Checkpointer = None,
    writer: SummaryWriter = None,
) -> None:
    """Trains an agent in a vectorized environment.

    Important:
        `env.close()` will be called at the end of the training.

    Args:
        seed (int): An int for reproducibility.
        agent (IAgent): An agent to train.
        env (EnvLike): An environment to train in.
        n_env_steps (int): The number of steps to do in the environment.
        algo_type (AlgoType): The type of algorithm.
        start_step (int): The starting step in the environment.
        saver (Saver): A saver instance.
    """
    env = check_env(env)
    start_time = time.time()

    observations, infos = env.reset(seed=seed + 1)

    buffer = buffer_factory(seed, algo_type, agent.config.update_cfg.max_buffer_size)

    with SaverContext(checkpointer, agent.config.train_cfg.save_frequency) as s:
        for step in range(start_step, n_env_steps + 1):
            global_step = step * agent.config.env_cfg.n_envs
            actions, log_probs = agent.explore(observations)
            (
                next_observations,
                rewards,
                terminateds,
                truncations,
                infos,
            ) = env.step(np.array(actions))

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
                update_dict = agent.update(buffer)
                if writer:
                    for key, value in update_dict.items():
                        writer.add_scalar(f"losses/{key}", value, global_step)
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

            s.update(step, agent.state)

            observations = next_observations

    env.close()
    writer.close()
