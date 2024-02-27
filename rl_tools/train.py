import jax
import numpy as np

from rl_tools.base import Base, EnvType, EnvProcs, AlgoType
from rl_tools.buffer import OnPolicyBuffer, Experience, OffPolicyBuffer
from rl_tools.save import Saver, SaverContext

from rl_tools.types import EnvLike

from rl_tools.callbacks import callback
from rl_tools.callbacks.callback import Callback, CallbackData
from rl_tools.callbacks.episode_return_callback import EpisodeReturnCallback
from rl_tools.logging import Logger


def process_action(action: jax.Array, env_type: EnvType, env_procs: EnvProcs):
    def single_one_process(action: jax.Array):
        # return int(action)
        return action

    def single_many_process(action: jax.Array):
        return np.array(action)

    def parallel_one_process(action: dict[str, jax.Array]):
        return {agent: int(a) for agent, a in action.items()}

    def parallel_many_process(action: dict[str, jax.Array]):
        return action

    if env_type == EnvType.SINGLE and env_procs == EnvProcs.ONE:
        return single_one_process(action)
    elif env_type == EnvType.SINGLE and env_procs == EnvProcs.MANY:
        return single_many_process(action)
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.ONE:
        return parallel_one_process(action)
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.MANY:
        return parallel_many_process(action)
    else:
        raise NotImplementedError


def process_reward(reward, env_type: EnvType, env_procs: EnvProcs):
    def single_one_process(reward: float):
        return reward

    def single_many_process(reward: jax.Array):
        return reward

    def parallel_one_process(reward: dict[str, float]):
        return sum(reward.values())

    def parallel_many_process(reward: dict[str, jax.Array]):
        return np.sum(np.array(list(reward.values())), axis=0)

    if env_type == EnvType.SINGLE and env_procs == EnvProcs.ONE:
        return single_one_process(reward)
    elif env_type == EnvType.SINGLE and env_procs == EnvProcs.MANY:
        return single_many_process(reward)
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.ONE:
        return parallel_one_process(reward)
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.MANY:
        return parallel_many_process(reward)
    else:
        raise NotImplementedError


def process_termination(
    step: int,
    env: EnvLike,
    done,
    trunc,
    logs: dict,
    env_type: EnvType,
    env_procs: EnvProcs,
    callbacks: list[Callback],
):
    def single_one_process(env, done, trunc, logs):
        if done or trunc:
            print(step, " > ", logs["episode_return"])
            callback.on_episode_end(
                callbacks, CallbackData.on_episode_end(0, logs["episode_return"])
            )
            logs["episode_return"] = 0.0
            next_observation, info = env.reset()
            return next_observation, info
        return None, None

    def single_many_process(env, done, trunc, logs):
        for i, (d, t) in enumerate(zip(done, trunc)):
            if d or t:
                callback.on_episode_end(
                    callbacks, CallbackData.on_episode_end(0, logs["episode_return"][i])
                )
                if i == 0:
                    print(step, " > ", logs["episode_return"][i])
                logs["episode_return"][i] = 0.0
        return None, None

    def parallel_one_process(env, done, trunc, logs):
        if any(done.values()) or any(trunc.values()):
            print(step, " > ", logs["episode_return"])
            callback.on_episode_end(
                callbacks, CallbackData.on_episode_end(0, logs["episode_return"])
            )
            logs["episode_return"] = 0.0
            next_observation, info = env.reset()
            return next_observation, info
        return None, None

    def parallel_many_process(env, done, trunc, logs):
        check_d, check_t = np.stack(list(done.values()), axis=1), np.stack(
            list(trunc.values()), axis=1
        )
        for i, (d, t) in enumerate(zip(check_d, check_t)):
            if np.any(d) or np.any(t):
                callback.on_episode_end(
                    callbacks, CallbackData.on_episode_end(0, logs["episode_return"][i])
                )
                if i == 0:
                    print(step, " > ", logs["episode_return"][i])
                logs["episode_return"][i] = 0.0
        return None, None

    if env_type == EnvType.SINGLE and env_procs == EnvProcs.ONE:
        return single_one_process(env, done, trunc, logs)
    elif env_type == EnvType.SINGLE and env_procs == EnvProcs.MANY:
        return single_many_process(env, done, trunc, logs)
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.ONE:
        return parallel_one_process(env, done, trunc, logs)
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.MANY:
        return parallel_many_process(env, done, trunc, logs)
    else:
        raise NotImplementedError


def train(
    seed: int,
    base: Base,
    env: EnvLike,
    n_env_steps: int,
    env_type: EnvType,
    env_procs: EnvProcs,
    algo_type: AlgoType,
    *,
    start_step: int = 1,
    saver: Saver = None,
    callbacks: list[Callback] = None,
):
    callbacks = callbacks if callbacks else []
    callbacks = [EpisodeReturnCallback(population_size=1)] + callbacks
    callback.on_train_start(callbacks, CallbackData())

    if algo_type == AlgoType.ON_POLICY:
        buffer = OnPolicyBuffer(seed, base.config.update_cfg.max_buffer_size)
    else:
        buffer = OffPolicyBuffer(seed, base.config.update_cfg.max_buffer_size)

    observation, info = env.reset(seed=seed + 1)

    logger = Logger(callbacks, env_type=env_type, env_procs=env_procs)
    logger.init_logs(observation)

    with SaverContext(saver, base.config.train_cfg.save_frequency) as s:
        for step in range(start_step, n_env_steps + 1):
            logger["step"] = step

            action, log_prob = base.explore(observation)

            next_observation, reward, done, trunc, info = env.step(
                process_action(action, env_type, env_procs)
            )
            logger["episode_return"] += process_reward(reward, env_type, env_procs)

            termination = process_termination(
                step * base.config.env_cfg.n_envs,
                env,
                done,
                trunc,
                logger,
                env_type,
                env_procs,
                callbacks,
            )
            if termination[0] is not None and termination[1] is not None:
                next_observation, info = termination

            buffer.add(
                Experience(
                    observation=observation,
                    action=action,
                    reward=reward,
                    done=done,
                    next_observation=next_observation,
                    log_prob=log_prob,
                )
            )

            if base.should_update(step, buffer):
                callback.on_update_start(callbacks, CallbackData())
                logger.update(base.update(buffer))
                callback.on_update_end(callbacks, CallbackData(logs=logger.get_logs()))

            s.update(step, base.state_dict)

            observation = next_observation

    env.close()
    callback.on_train_end(callbacks, CallbackData())
