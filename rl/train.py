import jax
import numpy as np

from rl.base import Base, EnvType, EnvProcs, AlgoType
from rl.buffer import OnPolicyBuffer, OnPolicyExp, OffPolicyBuffer
from rl.save import Saver, SaverContext

from rl.types import EnvLike

from rl.callbacks import callback
from rl.callbacks.callback import Callback, CallbackData
from rl.callbacks.episode_return_callback import EpisodeReturnCallback
from rl.logging import Logger


def process_action(action: jax.Array, env_type: EnvType, env_procs: EnvProcs):
    def single_one_process(action: jax.Array):
        return int(action)

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
        buffer = OnPolicyBuffer(seed, base.config.max_buffer_size)
    else:
        buffer = OffPolicyBuffer(seed, base.config.max_buffer_size)

    observation, info = env.reset(seed=seed + 1)

    logger = Logger(callbacks, env_type=env_type, env_procs=env_procs)
    logger.init_logs(observation)

    with SaverContext(saver, base.config.save_frequency) as s:
        for step in range(start_step, n_env_steps + 1):
            logger["step"] = step

            action, log_prob = base.explore(observation)

            next_observation, reward, done, trunc, info = env.step(
                process_action(action, env_type, env_procs)
            )
            logger["episode_return"] += process_reward(reward, env_type, env_procs)

            termination = process_termination(
                step * base.config.env_config.n_envs,
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
                OnPolicyExp(
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

            s.update(step, base.state)

            observation = next_observation

    env.close()
    callback.on_train_end(callbacks, CallbackData())


def process_termination_population(
    step: int,
    env: EnvLike,
    done,
    trunc,
    logs: dict,
    env_type: EnvType,
    env_procs: EnvProcs,
    agent_id: int,
    callbacks: list[Callback],
):
    def single_one_process(env, done, trunc, logs):
        if done or trunc:
            print(step, " > ", logs["episode_return"][agent_id])
            callback.on_episode_end(
                callbacks,
                CallbackData.on_episode_end(agent_id, logs["episode_return"][agent_id]),
            )
            logs["episode_return"][agent_id] = 0.0
            next_observation, info = env.reset()
            return next_observation, info
        return None, None

    def single_many_process(env, done, trunc, logs):
        for i, (d, t) in enumerate(zip(done, trunc)):
            if d or t:
                callback.on_episode_end(
                    callbacks,
                    CallbackData.on_episode_end(
                        agent_id, logs["episode_return"][agent_id][i]
                    ),
                )

                if i == 0:
                    print(step, " > ", logs["episode_return"][agent_id][i])
                logs["episode_return"][agent_id][i] = 0.0
        return None, None

    def parallel_one_process(env, done, trunc, logs):
        if any(done.values()) or any(trunc.values()):
            print(step, " > ", logs["episode_return"][agent_id])
            callback.on_episode_end(
                callbacks,
                CallbackData.on_episode_end(agent_id, logs["epiode_return"][agent_id]),
            )
            logs["episode_return"][agent_id] = 0.0
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
                    callbacks,
                    CallbackData.on_episode_end(
                        agent_id, logs["episode_return"][agent_id][i]
                    ),
                )
                if i == 0:
                    print(step, " > ", logs["episode_return"][agent_id][i])
                logs["episode_return"][agent_id][i] = 0.0
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


def train_population(
    seed: int,
    base: Base,
    envs: list[EnvLike],
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
    callbacks = [EpisodeReturnCallback(population_size=len(envs))] + callbacks
    callback.on_train_start(callbacks, CallbackData())

    if algo_type == AlgoType.ON_POLICY:
        buffers = [
            OnPolicyBuffer(seed, base.config.max_buffer_size) for i in range(len(envs))
        ]
    else:
        buffers = [
            OffPolicyBuffer(seed, base.config.max_buffer_size) for i in range(len(envs))
        ]

    observations, infos = zip(*[envs[i].reset(seed=seed + i) for i in range(len(envs))])

    logger = Logger(callbacks, env_type=env_type, env_procs=env_procs)
    logger.init_logs(observations)

    with SaverContext(saver, base.config.save_frequency) as s:
        for step in range(start_step, n_env_steps + 1):
            logger["step"] = step

            actions, log_probs = base.explore(observations)

            next_observations = []
            for i, env in enumerate(envs):
                next_observation, reward, done, trunc, info = env.step(
                    process_action(actions[i], env_type, env_procs)
                )
                logger["episode_return"][i] += process_reward(
                    reward, env_type, env_procs
                )

                termination = process_termination_population(
                    step * base.config.env_config.n_envs,
                    env,
                    done,
                    trunc,
                    logger,
                    env_type,
                    env_procs,
                    i,
                    callbacks,
                )
                if termination[0] is not None and termination[1] is not None:
                    next_observation, info = termination

                next_observations.append(next_observation)

                buffers[i].add(
                    OnPolicyExp(
                        observation=observations[i],
                        action=actions[i],
                        reward=reward,
                        done=done,
                        next_observation=next_observation,
                        log_prob=log_probs[i],
                    )
                )

            if base.should_update(step, buffers[0]):
                callback.on_update_start(callbacks, CallbackData())
                logger.update(base.update(buffers))
                callback.on_update_end(callbacks, CallbackData(logs=logger.get_logs()))

            s.update(step, base.state)

            observations = next_observations

    env.close()
    callback.on_train_end(callbacks, CallbackData())
