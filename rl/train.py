from envpool.python.api import EnvPool
import gymnasium as gym
import jax
import numpy as np
import pettingzoo
from vec_parallel_env import SubProcVecParallelEnv
import wandb

from rl.base import Base, EnvType, EnvProcs
from rl.buffer import OnPolicyBuffer, OnPolicyExp, FBXOnPolicyBuffer
from rl import callback
from rl.callback import Callback, CallbackData

EnvLike = gym.Env | EnvPool | pettingzoo.ParallelEnv | SubProcVecParallelEnv

from rl.save import Saver, SaverContext


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


def init_episode_return(
    first_observation: jax.Array, env_type: EnvType, env_procs: EnvProcs
):
    if env_type == EnvType.SINGLE and env_procs == EnvProcs.ONE:
        return 0.0
    elif env_type == EnvType.SINGLE and env_procs == EnvProcs.MANY:
        return np.zeros((first_observation.shape[0],))
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.ONE:
        return 0.0
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.MANY:
        return np.zeros((np.array(list(first_observation.values())[0]).shape[0],))
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
            print(step, " > ", logs["episode_return"], " | ", logs["kl_divergence"])
            callback.on_episode_end(
                callbacks, CallbackData(episode_return=logs["episode_return"])
            )
            logs["episode_return"] = 0.0
            next_observation, info = env.reset()
            return next_observation, info
        return None, None

    def single_many_process(env, done, trunc, logs):
        for i, (d, t) in enumerate(zip(done, trunc)):
            if d or t:
                callback.on_episode_end(
                    callbacks, CallbackData(episode_return=logs["episode_return"][i])
                )
                if i == 0:
                    print(
                        step,
                        " > ",
                        logs["episode_return"][i],
                        " | ",
                        logs["kl_divergence"],
                    )
                logs["episode_return"][i] = 0.0
        return None, None

    def parallel_one_process(env, done, trunc, logs):
        if any(done.values()) or any(trunc.values()):
            print(step, " > ", logs["episode_return"], " | ", logs["kl_divergence"])
            callback.on_episode_end(
                callbacks, CallbackData(episode_return=logs["episode_return"])
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
                    callbacks, CallbackData(episode_return=logs["episode_return"][i])
                )
                if i == 0:
                    print(
                        step,
                        " > ",
                        logs["episode_return"][i],
                        " | ",
                        logs["kl_divergence"],
                    )
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
    *,
    start_step: int = 1,
    saver: Saver = None,
    use_wandb: bool = False,
    callbacks: list[Callback] = [],
):
    callback.on_train_start(callbacks, CallbackData())
    buffer = OnPolicyBuffer(seed, base.config.max_buffer_size)

    observation, info = env.reset(seed=seed + 1)

    logs = {
        "episode_return": init_episode_return(observation, env_type, env_procs),
        "kl_divergence": 0.0,
    }

    with SaverContext(saver, base.config.save_frequency) as s:
        for step in range(start_step, n_env_steps + 1):
            logs["step"] = step

            action, log_prob = base.explore(observation)
            next_observation, reward, done, trunc, info = env.step(
                process_action(action, env_type, env_procs)
            )
            logs["episode_return"] += process_reward(reward, env_type, env_procs)

            termination = process_termination(
                step * base.n_envs,
                env,
                done,
                trunc,
                logs,
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

            if len(buffer) >= base.config.max_buffer_size:
                callback.on_update_start(callbacks, CallbackData())
                logs |= base.update(buffer)
                callback.on_update_end(callbacks, CallbackData())

                if use_wandb:
                    wandb.log(logs)

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
):
    def single_one_process(env, done, trunc, logs):
        if done or trunc:
            print(
                step,
                " > ",
                logs["episode_return"][agent_id],
                " | ",
                logs["kl_divergence"],
            )
            logs["episode_return"][agent_id] = 0.0
            next_observation, info = env.reset()
            return next_observation, info
        return None, None

    def single_many_process(env, done, trunc, logs):
        for i, (d, t) in enumerate(zip(done, trunc)):
            if d or t:
                if i == 0:
                    print(
                        step,
                        " > ",
                        logs["episode_return"][agent_id][i],
                        " | ",
                        logs["kl_divergence"],
                    )
                logs["episode_return"][agent_id][i] = 0.0
        return None, None

    def parallel_one_process(env, done, trunc, logs):
        if any(done.values()) or any(trunc.values()):
            print(
                step,
                " > ",
                logs["episode_return"][agent_id],
                " | ",
                logs["kl_divergence"],
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
                if i == 0:
                    print(
                        step,
                        " > ",
                        logs["episode_return"][agent_id][i],
                        " | ",
                        logs["kl_divergence"],
                    )
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
    *,
    start_step: int = 1,
    saver: Saver = None,
    use_wandb: bool = False,
):
    buffers = [
        OnPolicyBuffer(seed + i, base.config.max_buffer_size) for i in range(len(envs))
    ]

    observations, infos = zip(*[envs[i].reset(seed=seed + i) for i in range(len(envs))])

    logs = {
        "episode_return": [
            init_episode_return(obs, env_type, env_procs) for obs in observations
        ],
        "kl_divergence": 0.0,
    }

    with SaverContext(saver, base.config.save_frequency) as s:
        for step in range(start_step, n_env_steps + 1):
            logs["step"] = step

            actions, log_probs = base.explore(observations)

            next_observations = []
            for i, env in enumerate(envs):
                next_observation, reward, done, trunc, info = env.step(
                    process_action(actions[i], env_type, env_procs)
                )
                logs["episode_return"][i] += process_reward(reward, env_type, env_procs)

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

                termination = process_termination_population(
                    step, env, done, trunc, logs, env_type, env_procs, i
                )
                if termination[0] is not None and termination[1] is not None:
                    next_observation, info = termination

                next_observations.append(next_observation)

            if len(buffers[0]) >= base.config.max_buffer_size:
                logs |= base.update(buffers)

            if use_wandb:
                wandb.log(logs)

            s.update(step, base.state)

            observations = next_observations

        env.close()
