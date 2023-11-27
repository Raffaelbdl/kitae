from envpool.python.api import EnvPool
import gymnasium as gym
import jax
import numpy as np
import pettingzoo
from vec_parallel_env import SubProcVecParallelEnv
import wandb

from rl.base import Base, EnvType, EnvProcs
from rl.buffer import OnPolicyBuffer, OnPolicyExp

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
):
    def single_one_process(env, done, trunc, logs):
        if done or trunc:
            print(step, " > ", logs["episode_return"], " | ", logs["kl_divergence"])
            logs["episode_return"] = 0.0
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
                        logs["episode_return"][i],
                        " | ",
                        logs["kl_divergence"],
                    )
                logs["episode_return"][i] = 0.0
        return None, None

    def parallel_one_process(env, done, trunc, logs):
        if any(done.values()) or any(trunc.values()):
            print(step, " > ", logs["episode_return"], " | ", logs["kl_divergence"])
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
):
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

            termination = process_termination(
                step, env, done, trunc, logs, env_type, env_procs
            )
            if termination[0] is not None and termination[1] is not None:
                next_observation, info = termination

            if len(buffer) >= base.config.max_buffer_size:
                logs |= base.update(buffer)

                if use_wandb:
                    wandb.log(logs)

            s.update(step, base.state)

            observation = next_observation

    env.close()
