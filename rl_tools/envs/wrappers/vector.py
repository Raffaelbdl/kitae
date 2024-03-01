from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Any, Callable

import numpy as np
from pettingzoo import ParallelEnv


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


class CloudPickleWrapper:
    """Uses cloudpickle to serialize contents"""

    def __init__(self, x) -> None:
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class VecParallelEnv(ABC):
    """An abstract asynchronous, vectorized environment"""

    def __init__(self, num_envs: int, observation_spaces, action_spaces) -> None:
        self.num_envs = num_envs
        self.is_vector_env = True
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets an attribute from the wrapper and lower environments if `name` doesn't exist in this object.

        Args:
            name: The variable name to get

        Returns:
            The variable with name in wrapper or lower environments
        """
        if name in self.__dir__():  # todo change in v1.0.0 to `hasattr`
            return getattr(self, name)
        else:
            try:
                return self.env.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {self.class_name()} has no attribute {name!r}"
                ) from e


def worker(child_conn: Connection, parent_conn: Connection, env_fn_wrapper):
    parent_conn.close()
    env: ParallelEnv = env_fn_wrapper.x()  # using cloudpickle

    while True:
        cmd, data = child_conn.recv()

        if cmd == "close":
            env.close()
            child_conn.close()
            break

        elif cmd == "get_agents":
            child_conn.send(env.agents)
        elif cmd == "get_num_agents":
            child_conn.send((env.num_agents))
        elif cmd == "get_spaces":
            observation_spaces = {
                agent: env.observation_space(agent) for agent in env.agents
            }
            action_spaces = {agent: env.action_space(agent) for agent in env.agents}
            child_conn.send((observation_spaces, action_spaces))

        elif cmd == "step":
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(data)
            if any(terminated.values()) or any(truncated.values()):
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            child_conn.send(((observation, reward, terminated, truncated, info), True))
        elif cmd == "reset":
            child_conn.send(env.reset())

        else:
            raise NotImplementedError


class SubProcVecParallelEnv(VecParallelEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(self, env_fns: Callable[[Any], ParallelEnv]) -> None:
        self.waiting = False
        self.closed = False
        num_envs = len(env_fns)

        self.parent_conns, self.child_conns = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [
            Process(
                target=worker,
                args=(child_conn, parent_conn, CloudPickleWrapper(env_fn)),
            )
            for (parent_conn, child_conn, env_fn) in zip(
                self.parent_conns, self.child_conns, env_fns
            )
        ]
        for p in self.ps:
            # if the main process crashes, we should not cause things to hang
            p.daemon = True
            p.start()

        for conn in self.child_conns:
            conn.close()

        for conn in self.parent_conns:
            conn.send(("reset", None))
            conn.recv()

        self.parent_conns[0].send((("get_agents", None)))
        self.agents = self.parent_conns[0].recv()

        self.parent_conns[0].send((("get_num_agents", None)))
        self.num_agents = self.parent_conns[0].recv()

        self.parent_conns[0].send(("get_spaces", None))
        observation_spaces, action_spaces = self.parent_conns[0].recv()

        VecParallelEnv.__init__(self, num_envs, observation_spaces, action_spaces)

    def step_async(self, actions):
        unstack_actions = []
        for i in range(len(self.parent_conns)):
            unstack_actions.append(
                {agent: action[i] for agent, action in actions.items()}
            )
        for conn, action in zip(self.parent_conns, unstack_actions):
            conn.send(("step", action))
        self.waiting = True

    def step_wait(self):
        observations = {agent_id: [] for agent_id in self.agents}
        rewards = {agent_id: [] for agent_id in self.agents}
        terminateds = {agent_id: [] for agent_id in self.agents}
        truncateds = {agent_id: [] for agent_id in self.agents}
        infos = {}
        successes = []
        for i, conn in enumerate(self.parent_conns):
            result, success = conn.recv()
            obs, rew, terminated, truncated, info = result
            for agent_id in self.agents:
                observations[agent_id].append(obs[agent_id])
                rewards[agent_id].append(rew[agent_id])
                terminateds[agent_id].append(terminated[agent_id])
                truncateds[agent_id].append(truncated[agent_id])
                infos = self._add_info(infos, info, i)

        self.waiting = False
        return (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        )

    def reset(self):
        for conn in self.parent_conns:
            conn.send(("reset", None))
        results = [conn.recv() for conn in self.parent_conns]
        s, i = zip(*results)
        return stack(s), i

    def close(self):
        if self.closed:
            return

        if self.waiting:
            for conn in self.parent_conns:
                conn.recv()

        for conn in self.parent_conns:
            conn.send(("close", None))

        for p in self.ps:
            p.join()

        self.closed = True

    def _add_info(self, infos: dict, info: dict, env_num: int) -> dict:
        for k in info.keys():
            if k not in infos:
                info_array, array_mask = self._init_info_arrays(type(info[k]))
            else:
                info_array, array_mask = infos[k], infos[f"_{k}"]
            info_array[env_num], array_mask[env_num] = info[k], True
            infos[k], infos[f"_{k}"] = info_array, array_mask
        return infos

    def _init_info_arrays(self, dtype: type) -> tuple[np.ndarray, np.ndarray]:
        if dtype in [int, float, bool] or issubclass(dtype, np.number):
            array = np.zeros(self.num_envs, dtype=dtype)
        else:
            array = np.zeros(self.num_envs, dtype=object)
            array[:] = None
        array_mask = np.zeros(self.num_envs, dtype=bool)
        return array, array_mask


def stack(xs: list[dict[str, np.ndarray]]):
    ks = xs[0].keys()
    vs = list(zip(*[x.values() for x in xs]))
    return {k: np.stack(vs[i]) for i, k in enumerate(ks)}
