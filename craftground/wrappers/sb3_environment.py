from collections import OrderedDict
from typing import List, Callable, Dict, Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env


class StableBaseline3Wrapper(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata


def obs_space_info(
    obs_space: spaces.Space, parent_key: str = None
) -> Tuple[List[str], Dict[Any, Tuple[int, ...]], Dict[Any, np.dtype]]:
    """
    Get dict-structured information about a gym.Space.

    Dict spaces are represented directly by their dict of subspaces.
    Tuple spaces are converted into a dict with keys indexing into the tuple.
    Unstructured spaces are represented by {None: obs_space}.

    :param obs_space: an observation space
    :param parent_key: a prefix to prepend to dict keys
    :return: A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    # check_for_nested_spaces(obs_space)
    if isinstance(obs_space, spaces.Dict):
        assert isinstance(
            obs_space.spaces, OrderedDict
        ), "Dict space must have ordered subspaces"
        subspaces = obs_space.spaces
    elif isinstance(obs_space, spaces.Tuple):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}  # type: ignore[assignment]
    elif isinstance(obs_space, spaces.Sequence):
        subspaces = {}  # type: ignore[assignment]
    else:
        assert not hasattr(
            obs_space, "spaces"
        ), f"Unsupported structured space '{type(obs_space)}'"
        subspaces = {None: obs_space}  # type: ignore[assignment]
    keys = []
    shapes = {}
    dtypes = {}
    for key, space in subspaces.items():
        if parent_key is not None:
            full_key = f"{parent_key}/{key}" if key is not None else parent_key
        else:
            full_key = str(key) if key is not None else None
        if isinstance(space, (spaces.Dict, spaces.Tuple)):
            nested_keys, nested_shapes, nested_dtypes = obs_space_info(space, full_key)
            keys.extend(nested_keys)
            shapes.update(nested_shapes)
            dtypes.update(nested_dtypes)
        else:
            keys.append(full_key)
            shapes[full_key] = space.shape if space.shape is not None else ()
            dtypes[full_key] = space.dtype if space.dtype is not None else np.float32
    return keys, shapes, dtypes
