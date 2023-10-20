from collections import OrderedDict
from copy import deepcopy
from typing import Any, Optional, List, Type, Dict

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvStepReturn,
    VecEnvObs,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import (
    obs_space_info,
    dict_to_obs,
    copy_obs_dict,
)


# Sound wrapper
class StableBaseline3Wrapper(VecEnv):
    actions: np.ndarray

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        env_idx = 0
        obs, self.buf_rews[0], terminated, truncated, self.buf_infos[0] = self.env.step(
            self.actions[0]
        )
        # convert to SB3 VecEnv api
        self.buf_dones[0] = terminated or truncated
        # See https://github.com/openai/gym/issues/3102
        # Gym 0.26 introduces a breaking change
        self.buf_infos[0]["TimeLimit.truncated"] = truncated and not terminated

        if self.buf_dones[0]:
            # save final observation where user can get it, then reset
            self.buf_infos[0]["terminal_observation"] = obs
            obs, self.reset_infos[0] = self.env.reset()
        self._save_obs(0, obs)
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        return [self.env]

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][0] = obs
            else:
                self.buf_obs[key][0] = obs[key]  # type: ignore[call-overload]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def __init__(
        self,
        env: gym.Env,
    ):
        self.env = _patch_env(env)
        env = self.env
        super().__init__(1, env.observation_space, env.action_space)
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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
    ):
        obs, self.reset_infos[0] = self.env.reset(seed=self._seeds[0])
        self._save_obs(0, obs)
        # Seeds are only used once
        self._reset_seeds()
        return self._obs_from_buf()
