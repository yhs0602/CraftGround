from typing import SupportsFloat, Any, Optional, List, Dict

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.vector.utils import spaces

from . import token_providers
from wrappers import BimodalWrapper


# Sound wrapper
class BimodalTokenWrapper(BimodalWrapper):
    def __init__(
        self,
        env,
        x_dim,
        y_dim,
        sound_list: List[str],
        sound_coord_dim: int = 2,
        token_provider_configs: List[Dict] = None,
        **kwargs,
    ):
        self.env = env
        self.sound_list = sound_list
        self.sound_coord_dim = sound_coord_dim
        self.token_providers_configs = token_providers
        self.token_providers = []
        for token_provider_config in token_provider_configs:
            name = token_provider_config["name"]
            token_provider_cls = getattr(token_providers, name)
            token_provider = token_provider_cls(**token_provider_config)
            self.token_providers.append(token_provider)
        self.token_dim = sum(
            [token_provider.token_dim for token_provider in self.token_providers]
        )

        super().__init__(self.env, x_dim, y_dim, sound_list, sound_coord_dim, **kwargs)
        self.observation_space = spaces.Dict(
            {
                "vision": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, x_dim, y_dim),
                    dtype=np.uint8,
                ),
                "sound": gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=(len(sound_list) * sound_coord_dim + 3,),
                    dtype=np.float32,
                ),
                "token": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.token_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        rgb = info["rgb"]
        obs_info = info["obs"]
        sound_subtitles = obs_info.sound_subtitles
        sound_vector = self.encode_sound(
            sound_subtitles, obs_info.x, obs_info.y, obs_info.z, obs_info.yaw
        )
        token = np.zeros(shape=(self.token_dim,), dtype=np.float32)
        for token_provider in self.token_providers:
            token_provider.provide_token_step(obs, info, token)
        return (
            {
                "vision": rgb,
                "sound": np.array(sound_vector, dtype=np.float32),
                "token": token,
            },
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        obs, info = self.env.reset(seed=seed, options=options)
        rgb = info["rgb"]
        obs_info = info["obs"]
        sound_subtitles = obs_info.sound_subtitles
        sound_vector = self.encode_sound(
            sound_subtitles, obs_info.x, obs_info.y, obs_info.z, obs_info.yaw
        )
        token = np.zeros(shape=(self.token_dim,), dtype=np.float32)
        for token_provider in self.token_providers:
            token_provider.provide_token_reset(obs, info, token)
        return {
            "vision": rgb,
            "sound": np.array(sound_vector, dtype=np.float32),
            "token": token,
        }, info
