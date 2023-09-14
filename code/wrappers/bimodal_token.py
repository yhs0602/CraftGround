from typing import SupportsFloat, Any, Optional, List

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.vector.utils import spaces

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
        token_dim: int = 2,
        **kwargs,
    ):
        self.env = env
        self.sound_list = sound_list
        self.sound_coord_dim = sound_coord_dim
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
                    shape=(token_dim,),
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
        token = [obs_info.bobber_thrown]
        return (
            {
                "vision": rgb,
                "sound": np.array(sound_vector, dtype=np.float32),
                "token": np.array(token, dtype=np.float32),
            },
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(
        self,
        fast_reset: bool = True,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        obs, info = self.env.reset(fast_reset=fast_reset, seed=seed, options=options)
        rgb = info["rgb"]
        obs_info = info["obs"]
        sound_subtitles = obs_info.sound_subtitles
        sound_vector = self.encode_sound(
            sound_subtitles, obs_info.x, obs_info.y, obs_info.z, obs_info.yaw
        )
        token = [obs_info.bobber_thrown]
        return {
            "vision": rgb,
            "sound": np.array(sound_vector, dtype=np.float32),
            "token": np.array(token, dtype=np.float32),
        }, info
