from typing import SupportsFloat, Any, List, Optional

import gymnasium as gym
import math
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType


# Sound wrapper
class SoundWrapper(gym.Wrapper):
    def __init__(
        self, env, sound_list, zeroing_sound_list, coord_dim, null_value=0.0, **kwargs
    ):
        self.sound_list = sound_list
        self.zeroing_sound_list = zeroing_sound_list
        self.env = env
        self.coord_dim = coord_dim
        super().__init__(self.env)
        self.zero_offset = len(sound_list) * coord_dim
        self.null_value = null_value
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(len(sound_list) * coord_dim + len(zeroing_sound_list) + 2,),
            dtype=np.float32,
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        sound_subtitles = info_obs.sound_subtitles
        sound_vector = self.encode_sound(
            sound_subtitles, info_obs.x, info_obs.y, info_obs.z, info_obs.yaw
        )
        return (
            np.array(sound_vector, dtype=np.float32),
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def encode_sound(
        self, sound_subtitles: List, x: float, y: float, z: float, yaw: float
    ) -> List[float]:
        sound_vector = [self.null_value] * (
            len(self.sound_list) * self.coord_dim + len(self.zeroing_sound_list) + 2
        )
        for sound in sound_subtitles:
            if abs(sound.x - x) > 16 or abs(sound.z - z) > 16:
                continue
            if self.coord_dim == 3 and abs(sound.y - y) > 16:
                continue

            if sound.translate_key in self.sound_list:
                dx = sound.x - x
                if self.coord_dim == 3:
                    dy = sound.y - y
                else:
                    dy = 0
                dz = sound.z - z

                idx = self.sound_list.index(sound.translate_key)
                offset = idx * self.coord_dim

                if self.coord_dim == 2:
                    sound_vector[offset] = dx / 15
                    sound_vector[offset + 1] = dz / 15
                else:
                    sound_vector[offset] = dx / 15
                    sound_vector[offset + 1] = dy / 15
                    sound_vector[offset + 2] = dz / 15
            elif sound.translate_key in self.zeroing_sound_list:
                idx = self.zeroing_sound_list.index(sound.translate_key)
                offset = self.zero_offset + idx
                sound_vector[offset] = 1
        # Trigonometric encoding
        yaw_radians = math.radians(yaw)
        sound_vector[-2] = math.cos(yaw_radians)
        sound_vector[-1] = math.sin(yaw_radians)

        return sound_vector

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        obs, info = self.env.reset(seed=seed, options=options)
        obs_info = info["obs"]
        sound_subtitles = obs_info.sound_subtitles
        sound_vector = self.encode_sound(
            sound_subtitles, obs_info.x, obs_info.y, obs_info.z, obs_info.yaw
        )
        return np.array(sound_vector, dtype=np.float32), info
