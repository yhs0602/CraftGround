from typing import SupportsFloat, Any, Optional, List

import gymnasium as gym
import math
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.vector.utils import spaces


# Sound wrapper
class BimodalWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        x_dim,
        y_dim,
        sound_list: List[str],
        sound_coord_dim: int = 2,
        **kwargs,
    ):
        self.env = env
        self.sound_list = sound_list
        self.sound_coord_dim = sound_coord_dim
        super().__init__(self.env)
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
        return (
            {
                "vision": rgb,
                "sound": np.array(sound_vector, dtype=np.float32),
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
        return {
            "vision": rgb,
            "sound": np.array(sound_vector, dtype=np.float32),
        }, info

    def encode_sound(
        self, sound_subtitles: List, x: float, y: float, z: float, yaw: float
    ) -> List[float]:
        sound_vector = [0] * (len(self.sound_list) * self.sound_coord_dim + 3)
        for sound in sound_subtitles:
            if sound.x - x > 16 or sound.z - z > 16:
                continue
            if sound.x - x < -16 or sound.z - z < -16:
                continue
            if self.sound_coord_dim == 3 and (sound.y - y < -16 or sound.y - y > 16):
                continue
            for idx, translation_key in enumerate(self.sound_list):
                if translation_key == sound.translate_key:
                    dx = sound.x - x
                    if self.sound_coord_dim == 3:
                        dy = sound.y - y
                    else:
                        dy = 0
                    dz = sound.z - z
                    if self.sound_coord_dim == 2:
                        sound_vector[idx * self.sound_coord_dim] = dx / 15
                        sound_vector[idx * self.sound_coord_dim + 1] = dz / 15
                    else:
                        sound_vector[idx * self.sound_coord_dim] = dx / 15
                        sound_vector[idx * self.sound_coord_dim + 1] = dy / 15
                        sound_vector[idx * self.sound_coord_dim + 2] = dz / 15
                elif translation_key == "subtitles.entity.player.hurt":
                    sound_vector[-1] = 1  # player hurt sound

        # Trigonometric encoding
        yaw_radians = math.radians(yaw)
        sound_vector[-3] = math.cos(yaw_radians)
        sound_vector[-2] = math.sin(yaw_radians)

        return sound_vector
