import math
from typing import SupportsFloat, Any, List

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

from mydojo.minecraft import no_op


# abstract class for sound wrapper
class SoundWrapper(gym.Wrapper):
    def __init__(self, env, action_dim, sound_list, coord_dim):
        self.sound_list = sound_list
        self.env = env
        self.coord_dim = coord_dim
        super().__init__(self.env)
        self.action_space = gym.spaces.Discrete(action_dim)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(len(sound_list) * coord_dim + 3,), dtype=np.float32
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = self.int_to_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        # rgb = obs["rgb"]
        obs = obs["obs"]
        is_dead = obs.is_dead
        sound_subtitles = obs.sound_subtitles
        sound_vector = self.encode_sound(sound_subtitles, obs.x, obs.y, obs.z, obs.yaw)

        reward = 0.5  # initial reward
        if is_dead:  #
            if self.initial_env.isHardCore:
                reward = -100
                terminated = True
            else:  # send respawn packet
                reward = -1
                terminated = True
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
        sound_vector = [0] * (len(self.sound_list) * self.coord_dim + 3)
        for sound in sound_subtitles:
            if sound.x - x > 16 or sound.z - z > 16:
                continue
            if sound.x - x < -16 or sound.z - z < -16:
                continue
            if self.coord_dim == 3 and sound.y - y < -16 or sound.y - y > 16:
                continue
            for idx, translation_key in enumerate(self.sound_list):
                if translation_key == sound.translate_key:
                    dx = sound.x - x
                    if self.coord_dim == 3:
                        dy = sound.y - y
                    else:
                        dy = 0
                    dz = sound.z - z
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if distance > 0:
                        if self.coord_dim == 2:
                            sound_vector[idx * self.coord_dim] = dx / distance
                            sound_vector[idx * self.coord_dim + 1] = dz / distance
                        else:
                            sound_vector[idx * self.coord_dim] = dx / distance
                            sound_vector[idx * self.coord_dim + 1] = dy / distance
                            sound_vector[idx * self.coord_dim + 2] = dz / distance
                elif translation_key == "subtitles.entity.player.hurt":
                    sound_vector[-1] = 1  # player hurt sound

        # Trigonometric encoding
        yaw_radians = math.radians(yaw)
        sound_vector[-3] = math.cos(yaw_radians)
        sound_vector[-2] = math.sin(yaw_radians)

        return sound_vector

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        obs = obs["obs"]
        sound_subtitles = obs.sound_subtitles
        sound_vector = self.encode_sound(sound_subtitles, obs.x, obs.y, obs.z, obs.yaw)
        return np.array(sound_vector, dtype=np.float32)

    def int_to_action(self, input_act: int) -> List[float]:
        act = no_op()
        # act=0: no op
        if input_act == 1:  # go forward
            act[0] = 1  # 0: noop 1: forward 2 : back
        elif input_act == 2:  # go backward
            act[0] = 2  # 0: noop 1: forward 2 : back
        elif input_act == 3:  # move right
            act[1] = 1  # 0: noop 1: move right 2: move left
        elif input_act == 4:  # move left
            act[1] = 2  # 0: noop 1: move right 2: move left
        elif input_act == 5:  # Turn left
            act[4] = 12 - 1  # Camera delta yaw (0: -180, 24: 180)
        elif input_act == 6:  # Turn right
            act[4] = 12 + 1  # Camera delta yaw (0: -180, 24: 180)
        return act
