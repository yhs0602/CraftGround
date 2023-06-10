import math
from typing import List, SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from mydojo.minecraft import int_to_action


class HuskSoundWrapper(gym.Wrapper):
    def __init__(self, verbose=False, env_path=None, port=8000):
        self.env = mydojo.make(
            verbose=verbose,
            env_path=env_path,
            port=port,
            initialInventoryCommands=[],
            initialPosition=None,  # nullable
            initialMobsCommands=[
                # "minecraft:sheep",
                "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                # player looks at south (positive Z) when spawn
            ],
            imageSizeX=114,
            imageSizeY=64,
            visibleSizeX=114,
            visibleSizeY=64,
            seed=12345,  # nullable
            allowMobSpawn=False,
            alwaysDay=True,
            alwaysNight=False,
            initialWeather="clear",  # nullable
            isHardCore=False,
            isWorldFlat=True,  # superflat world
            obs_keys=["sound_subtitles"],
        )
        super().__init__(self.env)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = int_to_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        # rgb = obs["rgb"]
        obs = obs["obs"]
        is_dead = obs.is_dead
        sound_subtitles = obs.sound_subtitles
        sound_vector = self.encode_sound_and_yaw(sound_subtitles, obs.x, obs.z, obs.yaw)

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

    @staticmethod
    def encode_sound_and_yaw(
        sound_subtitles, x: float, z: float, yaw: float
    ) -> List[float]:
        sound_vector = [0.0] * 7
        for sound in sound_subtitles:
            if sound.x - x > 16 or sound.z - z > 16:
                continue
            if sound.x - x < -16 or sound.z - z < -16:
                continue
            if sound.translate_key == "subtitles.entity.husk.ambient":
                # normalize
                dx = sound.x - x
                dz = sound.z - z
                distance = math.sqrt(dx * dx + dz * dz)
                if distance > 0:
                    sound_vector[0] = dx / distance
                    sound_vector[1] = dz / distance
            elif sound.translate_key == "subtitles.block.generic.footsteps":
                # normalize
                dx = sound.x - x
                dz = sound.z - z
                distance = math.sqrt(dx * dx + dz * dz)
                if distance > 0:
                    sound_vector[2] = dx / distance
                    sound_vector[3] = dz / distance
            elif sound.translate_key == "subtitles.entity.player.hurt":
                sound_vector[4] = 1
        # Trigonometric encoding
        yaw_radians = math.radians(yaw)
        sound_vector[5] = math.sin(yaw_radians)
        sound_vector[6] = math.cos(yaw_radians)
        return sound_vector

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        obs = obs["obs"]
        sound_subtitles = obs.sound_subtitles
        sound_vector = self.encode_sound_and_yaw(sound_subtitles, obs.x, obs.z, obs.yaw)
        return np.array(sound_vector, dtype=np.float32)
