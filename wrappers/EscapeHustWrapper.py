from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from mydojo.minecraft import int_to_action, send_respawn


class EscapeHuskWrapper(gym.Wrapper):
    def __init__(self):
        self.env = mydojo.make(
            initialInventoryCommands=["minecraft:diamond_sword", "minecraft:shield"],
            initialPosition=None,  # nullable
            initialMobsCommands=[
                "minecraft:sheep",
                "minecraft:husk ~10 ~ ~ {HandItems:[{Count:1,id:wooden_shovel},{}]}",
            ],
            imageSizeX=64,
            imageSizeY=64,
            visibleSizeX=400,
            visibleSizeY=225,
            seed=12345,  # nullable
            allowMobSpawn=False,
            alwaysDay=False,
            alwaysNight=False,
            initialWeather="clear",  # nullable
            isHardCore=False,
            isWorldFlat=True,  # superflat world
        )
        super(EscapeHuskWrapper, self).__init__(self.env)
        self.action_space = gym.spaces.Discrete(4)
        initial_env = self.env.initial_env
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, initial_env.imageSizeX, initial_env.imageSizeY),
            dtype=np.uint8,
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = int_to_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        rgb = obs["rgb"]
        obs = obs["obs"]
        is_dead = obs.is_dead

        reward = 1  # initial reward
        if is_dead:  #
            if self.initial_env.isHardCore:
                reward = -10000000
                terminated = True
            else:  # send respawn packet
                # pass
                reward = -200
                terminated = True
                # send_respawn(self.json_socket)
                # print("Dead!!!!!")
                # res = self.json_socket.receive_json()  # throw away
        return rgb, reward, terminated, truncated, info  # , done: deprecated

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        return obs
