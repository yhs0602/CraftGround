from collections import deque
from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from mydojo.minecraft import int_to_action
from wrapper_runner import WrapperRunner


class EscapeHuskWrapper(gym.Wrapper):
    def __init__(self):
        self.env = mydojo.make(
            initialInventoryCommands=["minecraft:diamond_sword"],
            initialPosition=None,  # nullable
            initialMobsCommands=[
                # "minecraft:sheep",
                "minecraft:husk ~5 ~ ~ {HandItems:[{Count:1,id:iron_shovel},{}]}",
            ],
            initialExtraCommands=[
                "item replace entity @p weapon.offhand with minecraft:shield"
            ],
            imageSizeX=114,
            imageSizeY=64,
            visibleSizeX=400,
            visibleSizeY=225,
            seed=12345,  # nullable
            allowMobSpawn=False,
            alwaysDay=True,
            alwaysNight=False,
            initialWeather="clear",  # nullable
            isHardCore=False,
            isWorldFlat=True,  # superflat world
        )
        super(EscapeHuskWrapper, self).__init__(self.env)
        self.action_space = gym.spaces.Discrete(11)
        initial_env = self.env.initial_env
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, initial_env.imageSizeX, initial_env.imageSizeY),
            dtype=np.uint8,
        )
        self.health_deque = deque(maxlen=2)
        self.weapon_deque = deque(maxlen=2)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = int_to_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        rgb = obs["rgb"]
        obs = obs["obs"]
        is_dead = obs.is_dead
        self.health_deque.append(obs.health)
        if len(obs.inventory) > 0:
            durability = obs.inventory[0].durability
            self.weapon_deque.append(durability)

        is_hit = self.health_deque[0] > self.health_deque[1]
        had_hit = self.weapon_deque[0] > self.weapon_deque[1]  # durability decreased

        reward = 1  # initial reward
        if is_hit:
            reward = -5  # penalty
        if had_hit:
            reward += 10  # bonus
        if is_dead:  #
            if self.initial_env.isHardCore:
                reward = -10000000
                terminated = True
            else:  # send respawn packet
                reward = -200
                terminated = True
                # send_respawn(self.json_socket)
                # print("Dead!!!!!")
                # res = self.json_socket.receive_json()  # throw away

        return rgb, reward, terminated, truncated, info  # , done: deprecated

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        self.health_deque.clear()
        self.health_deque.append(20)
        return obs


def main():
    env = EscapeHuskWrapper()
    buffer_size = 1000000
    batch_size = 20
    gamma = 0.95
    learning_rate = 0.001
    runner = WrapperRunner(
        env, "HuntHuskWrapper", buffer_size, batch_size, gamma, learning_rate
    )
    runner.run_wrapper()


if __name__ == "__main__":
    main()
