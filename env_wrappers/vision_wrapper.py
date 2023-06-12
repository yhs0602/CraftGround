from typing import SupportsFloat, Any, List

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

from mydojo import MyEnv
from mydojo.minecraft import no_op


# abstract class for sound wrapper
class VisionWrapper(gym.Wrapper):
    def __init__(self, env: MyEnv, action_dim, reward_function=None):
        self.env = env
        super().__init__(self.env)
        self.action_space = gym.spaces.Discrete(action_dim)
        self.reward_function = reward_function
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, env.initial_env.imageSizeX, env.initial_env.imageSizeY),
            dtype=np.uint8,
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = self.int_to_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        rgb = obs["rgb"]
        obs = obs["obs"]
        is_dead = obs.is_dead

        if self.reward_function is None:
            reward = 0.5  # initial reward
            if is_dead:  #
                if self.initial_env.isHardCore:
                    reward = -100
                    terminated = True
                else:  # send respawn packet
                    reward = -1
                    terminated = True

        else:
            reward, terminated = self.reward_function(obs)
        return (
            rgb,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        return obs["rgb"]

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
