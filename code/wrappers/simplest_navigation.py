from typing import SupportsFloat, Any, List, Optional

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper
from mydojo.minecraft import no_op


# Converts the int action space to a box action space
# advantages navigation (no op, forward, back, left, right, turn left, turn right, jump, look up, look down)
# can attack
class SimplestNavigationWrapper(CleanUpFastResetWrapper):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2

    def __init__(self, env, num_actions):
        self.env = env
        super().__init__(self.env)
        self.action_space = gym.spaces.Discrete(num_actions)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = self.int_to_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)

        return (
            obs,
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
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(fast_reset=fast_reset, seed=seed, options=options)
        return obs, info

    def int_to_action(self, input_act: int) -> List[float]:
        act = no_op()
        if input_act == 0:  # go forward
            act[0] = 1  # 0: noop 1: forward 2 : back
        elif input_act == 1:  # Turn left
            act[4] = 12 - 1  # Camera delta yaw (0: -180, 24: 180)
        elif input_act == 2:  # Turn right
            act[4] = 12 + 1  # Camera delta yaw (0: -180, 24: 180)
        return act
