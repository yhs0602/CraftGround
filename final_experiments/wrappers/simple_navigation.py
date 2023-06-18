from typing import SupportsFloat, Any, List, Optional

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType

from mydojo.minecraft import no_op


# Converts the int action space to a box action space
# advantages navigation (no op, forward, back, left, right, turn left, turn right, jump, look up, look down)
# can attack
class SimpleNavigationWrapper(gym.Wrapper):
    NO_OP = 0
    FORWARD = 1
    BACKWARD = 2
    MOVE_RIGHT = 3
    MOVE_LEFT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6
    JUMP = 7
    LOOK_UP = 8
    LOOK_DOWN = 9
    ATTACK = 10

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
        options: Optional[dict[str, Any]] = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(fast_reset=fast_reset, seed=seed, options=options)
        return obs, info

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
        elif input_act == 7:  # Jump
            act[2] = 1  # 0: noop 1: jump
        elif input_act == 8:  # Look up
            act[3] = 12 + 1  # Camera delta pitch (0: -180, 24: 180)
        elif input_act == 9:  # Look down
            act[3] = 12 - 1  # Camera delta pitch (0: -180, 24: 180)
        elif input_act == 10:  # attack
            act[5] = 3
        return act
