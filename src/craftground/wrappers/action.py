from enum import Enum
from typing import SupportsFloat, Any, List, Optional

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType

from minecraft import no_op


# Converts the int action space to a box action space
# advantages navigation (no op, forward, back, left, right, turn left, turn right, jump, look up, look down)
# can attack


# convert int to box
class Action(Enum):
    NO_OP = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6
    JUMP = 7
    LOOK_UP = 8
    LOOK_DOWN = 9
    ATTACK = 10
    USE = 11
    JUMP_USE = 12


class ActionWrapper(gym.Wrapper):
    enabled_actions: List[Action]

    def __init__(self, env, enabled_actions, **kwargs):
        super().__init__(env)
        self.no_op = no_op
        if isinstance(enabled_actions[0], str):
            enabled_actions = [Action[action] for action in enabled_actions]
        elif isinstance(enabled_actions[0], int):
            enabled_actions = [Action(action) for action in enabled_actions]
        self.enabled_actions = enabled_actions
        self.action_space = gym.spaces.Discrete(len(enabled_actions))

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_enum: Action = self.enabled_actions[action]
        action_arr = self.int_to_action(action_enum)
        # print(f"Final Action: {action_arr}")
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
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def int_to_action(self, input_act: Action) -> List[float]:  # noqa: C901
        act = no_op()
        # act=0: no op
        if input_act == Action.FORWARD:  # go forward
            act[0] = 1  # 0: noop 1: forward 2 : back
        elif input_act == Action.BACKWARD:  # go backward
            act[0] = 2  # 0: noop 1: forward 2 : back
        elif input_act == Action.STRAFE_RIGHT:  # move right
            act[1] = 1  # 0: noop 1: move right 2: move left
        elif input_act == Action.STRAFE_LEFT:  # move left
            act[1] = 2  # 0: noop 1: move right 2: move left
        elif input_act == Action.TURN_LEFT:  # Turn left
            act[4] = 12 - 1  # Camera delta yaw (0: -180, 24: 180)
        elif input_act == Action.TURN_RIGHT:  # Turn right
            act[4] = 12 + 1  # Camera delta yaw (0: -180, 24: 180)
        elif input_act == Action.JUMP:  # Jump
            act[2] = 1  # 0: noop 1: jump
        elif input_act == Action.LOOK_UP:  # Look up
            act[3] = 12 - 1  # Camera delta pitch (0: -180, 24: 180)
        elif input_act == Action.LOOK_DOWN:  # Look down
            act[3] = 12 + 1  # Camera delta pitch (0: -180, 24: 180)
        elif input_act == Action.ATTACK:  # attack
            act[5] = (
                3  # 0: noop 1: use 2: drop 3: attack 4: craft 5: equip 6: place 7: destroy
            )
        elif input_act == Action.USE:  # use
            act[5] = (
                1  # 0: noop 1: use 2: drop 3: attack 4: craft 5: equip 6: place 7: destroy
            )
        elif input_act == Action.JUMP_USE:  # use while jumping
            act[2] = 1  # 0: noop 1: jump
            act[5] = 1
        return act
