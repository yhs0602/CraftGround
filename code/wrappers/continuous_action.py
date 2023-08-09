from enum import Enum
from typing import SupportsFloat, Any, List, Optional, Tuple

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType

from mydojo.minecraft import no_op
from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


class Action(Enum):
    NO_OP = 0
    JUMP = 1
    SNEAK = 2
    SPRINT = 3
    USE = 1
    DROP = 2
    ATTACK = 3
    CRAFT = 4
    EQUIP = 5
    PLACE = 6
    DESTROY = 7


class ContinuousActionWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, enabled_actions, **kwargs):
        super().__init__(env)
        self.no_op = no_op
        if isinstance(enabled_actions[0], str):
            enabled_actions = [Action[action] for action in enabled_actions]
        elif isinstance(enabled_actions[0], int):
            enabled_actions = [Action(action) for action in enabled_actions]
        self.enabled_actions = enabled_actions
        self.continuous_action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.special_move_space = gym.spaces.Discrete(
            len([Action.JUMP, Action.SNEAK, Action.SPRINT])
        )
        self.hand_space = gym.spaces.Discrete(
            len(
                [
                    Action.USE,
                    Action.DROP,
                    Action.ATTACK,
                    Action.CRAFT,
                    Action.EQUIP,
                    Action.PLACE,
                    Action.DESTROY,
                ]
            )
        )
        self.action_space = gym.spaces.Tuple(
            (self.continuous_action_space, self.special_move_space, self.hand_space)
        )
        # self.craft_space = gym.spaces.Discrete(9)
        # self.equip_space = gym.spaces.Discrete(9)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = self.convert_action(action)
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
        fast_reset: bool = True,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(fast_reset=fast_reset, seed=seed, options=options)
        return obs, info

    # act[0] : (-1 ~ 1) -> NoOp, Forward, Backward:
    # act[1] : (-1 ~ 1) -> Strafe Left/Right
    # act[2] : Discrete; NoOp, Jump, Sneak, Sprint
    # act[3] : (-1 ~ 1) Camera delta pitch (0: -90, 24: 90)
    # act[4] : (-1 ~ 1) Camera delta yaw (0: -180, 24: 180)
    # act[5] : Discrete; NoOp, Use, Drop, Attack, Craft, Equip, Place, Destroy
    def convert_action(self, input_act: Tuple[List[float], List[int]]) -> List[float]:
        act = no_op()
        continuous_act = input_act[0]
        discrete_act = input_act[1]
        if continuous_act[0] <= -0.5:
            act[0] = 2  # backward
        elif continuous_act[0] <= 0.5:
            act[0] = 0  # no op
        else:
            act[0] = 1  # forward
        if continuous_act[1] <= -0.5:
            act[1] = 2  # strafe left
        elif continuous_act[1] <= 0.5:
            act[1] = 0  # no op
        else:
            act[1] = 1  # strafe right

        act[3] = int(continuous_act[3] * 12 + 12)
        act[4] = int(continuous_act[4] * 12 + 12)

        act[2] = int(discrete_act[0])  # no op, jump, sneak, sprint
        act[5] = int(
            discrete_act[1]
        )  # no op, use, drop, attack, craft, equip, place, destroy
        act[6] = int(discrete_act[2])  # arg craft
        act[7] = int(discrete_act[3])  # arg inventory
        return act
