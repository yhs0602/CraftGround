from enum import Enum
from gymnasium.core import ActType
from typing import Dict, List, Union
import gymnasium as gym
import numpy as np

from craftground.proto.action_space_pb2 import ActionSpaceMessageV2


class ActionSpaceVersion(Enum):
    V1_MINEDOJO = 1
    V2_MINERL_HUMAN = 2


def no_op() -> List[int]:
    r = [0] * 8
    r[3] = 12
    r[4] = 12
    return r


def no_op_v2() -> Dict[str, Union[bool, float]]:
    noop_dict = {}
    for bool_key in [
        "attack",
        "back",
        "forward",
        "jump",
        "left",
        "right",
        "sneak",
        "sprint",
        "use",
        "drop",
        "inventory",
    ]:
        noop_dict[bool_key] = False
    for i in range(1, 10):
        noop_dict[f"hotbar.{i}"] = False
    noop_dict["camera_pitch"] = 0.0
    noop_dict["camera_yaw"] = 0.0
    return noop_dict


def translate_action_to_v2(action: ActType) -> Dict[str, Union[bool, float]]:
    translated_action = {
        "attack": action[5] == 3,
        "back": action[0] == 2,
        "forward": action[0] == 1,
        "jump": action[2] == 1,
        "left": action[1] == 1,
        "right": action[1] == 2,
        "sneak": action[2] == 2,
        "sprint": action[2] == 3,
        "use": action[5] == 1,
        "drop": action[5] == 2,
        "inventory": False,
    }
    for i in range(1, 10):
        translated_action[f"hotbar.{i}"] = False

    translated_action["camera_pitch"] = action[3] * 15 - 180.0
    translated_action["camera_yaw"] = action[4] * 15 - 180.0

    return translated_action


def action_v2_dict_to_message(
    action_v2: Dict[str, Union[bool, float]]
) -> ActionSpaceMessageV2:
    action_space = ActionSpaceMessageV2()
    action_space.attack = action_v2["attack"]
    action_space.back = action_v2["back"]
    action_space.forward = action_v2["forward"]
    action_space.jump = action_v2["jump"]
    action_space.left = action_v2["left"]
    action_space.right = action_v2["right"]
    action_space.sneak = action_v2["sneak"]
    action_space.sprint = action_v2["sprint"]
    action_space.use = action_v2["use"]
    action_space.drop = action_v2["drop"]
    action_space.inventory = action_v2["inventory"]
    action_space.hotbar_1 = action_v2["hotbar.1"]
    action_space.hotbar_2 = action_v2["hotbar.2"]
    action_space.hotbar_3 = action_v2["hotbar.3"]
    action_space.hotbar_4 = action_v2["hotbar.4"]
    action_space.hotbar_5 = action_v2["hotbar.5"]
    action_space.hotbar_6 = action_v2["hotbar.6"]
    action_space.hotbar_7 = action_v2["hotbar.7"]
    action_space.hotbar_8 = action_v2["hotbar.8"]
    action_space.hotbar_9 = action_v2["hotbar.9"]
    action_space.camera_pitch = action_v2["camera_pitch"]
    action_space.camera_yaw = action_v2["camera_yaw"]
    return action_space


def action_to_symbol(action) -> str:  # noqa: C901
    res = ""
    if action[0] == 1:
        res += "↑"
    elif action[0] == 2:
        res += "↓"
    if action[1] == 1:
        res += "←"
    elif action[1] == 2:
        res += "→"
    if action[2] == 1:
        res += "jump"  # "⤴"
    elif action[2] == 2:
        res += "sneak"  # "⤵"
    elif action[2] == 3:
        res += "sprint"  # "⚡"
    if action[3] > 12:  # pitch up
        res += "⤒"
    elif action[3] < 12:  # pitch down
        res += "⤓"
    if action[4] > 12:  # yaw right
        res += "⏭"
    elif action[4] < 12:  # yaw left
        res += "⏮"
    if action[5] == 1:  # use
        res += "use"  # "⚒"
    elif action[5] == 2:  # drop
        res += "drop"  # "🤮"
    elif action[5] == 3:  # attack
        res += "attack"  # "⚔"
    return res


def action_v2_to_symbol(action_v2: Dict[str, Union[int, float]]) -> str:  # noqa: C901
    res = ""

    if action_v2.get("forward") == 1:
        res += "↑"
    if action_v2.get("backward") == 1:
        res += "↓"
    if action_v2.get("left") == 1:
        res += "←"
    if action_v2.get("right") == 1:
        res += "→"
    if action_v2.get("jump") == 1:
        res += "JMP"
    if action_v2.get("sneak") == 1:
        res += "SNK"
    if action_v2.get("sprint") == 1:
        res += "SPRT"
    if action_v2.get("attack") == 1:
        res += "ATK"
    if action_v2.get("use") == 1:
        res += "USE"
    if action_v2.get("drop") == 1:
        res += "Q"
    if action_v2.get("inventory") == 1:
        res += "I"

    for i in range(1, 10):
        if action_v2.get(f"hotbar.{i}") == 1:
            res += f"hotbar.{i}"

    return res


def declare_action_space(action_space_version: ActionSpaceVersion) -> gym.spaces.Space:
    if action_space_version == ActionSpaceVersion.V1_MINEDOJO:
        # Same as the action space used in MineDojo
        return gym.spaces.MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])
    elif action_space_version == ActionSpaceVersion.V2_MINERL_HUMAN:
        return gym.spaces.Dict(
            {
                "attack": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "back": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "forward": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "jump": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "left": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "right": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "sneak": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "sprint": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "use": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "drop": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "inventory": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "hotbar.1": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "hotbar.2": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "hotbar.3": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "hotbar.4": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "hotbar.5": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "hotbar.6": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "hotbar.7": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "hotbar.8": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "hotbar.9": gym.spaces.Discrete(2),  # 0 or 1 (boolean)
                "camera": gym.spaces.Box(
                    low=np.array([-180, -180]),
                    high=np.array([180, 180]),
                    dtype=np.float32,
                ),
                # Camera pitch/yaw between -180 and 180 degrees
            }
        )
    else:
        raise ValueError(f"Unknown action space version: {action_space_version}")
