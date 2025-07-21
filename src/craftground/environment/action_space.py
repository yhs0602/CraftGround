from enum import Enum
from gymnasium.core import ActType
from typing import Dict, List, Union
import gymnasium as gym
import numpy as np

from craftground.proto.action_space_pb2 import ActionSpaceMessageV2


class ActionSpaceVersion(Enum):
    V1_MINEDOJO = 1
    V2_MINERL_HUMAN = 2


class ActionV1(np.ndarray):
    class Movement(Enum, int):
        NO_OP = 0
        FORWARD = 1
        BACKWARD = 2

    class Strafe(Enum, int):
        NO_OP = 0
        LEFT = 1
        RIGHT = 2

    class MovementModifier(Enum, int):
        NO_OP = 0
        JUMP = 1
        SNEAK = 2
        SPRINT = 3

    class Interaction(Enum, int):
        NO_OP = 0
        USE = 1
        DROP = 2
        ATTACK = 3
        CRAFT = 4
        EQUIP = 5
        PLACE = 6
        DESTROY = 7

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def movement(self) -> int:
        return self[ActionSpaceV1.Index.MOVEMENT.value]

    @movement.setter
    def movement(self, value: int):
        self[ActionSpaceV1.Index.MOVEMENT.value] = value

    @property
    def strafe(self) -> int:
        return self[ActionSpaceV1.Index.STRAFING.value]

    @strafe.setter
    def strafe(self, value: int):
        self[ActionSpaceV1.Index.STRAFING.value] = value

    @property
    def movement_modifier(self) -> int:
        return self[ActionSpaceV1.Index.MOVEMENT_MODIFIERS.value]

    @movement_modifier.setter
    def movement_modifier(self, value: int):
        self[ActionSpaceV1.Index.MOVEMENT_MODIFIERS.value] = value

    @property
    def camera_pitch(self) -> int:
        """
        Pitch: Player is facing up (negative) or down (positive)
        The value is between -90.0(-6) to 90.0(6) degrees.
        Negative Pitch Delta means looking up
        Positive Pitch Delta means looking down
        """
        return self[ActionSpaceV1.Index.CAMERA_PITCH.value]

    @camera_pitch.setter
    def camera_pitch(self, value: int):
        self[ActionSpaceV1.Index.CAMERA_PITCH.value] = value

    @property
    def camera_yaw(self) -> int:
        """
        Yaw: Player's horizontal rotation
        North (Negative Z) : 180.0 / -180.0
        East (Positive X) : -90.0
        South (Positive Z): 0.0
        West (Negative X) : 90.0

        Negative Yaw Delta means turning CCW (left)
        Positive Yaw Delta means turning CW (right)
        """
        return self[ActionSpaceV1.Index.CAMERA_YAW.value]

    @camera_yaw.setter
    def camera_yaw(self, value: int):
        self[ActionSpaceV1.Index.CAMERA_YAW.value] = value

    @property
    def interaction(self) -> int:
        return self[ActionSpaceV1.Index.INTERACTION.value]

    @interaction.setter
    def interaction(self, value: int):
        self[ActionSpaceV1.Index.INTERACTION.value] = value

    @property
    def crafting_arg(self) -> int:
        return self[ActionSpaceV1.Index.CRAFTING_ARG.value]

    @crafting_arg.setter
    def crafting_arg(self, value: int):
        self[ActionSpaceV1.Index.CRAFTING_ARG.value] = value

    @property
    def item_arg(self) -> int:
        return self[ActionSpaceV1.Index.ITEM_ARG.value]

    @item_arg.setter
    def item_arg(self, value: int):
        self[ActionSpaceV1.Index.ITEM_ARG.value] = value

    def __repr__(self) -> str:
        movement_str = ActionV1.Movement(self.movement).name.lower()
        strafe_str = ActionV1.Strafe(self.strafe).name.lower()
        movement_modifier_str = ActionV1.MovementModifier(
            self.movement_modifier
        ).name.lower()
        camera_pitch_str = f"{self.camera_pitch * 15 - 180:.2f}"
        camera_yaw_str = f"{self.camera_yaw * 15 - 180:.2f}"
        interaction_str = ActionV1.Interaction(self.interaction).name.lower()
        return f"ActionV1(Movement({movement_str}), Strafe({strafe_str}), MovementModifier({movement_modifier_str}), CameraPitch({camera_pitch_str}), CameraYaw({camera_yaw_str}), Interaction({interaction_str}({self.crafting_arg}/{self.item_arg}))"


# MineDojo action space v1
class ActionSpaceV1(gym.spaces.MultiDiscrete):
    NO_OP: ActionV1 = ActionV1(np.array([0, 0, 0, 12, 12, 0, 0, 0], dtype=np.int32))
    FORWARD: ActionV1 = ActionV1(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32))
    BACKWARD: ActionV1 = ActionV1(np.array([2, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32))
    LEFT_STRAFE: ActionV1 = ActionV1(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32))
    RIGHT_STRAFE: ActionV1 = ActionV1(
        np.array([0, 2, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    )
    JUMP: ActionV1 = ActionV1(np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int32))
    TURN_LEFT: ActionV1 = ActionV1(
        np.array([0, 0, 0, 0, 12 - 1, 0, 0, 0], dtype=np.int32)
    )
    TURN_RIGHT: ActionV1 = ActionV1(
        np.array([0, 0, 0, 0, 12 + 1, 0, 0, 0], dtype=np.int32)
    )
    LOOK_UP: ActionV1 = ActionV1(
        np.array([0, 0, 0, 12 - 1, 0, 0, 0, 0], dtype=np.int32)
    )
    LOOK_DOWN: ActionV1 = ActionV1(
        np.array([0, 0, 0, 12 + 1, 0, 0, 0, 0], dtype=np.int32)
    )
    USE: ActionV1 = ActionV1(np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int32))
    DROP: ActionV1 = ActionV1(np.array([0, 0, 0, 0, 0, 2, 0, 0], dtype=np.int32))
    ATTACK: ActionV1 = ActionV1(np.array([0, 0, 0, 0, 0, 3, 0, 0], dtype=np.int32))

    class Index(Enum, int):
        MOVEMENT = 0
        STRAFING = 1
        MOVEMENT_MODIFIERS = 2
        CAMERA_PITCH = 3
        CAMERA_YAW = 4
        INTERACTION = 5
        CRAFTING_ARG = 6
        ITEM_ARG = 7

    def __init__(self):
        super().__init__([3, 3, 4, 25, 25, 8, 244, 36])

    def __repr__(self) -> str:
        return f"ActionSpaceV1(Movement(3), Strafe(3), MovementModifier(4), CameraPitch(25), CameraYaw(25), Interaction(8), CraftingArg(244), ItemArg(36))"

    @staticmethod
    def no_op() -> ActionV1:
        return ActionSpaceV1.NO_OP.copy()

    @staticmethod
    def forward() -> ActionV1:
        return ActionSpaceV1.FORWARD.copy()

    @staticmethod
    def backward() -> ActionV1:
        return ActionSpaceV1.BACKWARD.copy()

    @staticmethod
    def left_strafe() -> ActionV1:
        return ActionSpaceV1.LEFT_STRAFE.copy()

    @staticmethod
    def right_strafe() -> ActionV1:
        return ActionSpaceV1.RIGHT_STRAFE.copy()

    @staticmethod
    def jump() -> ActionV1:
        return ActionSpaceV1.JUMP.copy()

    @staticmethod
    def turn_left() -> ActionV1:
        return ActionSpaceV1.TURN_LEFT.copy()

    @staticmethod
    def turn_right() -> ActionV1:
        return ActionSpaceV1.TURN_RIGHT.copy()

    @staticmethod
    def look_up() -> ActionV1:
        return ActionSpaceV1.LOOK_UP.copy()

    @staticmethod
    def look_down() -> ActionV1:
        return ActionSpaceV1.LOOK_DOWN.copy()

    @staticmethod
    def use() -> ActionV1:
        return ActionSpaceV1.USE.copy()

    @staticmethod
    def drop() -> ActionV1:
        return ActionSpaceV1.DROP.copy()

    @staticmethod
    def attack() -> ActionV1:
        return ActionSpaceV1.ATTACK.copy()


class ActionSpaceV2(gym.spaces.Dict):
    NO_OP: Dict[str, Union[bool, float]] = {
        "attack": False,
        "back": False,
        "forward": False,
        "jump": False,
        "left": False,
        "right": False,
        "sneak": False,
        "sprint": False,
        "use": False,
        "drop": False,
        "inventory": False,
        "hotbar.1": False,
        "hotbar.2": False,
        "hotbar.3": False,
        "hotbar.4": False,
        "hotbar.5": False,
        "hotbar.6": False,
        "hotbar.7": False,
        "hotbar.8": False,
        "hotbar.9": False,
        "camera": np.array([0.0, 0.0], dtype=np.float32),
    }

    def __init__(self):
        super().__init__(
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
            }
        )

    def __repr__(self) -> str:
        return r"ActionSpaceV2(attack, back, forward, jump, left, right, sneak, sprint, use, drop, inventory, hotbar.1-hotbar.9, camera {pitch:[-180, 180], yaw:[-180, 180]})"

    @staticmethod
    def no_op() -> Dict[str, Union[bool, float]]:
        return ActionSpaceV2.NO_OP.copy()


def no_op_v1() -> ActionV1:
    return ActionSpaceV1.no_op()


no_op = no_op_v1


def no_op_v2() -> Dict[str, Union[bool, float]]:
    return ActionSpaceV2.no_op()


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
    action_v2: Dict[str, Union[bool, float]],
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
    if "camera_pitch" in action_v2:
        action_space.camera_pitch = action_v2["camera_pitch"]
        action_space.camera_yaw = action_v2["camera_yaw"]
    elif "camera" in action_v2:
        action_space.camera_pitch = action_v2["camera"][0]
        action_space.camera_yaw = action_v2["camera"][1]
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


def declare_action_space(
    action_space_version: ActionSpaceVersion,
) -> Union[ActionSpaceV1, ActionSpaceV2]:
    if action_space_version == ActionSpaceVersion.V1_MINEDOJO:
        # Same as the action space used in MineDojo
        return ActionSpaceV1()
    elif action_space_version == ActionSpaceVersion.V2_MINERL_HUMAN:
        return ActionSpaceV2()
    else:
        raise ValueError(f"Unknown action space version: {action_space_version}")
