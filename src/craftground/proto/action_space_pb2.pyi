from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ActionSpaceMessageV2(_message.Message):
    __slots__ = ("attack", "back", "forward", "jump", "left", "right", "sneak", "sprint", "use", "drop", "inventory", "hotbar_1", "hotbar_2", "hotbar_3", "hotbar_4", "hotbar_5", "hotbar_6", "hotbar_7", "hotbar_8", "hotbar_9", "camera_pitch", "camera_yaw", "commands")
    ATTACK_FIELD_NUMBER: _ClassVar[int]
    BACK_FIELD_NUMBER: _ClassVar[int]
    FORWARD_FIELD_NUMBER: _ClassVar[int]
    JUMP_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    SNEAK_FIELD_NUMBER: _ClassVar[int]
    SPRINT_FIELD_NUMBER: _ClassVar[int]
    USE_FIELD_NUMBER: _ClassVar[int]
    DROP_FIELD_NUMBER: _ClassVar[int]
    INVENTORY_FIELD_NUMBER: _ClassVar[int]
    HOTBAR_1_FIELD_NUMBER: _ClassVar[int]
    HOTBAR_2_FIELD_NUMBER: _ClassVar[int]
    HOTBAR_3_FIELD_NUMBER: _ClassVar[int]
    HOTBAR_4_FIELD_NUMBER: _ClassVar[int]
    HOTBAR_5_FIELD_NUMBER: _ClassVar[int]
    HOTBAR_6_FIELD_NUMBER: _ClassVar[int]
    HOTBAR_7_FIELD_NUMBER: _ClassVar[int]
    HOTBAR_8_FIELD_NUMBER: _ClassVar[int]
    HOTBAR_9_FIELD_NUMBER: _ClassVar[int]
    CAMERA_PITCH_FIELD_NUMBER: _ClassVar[int]
    CAMERA_YAW_FIELD_NUMBER: _ClassVar[int]
    COMMANDS_FIELD_NUMBER: _ClassVar[int]
    attack: bool
    back: bool
    forward: bool
    jump: bool
    left: bool
    right: bool
    sneak: bool
    sprint: bool
    use: bool
    drop: bool
    inventory: bool
    hotbar_1: bool
    hotbar_2: bool
    hotbar_3: bool
    hotbar_4: bool
    hotbar_5: bool
    hotbar_6: bool
    hotbar_7: bool
    hotbar_8: bool
    hotbar_9: bool
    camera_pitch: float
    camera_yaw: float
    commands: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, attack: bool = ..., back: bool = ..., forward: bool = ..., jump: bool = ..., left: bool = ..., right: bool = ..., sneak: bool = ..., sprint: bool = ..., use: bool = ..., drop: bool = ..., inventory: bool = ..., hotbar_1: bool = ..., hotbar_2: bool = ..., hotbar_3: bool = ..., hotbar_4: bool = ..., hotbar_5: bool = ..., hotbar_6: bool = ..., hotbar_7: bool = ..., hotbar_8: bool = ..., hotbar_9: bool = ..., camera_pitch: _Optional[float] = ..., camera_yaw: _Optional[float] = ..., commands: _Optional[_Iterable[str]] = ...) -> None: ...
