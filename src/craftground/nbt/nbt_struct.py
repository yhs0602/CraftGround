# https://github.com/acfoltzer/nbt/blob/master/NBT-spec.txt
# https://minecraft.wiki/w/NBT_format
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Union


class TagType(int, Enum):
    EndType = 0
    ByteType = 1
    ShortType = 2
    IntType = 3
    LongType = 4
    FloatType = 5
    DoubleType = 6
    ByteArrayType = 7
    StringType = 8
    ListType = 9
    CompoundType = 10
    IntArrayType = 11
    LongArrayType = 12


@dataclass
class NBT:
    """NBT Data Structure"""

    name: str
    contents: "NbtContents"

    def dump_to_dict(self, d: dict) -> dict:
        """Add NBT to dictionary"""
        d[self.name] = self.contents.dump()
        return d

    def __iter__(self):
        yield self.name
        yield self.contents


@dataclass
class NbtContents:
    """NBT Data Structure"""

    tag_type: TagType
    value: Union[
        int,
        float,
        str,
        List[int],
        List[float],
        List[str],
        List["NbtContents"],  # ListTag
        List[NBT],  # CompoundTag
    ]

    def dump(self):
        if self.tag_type == TagType.CompoundType:
            return {nbt.name: nbt.contents.dump() for nbt in self.value}
        elif self.tag_type == TagType.ListType:
            return [nbt_contents.dump() for nbt_contents in self.value]
        else:
            return self.value

    @staticmethod
    def int(value: int) -> "NbtContents":
        return NbtContents(TagType.IntType, value)

    @staticmethod
    def float(value: float) -> "NbtContents":
        return NbtContents(TagType.FloatType, value)

    @staticmethod
    def byte(value: int) -> "NbtContents":
        return NbtContents(TagType.ByteType, value)

    @staticmethod
    def short(value: int) -> "NbtContents":
        return NbtContents(TagType.ShortType, value)

    @staticmethod
    def long(value: int) -> "NbtContents":
        return NbtContents(TagType.LongType, value)

    @staticmethod
    def double(value: float) -> "NbtContents":
        return NbtContents(TagType.DoubleType, value)

    @staticmethod
    def bytearray(value: List[int]) -> "NbtContents":
        return NbtContents(TagType.ByteArrayType, value)

    @staticmethod
    def string(value: str) -> "NbtContents":
        return NbtContents(TagType.StringType, value)

    @staticmethod
    def list(value: List["NbtContents"]) -> "NbtContents":
        return NbtContents(TagType.ListType, value)

    @staticmethod
    def compound(value: dict) -> "NbtContents":
        return NbtContents(
            TagType.CompoundType,
            [NBT(name, NbtContents.any(value)) for name, value in value.items()],
        )

    @staticmethod
    def intarray(value: List[int]) -> "NbtContents":
        return NbtContents(TagType.IntArrayType, value)

    @staticmethod
    def longarray(value: List[int]) -> "NbtContents":
        return NbtContents(TagType.LongArrayType, value)

    @staticmethod
    def any(value: Any) -> "NbtContents":
        if isinstance(value, int):
            return NbtContents.int(value)
        elif isinstance(value, float):
            return NbtContents.float(value)
        elif isinstance(value, str):
            return NbtContents.string(value)
        elif isinstance(value, list):
            return NbtContents.list(value)
        elif isinstance(value, dict):
            return NbtContents.compound(value)
        else:
            raise ValueError(f"Unknown type: {type(value)}")
