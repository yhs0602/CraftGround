from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any, Generic, List, Dict, TypeVar, Union
from nbt.nbt_struct import NBT, NbtContents, TagType

# TypeVar definition
T = TypeVar(
    "T",
    bound=Union[
        "NBTByte",
        "NBTShort",
        "NBTInt",
        "NBTLong",
        "NBTFloat",
        "NBTDouble",
        "NBTByteArray",
        "NBTString",
        "NBTList[Any]",
        "NBTCompound[Any]",
        "NBTIntArrayType",
        "NBTLongArrayType",
    ],
)


# Basic NBT Type support Wrappers
class NBTBase:
    def to_nbt_contents(self) -> NbtContents:
        raise NotImplementedError


class NBTByte(NBTBase):
    def __init__(self, value: bool):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.ByteType, self.value)


class NBTShort(NBTBase):
    def __init__(self, value: int):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.ShortType, self.value)


class NBTInt(NBTBase):
    def __init__(self, value: int):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.IntType, self.value)


class NBTLong(NBTBase):
    def __init__(self, value: int):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.LongType, self.value)


class NBTFloat(NBTBase):
    def __init__(self, value: float):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.FloatType, self.value)


class NBTDouble(NBTBase):
    def __init__(self, value: float):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.DoubleType, self.value)


class NBTByteArray(NBTBase):
    def __init__(self, value: List[int]):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.ByteArrayType, self.value)


class NBTString(NBTBase):
    def __init__(self, value: str):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.StringType, self.value)


class NBTIntArrayType(NBTBase):
    def __init__(self, value: List[int]):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.IntArrayType, self.value)


class NBTLongArrayType(NBTBase):
    def __init__(self, value: List[int]):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.LongArrayType, self.value)


# Recursive NBTList and NBTCompound
class NBTList(NBTBase, Generic[T]):
    def __init__(self, value: List[T]):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.ListType, [v.to_nbt_contents() for v in self.value])


class NBTCompound(NBTBase, Dict[str, NBTBase]):
    def __init__(self, value: Dict[str, NBTBase]):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(
            TagType.CompoundType,
            {k: v.to_nbt_contents() for k, v in self.value.items()},
        )


@dataclass
class NBTSerializable:
    """NBT Serializing Dataclass"""

    def __getattribute__(self, name):
        """Getter - Automatically unwrap NBTBase"""
        value = object.__getattribute__(self, name)
        if isinstance(value, NBTBase):
            return value.value
        return value

    def __setattr__(self, name, value):
        """Setter - Automatically wrap NBTBase, handle None safely"""
        field_types = {field.name: field.type for field in fields(self)}
        expected_type = field_types.get(name, None)

        if value is None:  # Explicitly allow None
            object.__setattr__(self, name, None)
            return

        if expected_type and issubclass(expected_type, NBTBase):
            object.__setattr__(self, name, expected_type(value))
        else:
            object.__setattr__(self, name, value)

    def to_nbt_contents(self) -> NbtContents:
        """Automatically convert to NbtContents, ignoring None values"""
        compound_list = []
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue  # Skip None fields
            if isinstance(value, NBTBase):
                compound_list.append(NBT(field.name, value.to_nbt_contents()))
        return NbtContents(TagType.CompoundType, compound_list)
