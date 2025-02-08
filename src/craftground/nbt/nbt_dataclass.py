from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Generic, List, Dict, Type, TypeVar, Union
from nbt.models.dict_nbt import CompoundNBT
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


U = TypeVar("U", bound="NBTSerializable")


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

    @classmethod
    def from_nbt(cls: Type[U], nbt: NBT) -> U:
        """Convert an NBT object to an NBTSerializable dataclass."""
        if (
            not isinstance(nbt.contents, NbtContents)
            or nbt.contents.tag_type != TagType.CompoundType
        ):
            raise ValueError(f"Expected CompoundType NBT, got {nbt.contents.tag_type}")

        # Extract raw NBT dictionary
        nbt_dict = {field.name: field for field in fields(cls)}
        parsed_values = {}

        for key, value in nbt.contents.value.items():
            if key not in nbt_dict:
                continue  # Skip unknown keys

            field_type = nbt_dict[key].type

            if is_dataclass(field_type) and issubclass(field_type, NBTSerializable):
                # Recursively parse nested NBTSerializable classes
                parsed_values[key] = field_type.from_nbt(NBT(key, value))

            elif field_type == NBTInt and value.tag_type == TagType.IntType:
                parsed_values[key] = NBTInt(value.value)

            elif field_type == NBTString and value.tag_type == TagType.StringType:
                parsed_values[key] = NBTString(value.value)

            elif field_type == NBTList and value.tag_type == TagType.ListType:
                parsed_values[key] = NBTList(
                    [NBTSerializable._parse_nbt_list_item(v) for v in value.value]
                )

            elif field_type == CompoundNBT and value.tag_type == TagType.CompoundType:
                parsed_values[key] = CompoundNBT(value.value)

            else:
                raise TypeError(f"Unsupported NBT field type: {field_type}")

        return cls(**parsed_values)

    @staticmethod
    def _parse_nbt_list_item(value: NbtContents):
        """Helper function to parse NBT list items correctly."""
        if value.tag_type == TagType.IntType:
            return NBTInt(value.value)
        elif value.tag_type == TagType.DoubleType:
            return NBTDouble(value.value)
        elif value.tag_type == TagType.FloatType:
            return NBTFloat(value.value)
        elif value.tag_type == TagType.LongType:
            return NBTLong(value.value)
        elif value.tag_type == TagType.ShortType:
            return NBTShort(value.value)
        elif value.tag_type == TagType.ByteType:
            return NBTByte(value.value)
        elif value.tag_type == TagType.ByteArrayType:
            return NBTByteArray(value.value)
        elif value.tag_type == TagType.IntArrayType:
            return NBTIntArrayType(value.value)
        elif value.tag_type == TagType.LongArrayType:
            return NBTLongArrayType(value.value)
        elif value.tag_type == TagType.StringType:
            return NBTString(value.value)
        elif value.tag_type == TagType.CompoundType:
            return CompoundNBT(value.value)
        else:
            return value  # Return raw if no explicit mapping exists
