from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from pydoc import locate
import sys
from typing import Any, Generic, List, Dict, Type, TypeVar, Union
import typing
from nbt_struct import NBT, NbtContents, TagType

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
        "NBTIntArray",
        "NBTLongArray",
    ],
)


# Basic NBT Type support Wrappers
class NBTBase:
    def to_nbt_contents(self) -> NbtContents:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    def to_snbt(self, indent: int, depth: int) -> str:
        raise NotImplementedError


class NBTByte(NBTBase):
    def __init__(self, value: bool):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.ByteType, self.value)

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}b"


class NBTShort(NBTBase):
    def __init__(self, value: int):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.ShortType, self.value)

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}s"


class NBTInt(NBTBase):
    def __init__(self, value: int):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.IntType, self.value)

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}"


class NBTLong(NBTBase):
    def __init__(self, value: int):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.LongType, self.value)

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}l"


class NBTFloat(NBTBase):
    def __init__(self, value: float):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.FloatType, self.value)

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}f"


class NBTDouble(NBTBase):
    def __init__(self, value: float):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.DoubleType, self.value)

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}d"


class NBTByteArray(NBTBase):
    def __init__(self, value: List[int]):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.ByteArrayType, self.value)

    def to_snbt(self, indent=0, depth=0):
        return f"[B;{','.join(f'{v}b' for v in self.value)}]"


class NBTString(NBTBase):
    def __init__(self, value: str):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.StringType, self.value)

    def to_snbt(self, indent=0, depth=0):
        """
                A string enclosed in quotes. For strings containing only 0-9, A-Z, a-z, _, -, ., and +, and not confused with other data types, quote enclosure is optional. Quotes can be either single quote ' or double ". Nested quotes can be included within a string by escaping the character with a \ escape. \s that are supposed to show up as \ need to be escaped to \\.
        <[a-zA-Z0-9_\-\.\+] text>, "<text>" (" within needs to be escaped to \"), or '<text>' (' within needs to be escaped to \')
        """
        # Escape quotes and backslashes
        escaped = (
            self.value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
        )
        return f'"{escaped}"'


class NBTIntArray(NBTBase):
    def __init__(self, value: List[int]):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.IntArrayType, self.value)

    def to_snbt(self, indent=0, depth=0):
        return f"[I;{','.join(str(v) for v in self.value)}]"


class NBTLongArray(NBTBase):
    def __init__(self, value: List[int]):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.LongArrayType, self.value)

    def to_snbt(self, indent=0, depth=0):
        return f"[L;{','.join(str(v) for v in self.value)}]"


# Recursive NBTList and NBTCompound
class NBTList(NBTBase, Generic[T]):
    def __init__(self, value: List[T]):
        self.value = value

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(TagType.ListType, [v.to_nbt_contents() for v in self.value])

    def to_snbt(self, indent=0, depth=0):
        if indent:
            items = [v.to_snbt(indent=indent, depth=depth + 1) for v in self.value]
            inner_indent = " " * (indent * (depth + 1))
            return (
                "[\n"
                + ",\n".join(inner_indent + v for v in items)
                + f"\n{' ' * (indent * depth)}]"
            )
        items = [v.to_snbt() for v in self.value]
        return "[" + ",".join(items) + "]"


class NBTCompound(NBTBase, Dict[str, NBTBase]):
    def __init__(self, value: Dict[str, NBTBase]):
        self.value = value

    def to_nbt_contents(self, indent=0, depth=0) -> NbtContents:
        return NbtContents(
            TagType.CompoundType,
            {k: v.to_nbt_contents() for k, v in self.value.items()},
        )

    def to_snbt(self, indent=0, depth=0):
        if indent:
            items = [
                f"{k}:{v.to_snbt(indent=indent, depth=depth + 1)}"
                for k, v in self.items()
            ]
            inner_indent = " " * (indent * (depth + 1))
            return (
                "{\n"
                + ",\n".join(inner_indent + item for item in items)
                + f"\n{' ' * (indent * depth)}}}"
            )
        items = [f"{k}:{v.to_snbt()}" for k, v in self.value.items()]
        return "{" + ",".join(items) + "}"


U = TypeVar("U", bound="NBTSerializable")


def str_to_class(classname: str) -> Type:
    if classname.startswith("Optional["):
        inner_type = classname[len("Optional[") : -1]  # 괄호 안의 타입 추출
        return typing.Optional[str_to_class(inner_type)]
    if hasattr(typing, classname):
        return getattr(typing, classname)
    return None  # None existing class


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
        expected_type = str_to_class(expected_type) if expected_type else None

        if value is None:  # Explicitly allow None
            object.__setattr__(self, name, None)
            return
        # print(f"Setattr: {name=} {value=} {expected_type=} {field_types=}")
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

        for key, value in nbt.contents.value:
            if key not in nbt_dict:
                continue  # Skip unknown keys

            field_type = nbt_dict[key].type

            if is_dataclass(field_type) and issubclass(field_type, NBTSerializable):
                # Recursively parse nested NBTSerializable classes
                parsed_values[key] = field_type.from_nbt(NBT(key, value))

            elif value.tag_type == TagType.ListType:
                parsed_values[key] = NBTList(
                    [NBTSerializable._parse_nbt_list_item(v) for v in value.value]
                )
            elif field_type == NBTCompound and value.tag_type == TagType.CompoundType:
                parsed_values[key] = NBTCompound(value.value)
            else:
                parsed_values[key] = NBTSerializable._parse_nbt_list_item(value)

        return cls(**parsed_values)

    @staticmethod
    def _parse_nbt_list_item(value: NbtContents) -> NBTBase:
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
            return NBTIntArray(value.value)
        elif value.tag_type == TagType.LongArrayType:
            return NBTLongArray(value.value)
        elif value.tag_type == TagType.StringType:
            return NBTString(value.value)
        elif value.tag_type == TagType.CompoundType:
            return NBTCompound(value.value)
        else:
            return value  # Return raw if no explicit mapping exists

    def to_snbt(self, indent: int = 0, depth_outer: int = 0) -> str:
        # Convert dataclass fields into SNBT format
        snbt_fields = []
        for field in fields(self):
            value = object.__getattribute__(self, field.name)
            if value is not None:
                if indent:
                    snbt_fields.append(
                        f"{field.name}:{value.to_snbt(indent, depth_outer + 1)}"
                    )
                else:
                    snbt_fields.append(f"{field.name}:{value.to_snbt()}")

        if indent:
            inner_indent = " " * (indent * (depth_outer + 1))
            return (
                "{\n"
                + ",\n".join(inner_indent + field for field in snbt_fields)
                + f"\n{' ' * (indent * depth_outer)}}}"
            )
        return "{" + ",".join(snbt_fields) + "}"
