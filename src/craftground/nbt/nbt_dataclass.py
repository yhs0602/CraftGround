# https://github.com/acfoltzer/nbt/blob/master/NBT-spec.txt
# https://minecraft.wiki/w/NBT_format

from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from pydoc import locate
import struct
import sys
from typing import Any, Generic, List, Dict, Type, TypeVar, Union
import typing
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Union


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
    type: TagType

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    def to_snbt(self, indent: int, depth: int) -> str:
        raise NotImplementedError

    def write_to_file(self, f):
        raise NotImplementedError


class NBTByte(NBTBase):
    def __init__(self, value: int):
        self.type = TagType.ByteType
        self.value = value

    def write_to_file(self, f):
        f.write(struct.pack(">b", self.value))

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}b"


class NBTShort(NBTBase):
    def __init__(self, value: int):
        self.type = TagType.ShortType
        self.value = value

    def write_to_file(self, f):
        f.write(struct.pack(">h", self.value))

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}s"


class NBTInt(NBTBase):
    def __init__(self, value: int):
        self.type = TagType.IntType
        self.value = value

    def write_to_file(self, f):
        f.write(struct.pack(">i", self.value))

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}"


class NBTLong(NBTBase):
    def __init__(self, value: int):
        self.type = TagType.LongType
        self.value = value

    def write_to_file(self, f):
        f.write(struct.pack(">q", self.value))

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}l"


class NBTFloat(NBTBase):
    def __init__(self, value: float):
        self.type = TagType.FloatType
        self.value = value

    def write_to_file(self, f):
        f.write(struct.pack(">f", self.value))

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}f"


class NBTDouble(NBTBase):
    def __init__(self, value: float):
        self.type = TagType.DoubleType
        self.value = value

    def write_to_file(self, f):
        f.write(struct.pack(">d", self.value))

    def to_snbt(self, indent=0, depth=0):
        return f"{self.value}d"


class NBTByteArray(NBTBase):
    def __init__(self, value: List[int]):
        self.type = TagType.ByteArrayType
        self.value = value

    def write_to_file(self, f):
        f.write(struct.pack(">i", len(self.value)))
        # TODO
        f.write(self.value.tobytes())

    def to_snbt(self, indent=0, depth=0):
        return f"[B;{','.join(f'{v}b' for v in self.value)}]"


class NBTString(NBTBase):
    def __init__(self, value: str):
        assert isinstance(value, str)
        self.type = TagType.StringType
        self.value = value

    def write_to_file(self, f):
        data = self.value.encode("utf-8")
        f.write(struct.pack(">h", len(data)))
        f.write(data)

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

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, NBTString):
            return False
        return self.value == value.value


class NBTIntArray(NBTBase):
    def __init__(self, value: List[int]):
        self.type = TagType.IntArrayType
        self.value = value

    def write_to_file(self, f):
        f.write(struct.pack(">i", len(self.value)))
        # TODO
        f.write(self.value.tobytes())

    def to_snbt(self, indent=0, depth=0):
        return f"[I;{','.join(str(v) for v in self.value)}]"


class NBTLongArray(NBTBase):
    def __init__(self, value: List[int]):
        self.type = TagType.LongArrayType
        self.value = value

    def write_to_file(self, f):
        f.write(struct.pack(">i", len(self.value)))
        # TODO
        f.write(self.value.tobytes())

    def to_snbt(self, indent=0, depth=0):
        return f"[L;{','.join(str(v) for v in self.value)}]"


# Recursive NBTList and NBTCompound
class NBTList(NBTBase, Generic[T], List[T]):
    def __init__(self, value: List[T]):
        self.type = TagType.ListType
        self.value = value
        super().__init__(value)

    def write_to_file(self, f):
        """Writes a TAG_List to the file."""
        if not self.value:
            f.write(struct.pack(">B", TagType.EndType.value))
            f.write(struct.pack(">i", 0))
            return

        tag_type = self.value[0].type
        f.write(struct.pack(">B", tag_type.value))
        f.write(struct.pack(">i", len(self.value)))

        for item in self.value:
            item.write_to_file(f)

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
        self.type = TagType.CompoundType
        self.value: Dict[str, NBTBase] = value
        super().__init__(value)

    def __getattr__(self, name: str) -> NBTBase:
        """Allow accessing NBT values as properties."""
        try:
            return self.value[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )

    def __hash__(self):
        """Make NBTCompound hashable if all values are hashable."""
        return hash(tuple(sorted((k, hash(v)) for k, v in self.value.items())))

    def write_to_file(self, f):
        """Writes a TAG_Compound to the file."""
        for name, value in self.value.items():
            if value is None:
                continue
            f.write(struct.pack(">B", value.type.value))
            name_data = name.encode("utf-8")
            f.write(struct.pack(">h", len(name_data)))
            f.write(name_data)
            value.write_to_file(f)

        f.write(struct.pack(">B", TagType.EndType.value))

    def to_snbt(self, indent=0, depth=0):
        if indent:
            items = [
                f"{k}:{v.to_snbt(indent=indent, depth=depth + 1)}"
                for k, v in self.value.items()
            ]
            inner_indent = " " * (indent * (depth + 1))
            return (
                "{\n"
                + ",\n".join(inner_indent + item for item in items)
                + f"\n{' ' * (indent * depth)}}}"
            )
        items = [f"{k}:{v.to_snbt()}" for k, v in self.value.items()]
        return "{" + ",".join(items) + "}"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, NBTCompound):
            return False
        return self.value == value.value


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


U = TypeVar("U", bound="NBTSerializable")


def str_to_class(classname: str) -> typing.Type:
    if classname.startswith("Optional["):
        inner_type = classname[len("Optional[") : -1]
        return typing.Optional[str_to_class(inner_type)]

    if "[" in classname and "]" in classname:  # Handle generics like NBTList[NBTInt]
        base_type, inner_types = classname.split("[", 1)
        inner_types = inner_types.rstrip("]")  # Remove closing bracket

        # Resolve the base type
        base_cls = locate(base_type) or getattr(sys.modules[__name__], base_type, None)
        if not base_cls:
            return None  # Unknown base type

        # Resolve inner types recursively
        inner_cls = tuple(
            str_to_class(inner.strip()) for inner in inner_types.split(",")
        )

        # Construct generic type
        return base_cls[inner_cls[0] if len(inner_cls) == 1 else inner_cls]

    return locate(classname) or getattr(sys.modules[__name__], classname, None)


def get_type_hints_with_locate(obj):
    """Enhances get_type_hints by using locate to find types that haven't been imported yet"""
    globalns = sys.modules[
        obj.__module__
    ].__dict__.copy()  # Copy the namespace to avoid modifying the original
    localns = {}

    # __annotations__에서 타입 문자열을 가져오고 locate로 찾아서 globalns에 추가
    for name, type_str in getattr(obj, "__annotations__", {}).items():
        if isinstance(type_str, str) and type_str not in globalns:
            resolved = locate(type_str)
            if resolved:
                globalns[type_str] = resolved  # locate로 찾은 타입 추가

    return typing.get_type_hints(obj, globalns, localns)


@dataclass
class NBTSerializable:
    """NBT Serializing Dataclass with SNBT and pretty-print support"""

    type = TagType.CompoundType

    @classmethod
    def from_nbt(cls: Type[U], nbt: NBTCompound) -> U:
        """Convert an NBT object to an NBTSerializable dataclass."""
        inner_nbt = nbt.value[""]
        return cls(**inner_nbt.value)

    def to_snbt(self, indent: int = 0, depth: int = 0) -> str:
        """Convert NBTSerializable to SNBT string format with optional indentation"""
        snbt_fields = []
        for field in fields(self):
            value = object.__getattribute__(self, field.name)
            if value is not None:
                snbt_fields.append(f"{field.name}:{value.to_snbt(indent, depth + 1)}")

        if indent:
            inner_indent = " " * (indent * (depth + 1))
            return (
                "{\n"
                + ",\n".join(inner_indent + field for field in snbt_fields)
                + f"\n{' ' * (indent * depth)}}}"
            )
        return "{" + ",".join(snbt_fields) + "}"

    def write_to_file(self, f):
        """Writes the NBTSerializable to a file."""
        # Make nbt compound
        nbt = NBTCompound(
            {field.name: getattr(self, field.name) for field in fields(self)}
        )
        nbt.write_to_file(f)

    def __setattr__(self, name: str, value: Any) -> None:
        if not isinstance(value, NBTBase):
            # print(f"Expected NBTBase, got {type(value).__name__}")
            if value is None:
                return  # Ignore None values
            # Get the original type and convert the value to corresponding NBTBase
            type_hints = get_type_hints_with_locate(self.__class__)
            field_type = type_hints.get(name)
            # print(f"Field type for {name}: {field_type}, {type_hints=}")
            if not field_type:
                raise ValueError(f"Unknown field type for {name}")
            value = NBTSerializable.convert_to_nbtbase(value, field_type)
            # print(f"Converted {type(value).__name__} to {field_type}")  # : {value}

        # Set the attribute
        object.__setattr__(self, name, value)

    @staticmethod
    def convert_to_nbtbase(value: Any, field_type: Type) -> NBTBase:
        """Converts a value to an NBTBase object."""
        # print(f"Converting {value} to {field_type}")
        # Resolve generic types
        origin_type = typing.get_origin(field_type)
        if origin_type == NBTList:
            return NBTList(
                [
                    NBTSerializable.convert_to_nbtbase(
                        item, typing.get_args(field_type)[0]
                    )
                    for item in value
                ]
            )
        if origin_type == NBTCompound:
            return NBTCompound(value)
        if isinstance(value, field_type):
            return value
        if is_dataclass(field_type):
            # print(f"Converting {value} to {field_type}")
            return field_type(value)
        # print(f"Converting {value} to {field_type}")
        return field_type(value)
