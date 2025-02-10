import gzip
import struct
from typing import Tuple

import numpy as np
from .nbt_dataclass import (
    NBTBase,
    NBTByte,
    NBTByteArray,
    NBTCompound,
    NBTDouble,
    NBTFloat,
    NBTInt,
    NBTIntArray,
    NBTList,
    NBTLong,
    NBTLongArray,
    NBTShort,
    NBTString,
    TagType,
)


def read_nbt(file_path: str) -> NBTCompound:
    """Reads an NBT file and returns an NBT object."""
    with gzip.open(file_path, "rb") as f:
        entry = _read_named_tag(f)
        if entry is None:
            raise ValueError("Empty NBT file")
        name, content = entry
        return NBTCompound({name: content})


def _read_named_tag(f) -> Tuple[str, NBTBase]:
    """Reads a Named Tag and returns an NBT object."""
    tag_type = struct.unpack(">B", f.read(1))[0]
    if tag_type == TagType.EndType.value:
        return None

    name = _read_string(f)
    contents = _read_payload(f, TagType(tag_type))
    return (name, contents)


def _read_payload(f, tag_type: TagType) -> NBTBase:
    """Reads the payload of an NBT tag."""
    if tag_type == TagType.ByteType:
        return NBTByte(struct.unpack(">b", f.read(1))[0])
    elif tag_type == TagType.ShortType:
        return NBTShort(struct.unpack(">h", f.read(2))[0])
    elif tag_type == TagType.IntType:
        return NBTInt(struct.unpack(">i", f.read(4))[0])
    elif tag_type == TagType.LongType:
        return NBTLong(struct.unpack(">q", f.read(8))[0])
    elif tag_type == TagType.FloatType:
        return NBTFloat(struct.unpack(">f", f.read(4))[0])
    elif tag_type == TagType.DoubleType:
        return NBTDouble(struct.unpack(">d", f.read(8))[0])
    elif tag_type == TagType.StringType:
        return NBTString(_read_string(f))
    elif tag_type == TagType.ByteArrayType:
        return NBTByteArray(_read_array(f, np.int8).tolist())
    elif tag_type == TagType.IntArrayType:
        return NBTIntArray(_read_array(f, np.int32).tolist())
    elif tag_type == TagType.LongArrayType:
        return NBTLongArray(_read_array(f, np.int64).tolist())
    elif tag_type == TagType.ListType:
        return _read_list(f)
    elif tag_type == TagType.CompoundType:
        return _read_compound(f)
    else:
        raise ValueError(f"Unknown tag type: {tag_type}")


def _read_string(f) -> str:
    """Reads a TAG_String from the file."""
    length = struct.unpack(">h", f.read(2))[0]
    return f.read(length).decode("utf-8")


def _read_array(f, dtype) -> np.ndarray:
    """Reads an NBT array (ByteArray, IntArray, LongArray)."""
    length = struct.unpack(">i", f.read(4))[0]
    return np.frombuffer(f.read(length * np.dtype(dtype).itemsize), dtype=dtype)


def _read_list(f) -> NBTList:
    """Reads a TAG_List and returns it as an NbtContents object."""
    tag_type = struct.unpack(">B", f.read(1))[0]
    length = struct.unpack(">i", f.read(4))[0]
    elements = [_read_payload(f, TagType(tag_type)) for _ in range(length)]
    return NBTList(elements)


def _read_compound(f) -> NBTCompound:
    """Reads a TAG_Compound and returns it as an NbtContents object."""
    result = {}
    while True:
        entry = _read_named_tag(f)
        if entry is None:  # TAG_End encountered
            break
        name, content = entry
        result[name] = content
    return NBTCompound(result)


def write_nbt(nbt: NBTCompound, file_path: str):
    """Writes an NBT object to a file."""
    with gzip.open(file_path, "wb") as f:
        nbt.write_to_file(f)
