import gzip
import struct
from typing import List

import numpy as np
from nbt_struct import NBT, NbtContents, TagType


def read_nbt(file_path: str) -> NBT:
    """Reads an NBT file and returns an NBT object."""
    with gzip.open(file_path, "rb") as f:
        return _read_named_tag(f)


def _read_named_tag(f) -> NBT:
    """Reads a Named Tag and returns an NBT object."""
    tag_type = struct.unpack(">B", f.read(1))[0]
    if tag_type == TagType.EndType.value:
        return None

    name = _read_string(f)
    contents = _read_payload(f, TagType(tag_type))

    return NBT(name, contents)


def _read_payload(f, tag_type: TagType) -> NbtContents:
    """Reads the payload of an NBT tag."""
    if tag_type == TagType.ByteType:
        return NbtContents(tag_type, struct.unpack(">b", f.read(1))[0])
    elif tag_type == TagType.ShortType:
        return NbtContents(tag_type, struct.unpack(">h", f.read(2))[0])
    elif tag_type == TagType.IntType:
        return NbtContents(tag_type, struct.unpack(">i", f.read(4))[0])
    elif tag_type == TagType.LongType:
        return NbtContents(tag_type, struct.unpack(">q", f.read(8))[0])
    elif tag_type == TagType.FloatType:
        return NbtContents(tag_type, struct.unpack(">f", f.read(4))[0])
    elif tag_type == TagType.DoubleType:
        return NbtContents(tag_type, struct.unpack(">d", f.read(8))[0])
    elif tag_type == TagType.StringType:
        return NbtContents(tag_type, _read_string(f))
    elif tag_type == TagType.ByteArrayType:
        return NbtContents(tag_type, _read_array(f, np.int8).tolist())
    elif tag_type == TagType.IntArrayType:
        return NbtContents(tag_type, _read_array(f, np.int32).tolist())
    elif tag_type == TagType.LongArrayType:
        return NbtContents(tag_type, _read_array(f, np.int64).tolist())
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


def _read_list(f) -> NbtContents:
    """Reads a TAG_List and returns it as an NbtContents object."""
    tag_type = struct.unpack(">B", f.read(1))[0]
    length = struct.unpack(">i", f.read(4))[0]
    elements = [_read_payload(f, TagType(tag_type)) for _ in range(length)]
    return NbtContents(TagType.ListType, elements)


def _read_compound(f) -> NbtContents:
    """Reads a TAG_Compound and returns it as an NbtContents object."""
    result = []
    while True:
        entry = _read_named_tag(f)
        if entry is None:  # TAG_End encountered
            break
        result.append(entry)
    return NbtContents(TagType.CompoundType, result)


def write_nbt(nbt: NBT, file_path: str):
    """Writes an NBT object to a file."""
    with gzip.open(file_path, "wb") as f:
        _write_named_tag(f, nbt)
        f.write(struct.pack(">B", TagType.EndType.value))  # End tag


def _write_named_tag(f, nbt: NBT):
    """Writes a Named Tag to the file."""
    f.write(struct.pack(">B", nbt.contents.tag_type.value))
    _write_string(f, nbt.name)
    _write_payload(f, nbt.contents)


def _write_payload(f, contents: NbtContents):
    """Writes the payload of an NBT tag."""
    if contents.tag_type in {
        TagType.ByteType,
        TagType.ShortType,
        TagType.IntType,
        TagType.LongType,
        TagType.FloatType,
        TagType.DoubleType,
    }:
        f.write(struct.pack(_get_struct_format(contents.tag_type), contents.value))
    elif contents.tag_type == TagType.StringType:
        _write_string(f, contents.value)
    elif contents.tag_type in {
        TagType.ByteArrayType,
        TagType.IntArrayType,
        TagType.LongArrayType,
    }:
        _write_array(f, contents.value)
    elif contents.tag_type == TagType.ListType:
        _write_list(f, contents.value)
    elif contents.tag_type == TagType.CompoundType:
        for tag in contents.value:
            _write_named_tag(f, tag)
        f.write(struct.pack(">B", TagType.EndType.value))  # End tag
    else:
        raise ValueError(f"Unknown tag type: {contents.tag_type}")


def _write_string(f, value: str):
    """Writes a TAG_String to the file."""
    encoded = value.encode("utf-8")
    f.write(struct.pack(">h", len(encoded)))
    f.write(encoded)


def _write_array(f, value: np.ndarray):
    """Writes an NBT array (ByteArray, IntArray, LongArray)."""
    f.write(struct.pack(">i", len(value)))
    f.write(value.tobytes())


def _write_list(f, value: List[NbtContents]):
    """Writes a TAG_List to the file."""
    if not value:
        f.write(struct.pack(">B", TagType.EndType.value))
        f.write(struct.pack(">i", 0))
        return

    tag_type = value[0].tag_type
    f.write(struct.pack(">B", tag_type.value))
    f.write(struct.pack(">i", len(value)))

    for item in value:
        _write_payload(f, item)


def _get_struct_format(tag_type: TagType) -> str:
    """Returns the struct format string for a given TagType."""
    return {
        TagType.ByteType: ">b",
        TagType.ShortType: ">h",
        TagType.IntType: ">i",
        TagType.LongType: ">q",
        TagType.FloatType: ">f",
        TagType.DoubleType: ">d",
    }[tag_type]
