from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from ..models.entity import EntityNBT
from ..nbt_dataclass import (
    NBTBase,
    NBTCompound,
    NBTDouble,
    NBTInt,
    NBTList,
    NBTSerializable,
    NBTString,
)


@dataclass
class PaletteNBT(NBTSerializable):
    """Represents a block state in the structure palette."""

    Name: NBTString
    Properties: Optional[NBTCompound[NBTString]] = None  # Key-value property list

    def __hash__(self) -> int:
        # Hash based on the name and properties
        return hash((self.Name, self.Properties))

    def __eq__(self, value: object) -> bool:
        # Compare based on the name and properties
        if not isinstance(value, PaletteNBT):
            return False
        return self.Name == value.Name and self.Properties == value.Properties


@dataclass
class BlockNBT(NBTSerializable):
    """Represents an individual block in the structure."""

    state: NBTInt  # Index in the palette
    pos: NBTList[NBTInt]  # (x, y, z) position
    nbt: Optional[NBTCompound[NBTBase]] = None  # Block entity NBT (optional)


@dataclass
class StructureEntityNBT(NBTSerializable):
    """Represents an entity in the structure."""

    pos: NBTList[NBTDouble]  # Exact position (x, y, z)
    blockPos: NBTList[NBTInt]  # Block-aligned position (x, y, z)
    nbt: EntityNBT  # Entity data (mandatory)


@dataclass
class StructureNBT(NBTSerializable):
    """Root NBT structure for Minecraft structure files."""

    DataVersion: NBTInt  # Version number
    author: Optional[NBTString] = None  # Creator name (1.13 이전만 존재)
    size: NBTList[NBTInt] = (0, 0, 0)  # Structure size (3 ints)
    palette: NBTList[PaletteNBT] = field(default_factory=list)  # Default block palette
    palettes: Optional[NBTList[NBTList[PaletteNBT]]] = (
        None  # Random palettes (for shipwrecks)
    )
    blocks: NBTList[BlockNBT] = field(default_factory=list)  # List of individual blocks
    entities: NBTList[StructureEntityNBT] = field(
        default_factory=list
    )  # List of entities in the structure
