from dataclasses import dataclass
from nbt.nbt_dataclass import NBTSerializable
from nbt.nbt_struct import NbtContents, TagType
from typing import Dict, Any

# A NBTSerializable class that represents unknown NBT data.
# For example, AmethystNBT is not declared in this package,
# but it can be parsed as a CompoundNBT.
# structure: StructureNBT = parse_structure_file("my_structure.nbt")
# structure.palette: NBTList[PaletteNBT]
# structure.palette.Properties: NBTCompound[???]
# here comes the CompoundNBT.
# structure.palette.Properties: CompoundNBT
# structure.palette.Properties.facing: NBTString = "up" | "down" | "north" | "south" | "west" | "east"
# structure.palette.Properties.waterlogged: NBTString = "true" | "false"
# If a new NBT key is added to the structure.palette.Properties,
# the type of the new key must be declared in the CompoundNBT beforehand.
# structure.palette.Properties.facing = "up"


@dataclass
class CompoundNBT(NBTSerializable):
    """A generic NBT Compound that enforces existing NBT tag types on attribute setting."""

    def __init__(self, field_dict: Dict[str, NbtContents]):
        """Initialize with a dictionary of field values."""
        super().__setattr__("field_dict", field_dict)  # Bypass __setattr__

    def __setattr__(self, name: str, value: Any):
        """Set an attribute, ensuring type consistency based on existing NBT tag type."""
        if name == "field_dict":
            super().__setattr__(name, value)  # Allow setting field_dict directly
            return

        if name not in self.field_dict:
            raise AttributeError(
                f"Cannot dynamically add new key '{name}' to CompoundNBT. Define it in advance."
            )

        original_value = self.field_dict[name]
        expected_tag_type = original_value.tag_type

        # Convert value to match the existing NBT type
        if expected_tag_type == TagType.ByteType and isinstance(value, bool):
            value = NbtContents(TagType.ByteType, value)
        elif expected_tag_type == TagType.IntType and isinstance(value, int):
            value = NbtContents(TagType.IntType, value)
        elif expected_tag_type == TagType.DoubleType and isinstance(value, float):
            value = NbtContents(TagType.DoubleType, value)
        elif expected_tag_type == TagType.StringType and isinstance(value, str):
            value = NbtContents(TagType.StringType, value)
        elif expected_tag_type == TagType.CompoundType and isinstance(
            value, CompoundNBT
        ):
            value = value.to_nbt_contents()
        else:
            raise ValueError(
                f"Invalid type for {name}: Expected {expected_tag_type}, got {type(value)} ({value})"
            )

        self.field_dict[name] = value

    def __getattr__(self, name: str) -> Any:
        """Retrieve an attribute value, automatically unwrapping NBTBase values."""
        if name not in self.field_dict:
            raise AttributeError(f"Attribute {name} not found in CompoundNBT")

        value = self.field_dict[name]
        if isinstance(value, NbtContents):
            return value.value  # Return the primitive value

        return value

    def to_nbt_contents(self) -> NbtContents:
        """Convert CompoundNBT to an NbtContents dictionary."""
        return NbtContents(
            TagType.CompoundType,
            {k: v.to_nbt_contents() for k, v in self.field_dict.items()},
        )
