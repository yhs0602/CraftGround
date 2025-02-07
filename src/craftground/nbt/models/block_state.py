from dataclasses import dataclass

from nbt.nbt_dataclass import NBTByte, NBTSerializable, NBTString


@dataclass
class WaterNBT(NBTSerializable):
    """Represents a water block in the structure."""

    falling: NBTByte  # Whether the water is falling, always False


@dataclass
class FlowingWaterNBT(NBTSerializable):
    """Represents a falling water block in the structure."""

    falling: NBTByte  # True for falling water, false for water with a block below
    level: NBTByte  # Water level (1-8)


@dataclass
class LavaNBT(NBTSerializable):
    """Represents a lava block in the structure."""

    falling: NBTByte  # Whether the lava is falling, always False


@dataclass
class FlowingLavaNBT(NBTSerializable):
    """Represents a falling lava block in the structure."""

    falling: NBTByte
    level: NBTByte  # Lava level (1-8)


@dataclass
class TNTNBT(NBTSerializable):
    """Represents a TNT block in the structure."""

    unstable: NBTByte  # Whether the TNT is unstable, always False


@dataclass
class TorchNBT(NBTSerializable):
    """Represents a torch block in the structure."""

    facing: NBTString  # Torch orientation (e.g., "north")
