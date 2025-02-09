from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BlockEntityNBT:
    pass


@dataclass
class EntityNBT:
    Air: int  # TAG_Short
    CustomName: str  # TAG_String, JSON
    CustomNameVisible: bool  # TAG_Byte
    FallDistance: float  # TAG_Float
    fall_distance: float  # TAG_Double
    Fire: int  # TAG_Short
    Glowing: bool  # TAG_Byte
    HasVisualFire: bool  # TAG_Byte
    id: str  # TAG_String
    Invulnerable: bool  # TAG_Byte
    Motion: List[float]  # List[TAG_Double]
    NoGravity: bool  # TAG_Byte
    OnGround: bool  # TAG_Byte
    Passengers: List["EntityNBT"]  # List[TAG_Compound]
    PortalCooldown: int  # TAG_Int
    Pos: List[float]  # List[TAG_Double]
    Rotation: List[float]  # List[TAG_Float] # yaw, pitch
    Silent: bool  # TAG_Byte
    Tags: List[str]  # List[TAG_String], list of scoreboard tags
    TicksFrozen: int  # TAG_Int
    UUID: List[int]  # List[TAG_Int]


@dataclass
class StructureEntityNBT:
    post: List[float]  # TAG_Double
    blockPos: List[int]  # TAG_Int
    nbt: Optional[EntityNBT] = None


@dataclass
class BlockNBT:
    state: int
    pos: List[int]
    nbt: Optional[BlockEntityNBT] = None


@dataclass
class PaletteNBT:
    Name: str
    Properties: Optional[dict] = None


@dataclass
class StructureNBT:
    """NBT Data Structure"""

    DataVersion: int
    size: List[int]
    entities: List[StructureEntityNBT]
    palette: List[PaletteNBT]
    blocks: List[BlockNBT]
