from dataclasses import dataclass
from typing import List, Optional

from nbt.nbt_struct import NBT, NbtContents, TagType


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

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(
            TagType.CompoundType,
            [
                NBT("Air", NbtContents.short(self.Air)),
                NBT("CustomName", NbtContents.string(self.CustomName)),
                NBT("CustomNameVisible", NbtContents.byte(self.CustomNameVisible)),
                NBT("FallDistance", NbtContents.float(self.FallDistance)),
                NBT("fall_distance", NbtContents.double(self.fall_distance)),
                NBT("Fire", NbtContents.short(self.Fire)),
                NBT("Glowing", NbtContents.byte(self.Glowing)),
                NBT("HasVisualFire", NbtContents.byte(self.HasVisualFire)),
                NBT("id", NbtContents.string(self.id)),
                NBT("Invulnerable", NbtContents.byte(self.Invulnerable)),
                NBT(
                    "Motion",
                    NbtContents.list([NbtContents.double(m) for m in self.Motion]),
                ),
                NBT("NoGravity", NbtContents.byte(self.NoGravity)),
                NBT("OnGround", NbtContents.byte(self.OnGround)),
                NBT(
                    "Passengers",
                    NbtContents.list(
                        [passenger.to_nbt_contents() for passenger in self.Passengers]
                    ),
                ),
                NBT("PortalCooldown", NbtContents.int(self.PortalCooldown)),
                NBT("Pos", NbtContents.list([NbtContents.double(p) for p in self.Pos])),
                NBT(
                    "Rotation",
                    NbtContents.list([NbtContents.float(r) for r in self.Rotation]),
                ),
                NBT("Silent", NbtContents.byte(self.Silent)),
                NBT(
                    "Tags", NbtContents.list([NbtContents.string(t) for t in self.Tags])
                ),
                NBT("TicksFrozen", NbtContents.int(self.TicksFrozen)),
                NBT("UUID", NbtContents.list([NbtContents.int(u) for u in self.UUID])),
            ],
        )


@dataclass
class StructureEntityNBT:
    post: List[float]  # TAG_Double
    blockPos: List[int]  # TAG_Int
    nbt: Optional[EntityNBT] = None

    def to_nbt_contents(self) -> NbtContents:
        compound_list = [
            NBT(
                "pos",
                NbtContents.list(
                    [
                        NbtContents.double(self.post[0]),
                        NbtContents.double(self.post[1]),
                        NbtContents.double(self.post[2]),
                    ]
                ),
            ),
            NBT(
                "blockPos",
                NbtContents.list(
                    [
                        NbtContents.int(self.blockPos[0]),
                        NbtContents.int(self.blockPos[1]),
                        NbtContents.int(self.blockPos[2]),
                    ]
                ),
            ),
        ]
        if self.nbt:
            compound_list.append(NBT("nbt", self.nbt.to_nbt_contents()))

        return NbtContents(
            TagType.CompoundType,
            compound_list,
        )


@dataclass
class BlockNBT:
    state: int
    pos: List[int]
    nbt: Optional[BlockEntityNBT] = None

    def to_nbt_contents(self) -> NbtContents:
        compound_list = [
            NBT("state", NbtContents.int(self.state)),
            NBT(
                "pos",
                NbtContents.list(
                    [
                        NbtContents.int(self.pos[0]),
                        NbtContents.int(self.pos[1]),
                        NbtContents.int(self.pos[2]),
                    ]
                ),
            ),
        ]
        if self.nbt:
            compound_list.append(NBT("nbt", self.nbt.to_nbt_contents()))

        return NbtContents(
            TagType.CompoundType,
            compound_list,
        )


@dataclass
class PaletteNBT:
    Name: str
    Properties: Optional[dict] = None

    def to_nbt_contents(self) -> NbtContents:
        compound_list = [
            NBT("Name", NbtContents.string(self.Name)),
        ]
        if self.Properties:
            compound_list.append(
                NBT("Properties", NbtContents.compound(self.Properties))
            )

        return NbtContents(
            TagType.CompoundType,
            compound_list,
        )


@dataclass
class StructureNBT:
    """NBT Data Structure"""

    DataVersion: int
    size: List[int]
    entities: List[StructureEntityNBT]
    palette: List[PaletteNBT]
    blocks: List[BlockNBT]

    def to_nbt_contents(self) -> NbtContents:
        return NbtContents(
            TagType.CompoundType,
            [
                NBT("DataVersion", NbtContents.int(self.DataVersion)),
                NBT(
                    "size",
                    NbtContents.list(
                        [
                            NbtContents.int(self.size[0]),
                            NbtContents.int(self.size[1]),
                            NbtContents.int(self.size[2]),
                        ]
                    ),
                ),
                NBT(
                    "entities",
                    NbtContents.list(
                        [entity.to_nbt_contents() for entity in self.entities]
                    ),
                ),
                NBT(
                    "palette",
                    NbtContents.list(
                        [palette.to_nbt_contents() for palette in self.palette]
                    ),
                ),
                NBT(
                    "blocks",
                    NbtContents.list(
                        [block.to_nbt_contents() for block in self.blocks]
                    ),
                ),
            ],
        )
