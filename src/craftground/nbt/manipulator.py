from typing import Optional, Dict, Tuple
from nbt.nbt_io import read_nbt, write_nbt
from nbt.nbt_struct import NBT, NbtContents, TagType
import math


class Structure:
    def __init__(self, nbt_file: Optional[str] = None):
        self.nbt_file = nbt_file
        self.nbt_dict: dict = {}

        self._blocks: Dict[Tuple[int, int, int], dict] = {}
        self._palette: Dict[dict, int] = {}

        if nbt_file:
            nbt = read_nbt(nbt_file)
            self.nbt_dict = nbt.dump_to_dict({})[""]
            for block in self.nbt_dict["blocks"]:
                self._blocks[tuple(block["pos"])] = block
            for palette_index, palette in enumerate(self.nbt_dict["palette"]):
                self._palette[palette] = palette_index

    def build_entity_nbt(self) -> NbtContents:
        pass

    @staticmethod
    def build_block_nbt_nbt(nbt: dict) -> NbtContents:
        pass

    @staticmethod
    def build_block_nbt(block: dict) -> NbtContents:
        temp = [
            NBT(
                "state",
                NbtContents.int(block["state"]),
            ),
            NBT(
                "pos",
                NbtContents.list(
                    [
                        NbtContents.int(block["pos"][0]),
                        NbtContents.int(block["pos"][1]),
                        NbtContents.int(block["pos"][2]),
                    ]
                ),
            ),
        ]
        if "nbt" in block:
            temp.append(NBT("nbt", NbtContents.compound(block["nbt"])))

        return NbtContents(
            TagType.CompoundType,
        )

    def build_palette_nbt(self) -> NbtContents:
        pass

    def save(self, out_file: str):
        """Saves the structure to a file."""
        # calculate the size of the structure
        min_x = min(x for x, _, _ in self._blocks)
        max_x = max(x for x, _, _ in self._blocks)
        min_y = min(y for _, y, _ in self._blocks)
        max_y = max(y for _, y, _ in self._blocks)
        min_z = min(z for _, _, z in self._blocks)
        max_z = max(z for _, _, z in self._blocks)
        size = (max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1)

        structure_compound = NbtContents(
            TagType.CompoundType,
            [
                NBT("DataVersion", NbtContents(TagType.IntType, 3953)),
                NBT(
                    "size",
                    NbtContents.list(
                        [
                            NbtContents.int(size[0]),
                            NbtContents.int(size[1]),
                            NbtContents.int(size[2]),
                        ]
                    ),
                ),
                NBT(
                    "palette",
                    NbtContents.list(
                        [
                            NbtContents.compound(palette)
                            for palette in self._palette.keys()
                        ]
                    ),
                ),
                NBT(
                    "blocks",
                    NbtContents.list(
                        [
                            NbtContents.compound(
                                {
                                    "pos": NbtContents.list(
                                        [
                                            NbtContents.int(x - min_x),
                                            NbtContents.int(y - min_y),
                                            NbtContents.int(z - min_z),
                                        ]
                                    ),
                                    "state": NbtContents.int(block["state"]),
                                    **block.get("nbt", {}),
                                }
                            )
                            for (x, y, z), block in self._blocks.items()
                        ]
                    ),
                ),
            ],
        )
        nbt = NBT("", structure_compound)
        write_nbt(nbt, out_file)

    def set_block_palette(
        self, x: int, y: int, z: int, palette_index: int, nbt: Optional[dict] = None
    ):
        """Sets a block using a given palette index."""
        data = {"pos": (x, y, z), "state": palette_index}
        if nbt:
            data["nbt"] = nbt
        self._blocks[(x, y, z)] = data

    def set_block(
        self,
        x: int,
        y: int,
        z: int,
        name: str,
        properties: Optional[dict] = None,
        nbt: Optional[dict] = None,
    ):
        """Sets a block at the given coordinates with name, properties, and optional NBT data."""
        palette = {"Name": name}
        if properties is not None:
            palette["Properties"] = properties

        # Add to palette if not exists
        palette_index = self._palette.setdefault(palette, len(self._palette))

        block_obj = {"pos": (x, y, z), "state": palette_index}
        if nbt is not None:
            block_obj["nbt"] = nbt
        self._blocks[(x, y, z)] = block_obj

    def set_cuboid(
        self,
        x0: int,
        y0: int,
        z0: int,
        x1: int,
        y1: int,
        z1: int,
        name: str,
        properties: Optional[dict] = None,
        nbt: Optional[dict] = None,
    ):
        """Creates a solid cuboid from (x0, y0, z0) to (x1, y1, z1)."""
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for z in range(min(z0, z1), max(z0, z1) + 1):
                    self.set_block(x, y, z, name, properties, nbt)

    def set_walls(
        self,
        x0: int,
        y0: int,
        z0: int,
        x1: int,
        y1: int,
        z1: int,
        name: str,
        properties: Optional[dict] = None,
        nbt: Optional[dict] = None,
        remove_ceiling: bool = False,
        remove_floor: bool = False,
    ):
        """Creates walls without ceiling or floor if specified."""
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(y0, y1 + 1):
                self.set_block(x, y, z0, name, properties, nbt)
                self.set_block(x, y, z1, name, properties, nbt)

        for z in range(min(z0, z1), max(z0, z1) + 1):
            for y in range(y0, y1 + 1):
                self.set_block(x0, y, z, name, properties, nbt)
                self.set_block(x1, y, z, name, properties, nbt)

        if not remove_floor:
            self.set_cuboid(x0, y0, z0, x1, y0, z1, name, properties, nbt)
        if not remove_ceiling:
            self.set_cuboid(x0, y1, z0, x1, y1, z1, name, properties, nbt)

    def set_line(
        self,
        x0: int,
        y0: int,
        z0: int,
        x1: int,
        y1: int,
        z1: int,
        name: str,
        properties: Optional[dict] = None,
        nbt: Optional[dict] = None,
    ):
        """Creates a 4-connected diagonal line between two points."""
        steps = max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
        for i in range(steps + 1):
            x = round(x0 + (i * (x1 - x0) / steps))
            y = round(y0 + (i * (y1 - y0) / steps))
            z = round(z0 + (i * (z1 - z0) / steps))
            self.set_block(x, y, z, name, properties, nbt)

    def set_filled_sphere(
        self,
        x: int,
        y: int,
        z: int,
        r: int,
        name: str,
        properties: Optional[dict] = None,
        nbt: Optional[dict] = None,
    ):
        """Creates a filled sphere centered at (x, y, z) with radius r."""
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx**2 + dy**2 + dz**2 <= r**2:
                        self.set_block(x + dx, y + dy, z + dz, name, properties, nbt)

    def set_hollow_sphere(
        self,
        x: int,
        y: int,
        z: int,
        r1: int,
        r2: int,
        name: str,
        properties: Optional[dict] = None,
        nbt: Optional[dict] = None,
    ):
        """Creates a hollow sphere with inner radius r1 and outer radius r2."""
        for dx in range(-r2, r2 + 1):
            for dy in range(-r2, r2 + 1):
                for dz in range(-r2, r2 + 1):
                    dist_sq = dx**2 + dy**2 + dz**2
                    if r1**2 <= dist_sq <= r2**2:
                        self.set_block(x + dx, y + dy, z + dz, name, properties, nbt)

    def set_cylinder(
        self,
        x: int,
        z: int,
        r: int,
        h: int,
        name: str,
        properties: Optional[dict] = None,
        nbt: Optional[dict] = None,
    ):
        """Creates a solid cylinder centered at (x, z) with radius r and height h."""
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx**2 + dz**2 <= r**2:
                    for dy in range(h):
                        self.set_block(x + dx, dy, z + dz, name, properties, nbt)

    def set_hollow_cylinder(
        self,
        x: int,
        z: int,
        r1: int,
        r2: int,
        h: int,
        name: str,
        properties: Optional[dict] = None,
        nbt: Optional[dict] = None,
    ):
        """Creates a hollow cylinder with inner radius r1 and outer radius r2."""
        for dx in range(-r2, r2 + 1):
            for dz in range(-r2, r2 + 1):
                dist_sq = dx**2 + dz**2
                if r1**2 <= dist_sq <= r2**2:
                    for dy in range(h):
                        self.set_block(x + dx, dy, z + dz, name, properties, nbt)
