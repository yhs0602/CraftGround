from typing import List, Optional, Dict, Tuple
from .models.structure import BlockNBT, PaletteNBT, StructureEntityNBT, StructureNBT
from .models.entity import ItemEntityNBT, ItemNBT
from .nbt_dataclass import NBTCompound, NBTInt, NBTShort, NBTString
from .nbt_io import read_nbt, write_nbt


class Structure:
    def __init__(self, nbt_file: Optional[str] = None):
        self.nbt_file = nbt_file
        self.nbt_dict: dict = {}

        self._blocks: Dict[Tuple[int, int, int], dict] = {}
        self._palette: Dict[dict, int] = {}
        self._entites: List[StructureEntityNBT] = []

        if nbt_file:
            nbt = read_nbt(nbt_file)
            nbt_struct = StructureNBT.from_nbt(nbt)
            for block in nbt_struct.blocks:
                self._blocks[tuple(block["pos"])] = block
            for palette_index, palette in enumerate(nbt_struct.palette):
                self._palette[palette] = palette_index
            for entity in nbt_struct.entities:
                self._entites.append(entity)

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

        structure_file = StructureNBT(
            DataVersion=NBTInt(19133),
            size=size,
            palette=[palette for palette in self._palette],
            blocks=[
                BlockNBT(
                    state=block["state"],
                    pos=[
                        block["pos"][0] - min_x,
                        block["pos"][1] - min_y,
                        block["pos"][2] - min_z,
                    ],
                    nbt=NBTCompound(block["nbt"]) if "nbt" in block else None,
                )
                for block in self._blocks.values()
            ],
            entities=self._entites,  # no entities for now
        )
        # create the NBT structure
        # print(structure_file.to_snbt(indent=2))
        write_nbt(NBTCompound({"": structure_file}), out_file)

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
        palette = PaletteNBT(Name=NBTString(name))
        if properties is not None:
            palette.Properties = properties

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
        y: int,
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
                        self.set_block(x + dx, y + dy, z + dz, name, properties, nbt)

    def set_hollow_cylinder(
        self,
        x: int,
        y: int,
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
                        self.set_block(x + dx, y + dy, z + dz, name, properties, nbt)

    def add_item(
        self,
        x: float,
        y: float,
        z: float,
        item: str,
        count: int = 1,
        auto_destroy: bool = False,
        can_pickup: bool = True,
    ):
        """Adds an item to the structure."""
        self._entities.append(
            StructureEntityNBT(
                pos=[x, y, z],
                blockPos=[int(x), int(y), int(z)],
                nbt=ItemEntityNBT(
                    Item=ItemNBT(
                        id=NBTString(item),
                        Count=NBTInt(count),
                    ),
                    Age=NBTShort(0) if auto_destroy else NBTShort(-32768),
                    PickupDelay=NBTShort(0) if can_pickup else NBTShort(32767),
                ),
            )
        )


if __name__ == "__main__":
    structure = Structure()
    structure.set_cuboid(0, 0, 0, 10, 10, 10, "minecraft:stone")
    structure.set_walls(0, 0, 0, 10, 10, 10, "minecraft:diamond_ore")
    structure.set_line(0, 0, 0, 10, 10, 10, "minecraft:gold_ore")
    structure.set_filled_sphere(5, 5, 5, 5, "minecraft:glass")
    structure.set_hollow_sphere(5, 5, 5, 3, 5, "minecraft:air")
    structure.set_cylinder(5, 5, 5, 5, 5, "minecraft:gold_block")
    structure.set_hollow_cylinder(5, 5, 5, 3, 5, 5, "minecraft:oak_planks")
    structure.save("output.nbt")

    structure = Structure()

    # 1. Basic cuboid test (box filled with stone)
    structure.set_cuboid(0, 0, 0, 10, 10, 10, "minecraft:stone")

    # 2. Wall creation test (walls made of diamond ore)
    structure.set_walls(15, 0, 0, 25, 10, 10, "minecraft:diamond_ore")

    # 3. Line drawing test (diagonal line made of gold ore)
    structure.set_line(30, 0, 0, 40, 10, 10, "minecraft:gold_ore")

    # 4. Filled sphere test (glass sphere)
    structure.set_filled_sphere(55, 5, 5, 5, "minecraft:glass")

    # 5. Hollow sphere test (floating sphere)
    structure.set_hollow_sphere(75, 5, 5, 3, 5, "minecraft:torch")

    # 6. Cylinder test (pillar made of gold blocks)
    structure.set_cylinder(95, 0, 5, 5, 10, "minecraft:gold_block")

    # 7. Hollow cylinder test (cylindrical wall made of oak planks)
    structure.set_hollow_cylinder(115, 0, 5, 5, 10, 10, "minecraft:oak_planks")

    # Save result
    structure.save("debug_structure.nbt")
