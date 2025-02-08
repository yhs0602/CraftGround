import json
import os

from models.structure import StructureNBT
from nbt_struct import NBT
from nbt_io import read_nbt


current_dir = os.path.dirname(os.path.abspath(__file__))
nbt_path = os.path.join(current_dir, "room_with_item.nbt")


if __name__ == "__main__":
    parsed: NBT = read_nbt(nbt_path)
    structure: StructureNBT = StructureNBT.from_nbt(parsed)
    # print(json.dumps(parsed.dump_to_dict({}), indent=2))
    print(f"{structure.author=}")
    print(f"{structure.size=}")
    input()
    print(f"{structure.palette=}")
    input()
    print(f"{structure.blocks=}")
    input()
    print(f"{structure.entities=}")
    input()
    print(f"{structure.palettes=}")
