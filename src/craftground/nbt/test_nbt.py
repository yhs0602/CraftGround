import json
import os

from models.structure import StructureNBT
from nbt_dataclass import NBTCompound
from nbt_io import read_nbt


current_dir = os.path.dirname(os.path.abspath(__file__))
nbt_path = os.path.join(current_dir, "nbts/room_with_item.nbt")


if __name__ == "__main__":
    parsed: NBTCompound = read_nbt(nbt_path)
    # print(parsed)
    structure: StructureNBT = StructureNBT.from_nbt(parsed)
    # print(structure.to_snbt(indent=2))
    # print(json.dumps(parsed.dump_to_dict({}), indent=2))
    # print(f"{structure.author=}")
    # print(f"{structure.size=}")
    # # input()
    # for palette in structure.palette:
    #     print(f"{palette=}")
    #     # input()
    # # input()
    # print(f"{structure.blocks=}")
    # # input()
    # print(f"{structure.entities=}")
    # # input()
    # print(f"{structure.palettes=}")
    print(structure.to_snbt(indent=2))
