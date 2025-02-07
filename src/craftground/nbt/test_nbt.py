from dataclasses import asdict
import json
import os

from nbt_struct import NBT
from nbt_io import read_nbt


current_dir = os.path.dirname(os.path.abspath(__file__))
nbt_path = os.path.join(current_dir, "room_with_item.nbt")


if __name__ == "__main__":
    parsed: NBT = read_nbt(nbt_path)
    print(json.dumps(parsed.dump_to_dict({}), indent=2))
