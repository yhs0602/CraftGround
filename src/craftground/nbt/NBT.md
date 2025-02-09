# Usage of NBT parser

1. read an NBT file (file -> `NBT`)
   ```bnf
   NBT = (name: str, content: NbtContents)
    NbtContents = (tag_type: TagType, value: int | str | float | list[int] | list[str] | list[float] | list[NbtContents] | list[NBT])
   ```
2. NBT is a raw data structure, so it needs to be parsed into a semantic data structure.
3. Call the `from_nbt` method to semantically parse the NBT file into a dataclass. (`NBT` -> `NBTSerializable`)