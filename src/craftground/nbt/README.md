# NBT structure format reference
An NBT file is a list of tags, and can be parsed as a tree structure like JSON. This simple package parses the provided NBT file and returns a dictionary with the parsed data.

## Structure format
For more information, see [the wiki](https://minecraft.wiki/w/Structure_file).

### DataVersion
It is an integer representing the version of the NBT structure.

### size
It describes the size of the structure in blocks.

### entities
It is a list of entities in the structure.
- nbt: It is a dictionary of the entity's NBT data.
  - Motion
  - Facing
  - ItemRotation
  - Invulnerable
  - Air
  - OnGround
  - PortalCooldown
  - Rotation
  - FallDistnace
  - Item
  - ItemDropChance
  - Pos
  - Fire
  - TileY
  - id
  - TileX
  - Invisible
  - UUID
  - TileZ
  - Fixed
- blockPos: It is an integer list of the block position of the entity.
- pos: It is a double list of the position of the entity.

### blocks
It is a list of blocks in the structure.
- pos: It is an integer list of the block position.
- state: It is the index of the block state in the palette.
- nbt: It is an optional dictionary of the block's NBT data.

### palette
It describes the actual information of the blocks in the structure.
- Name: It is the name of the block. (Example: `minecraft:redstone_block`)
- Properties: It is an optional dictionary of the block's properties. (Example: `{"facing": "north"}`)