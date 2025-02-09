---
title: NBT Structure API
parent: Configuration
---

# Structure API Documentation

## Overview
The `Structure` class allows for the creation and manipulation of Minecraft structure files using the `.nbt` format. It provides methods for placing blocks, creating geometric structures, and saving them as `.nbt` files.

## Class: `Structure`

### Constructor
```python
Structure(nbt_file: Optional[str] = None)
```
- **nbt_file** (Optional[str]): If provided, loads an existing NBT file.

## Methods

### `save`
```python
save(out_file: str)
```
Saves the current structure as an NBT file.
- **out_file** (str): The filename to save the structure.

### `set_block`
```python
set_block(x: int, y: int, z: int, name: str, properties: Optional[dict] = None, nbt: Optional[dict] = None)
```
Places a block at the given coordinates.
- **x, y, z** (int): Block position.
- **name** (str): Minecraft block ID.
- **properties** (Optional[dict]): Block properties.
- **nbt** (Optional[dict]): Block-specific NBT data.

### `set_cuboid`
```python
set_cuboid(x0: int, y0: int, z0: int, x1: int, y1: int, z1: int, name: str, properties: Optional[dict] = None, nbt: Optional[dict] = None)
```
Creates a solid cuboid of blocks from `(x0, y0, z0)` to `(x1, y1, z1)`.

### `set_walls`
```python
set_walls(x0: int, y0: int, z0: int, x1: int, y1: int, z1: int, name: str, properties: Optional[dict] = None, nbt: Optional[dict] = None, remove_ceiling: bool = False, remove_floor: bool = False)
```
Creates walls around a cuboid without filling the inside.
- **remove_ceiling** (bool): If `True`, removes the top face.
- **remove_floor** (bool): If `True`, removes the bottom face.

### `set_line`
```python
set_line(x0: int, y0: int, z0: int, x1: int, y1: int, z1: int, name: str, properties: Optional[dict] = None, nbt: Optional[dict] = None)
```
Draws a 4-connected diagonal line between two points.

### `set_filled_sphere`
```python
set_filled_sphere(x: int, y: int, z: int, r: int, name: str, properties: Optional[dict] = None, nbt: Optional[dict] = None)
```
Creates a solid sphere centered at `(x, y, z)` with radius `r`.

### `set_hollow_sphere`
```python
set_hollow_sphere(x: int, y: int, z: int, r1: int, r2: int, name: str, properties: Optional[dict] = None, nbt: Optional[dict] = None)
```
Creates a hollow sphere between radii `r1` and `r2`.

### `set_cylinder`
```python
set_cylinder(x: int, y: int, z: int, r: int, h: int, name: str, properties: Optional[dict] = None, nbt: Optional[dict] = None)
```
Creates a solid cylinder centered at `(x, z)` with height `h` and radius `r`.

### `set_hollow_cylinder`
```python
set_hollow_cylinder(x: int, y: int, z: int, r1: int, r2: int, h: int, name: str, properties: Optional[dict] = None, nbt: Optional[dict] = None)
```
Creates a hollow cylinder with inner radius `r1` and outer radius `r2`.

## Example Usage
```python
if __name__ == "__main__":
    structure = Structure()
    
    structure.set_cuboid(0, 0, 0, 10, 10, 10, "minecraft:stone")
    structure.set_walls(15, 0, 0, 25, 10, 10, "minecraft:diamond_ore")
    structure.set_line(30, 0, 0, 40, 10, 10, "minecraft:gold_ore")
    structure.set_filled_sphere(55, 5, 5, 5, "minecraft:glass")
    structure.set_hollow_sphere(75, 5, 5, 3, 5, "minecraft:torch")
    structure.set_cylinder(95, 0, 5, 5, 10, "minecraft:gold_block")
    structure.set_hollow_cylinder(115, 0, 5, 5, 10, 10, "minecraft:oak_planks")
    
    structure.save("debug_structure.nbt")
```
This will generate an NBT file containing various geometric structures that can be loaded into Minecraft.

