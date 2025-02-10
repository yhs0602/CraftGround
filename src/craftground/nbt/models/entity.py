from dataclasses import dataclass
from typing import Optional
from ..nbt_dataclass import (
    NBTBase,
    NBTByte,
    NBTCompound,
    NBTDouble,
    NBTFloat,
    NBTInt,
    NBTList,
    NBTLong,
    NBTSerializable,
    NBTShort,
    NBTString,
)


@dataclass
class EntityNBT(NBTSerializable):
    """Base class for all entities."""

    Air: NBTShort
    CustomName: Optional[NBTString]
    CustomNameVisible: Optional[NBTByte]
    FallDistance: Optional[NBTFloat]
    fall_distance: Optional[NBTDouble]
    Fire: NBTShort
    Glowing: NBTByte
    HasVisualFire: NBTByte
    id: NBTString
    Invulnerable: NBTByte
    Motion: NBTList[NBTDouble]
    NoGravity: NBTByte
    OnGround: NBTByte
    Passengers: Optional[NBTList["EntityNBT"]]
    PortalCooldown: NBTInt
    Pos: NBTList[NBTDouble]
    Rotation: NBTList[NBTFloat]
    Silent: Optional[NBTByte]
    Tags: Optional[NBTList[NBTString]]
    TicksFrozen: Optional[NBTInt]
    UUID: NBTList[NBTInt]


@dataclass
class PotionEffectNBT(NBTSerializable):
    """Represents a potion effect applied to an entity."""

    ambient: NBTByte  # If true, effect is provided by a beacon
    amplifier: NBTByte  # Potion effect level (0 = level 1)
    duration: NBTInt  # Duration in game ticks (-1 = infinite)
    hidden_effect: Optional[NBTCompound]  # Lower amplifier effect of the same type
    id: NBTString  # Effect name (e.g., "minecraft:strength")
    show_icon: NBTByte  # If true, effect icon is shown
    show_particles: NBTByte  # If true, particles are shown


@dataclass
class ItemNBT(NBTSerializable):
    """Represents an item in an entity's inventory."""

    id: NBTString  # Item ID (e.g., "minecraft:diamond_sword")
    count: NBTInt  # Number of items in the stack
    components: Optional[NBTCompound] = None  # Item components


@dataclass
class ItemEntityNBT(EntityNBT):
    """Item entity data."""

    Age: NBTShort
    Health: NBTFloat
    Item: ItemNBT
    Owner: NBTList[
        NBTInt
    ]  # UUID of the player who owns the item. Used by give command.
    PickupDelay: NBTShort
    Thrower: NBTList[NBTInt]  # UUID of the player who threw the item


@dataclass
class ExperienceOrbEntityNBT(EntityNBT):
    """Experience orb entity data."""

    Age: NBTShort
    Count: NBTInt
    Health: NBTShort
    Value: NBTShort


# @dataclass
# class PiglinMemoryNBT(NBTSerializable):
#     `minecraft:admiring_disabled`: NBTByte


@dataclass
class BrainNBT(NBTSerializable):
    """Brain data for an entity."""

    memories: Optional[NBTCompound] = None


@dataclass
class LivingEntityNBT(EntityNBT):
    """Entities that are alive (mobs, players, etc.)."""

    AbsorptionAmount: Optional[NBTFloat] = None
    active_effects: Optional[NBTList[PotionEffectNBT]] = None
    ArmorDropChances: Optional[NBTList[NBTFloat]] = None
    ArmorItems: Optional[NBTList[ItemNBT]] = None
    attributes: Optional[NBTList[NBTCompound]] = None
    body_armor_drop_chance: (
        NBTFloat  # Until JE 1.21.5,  Chance to drop the item in the body armor slot.
    ) = 0.0
    body_armor_item: Optional[ItemNBT] = None  # Until JE 1.21.5, Body armor item.
    Brain: Optional[NBTCompound[NBTBase]] = None
    CanPickUpLoot: Optional[NBTByte] = None
    DeathTime: Optional[NBTShort] = None
    drop_chances: Optional[NBTCompound[NBTFloat]] = None
    equipment: Optional[NBTCompound[NBTCompound]] = None
    FallFlying: Optional[NBTByte] = None
    Health: NBTFloat = 1.0
    HurtByTimestamp: Optional[NBTInt] = None
    HurtTime: Optional[NBTShort] = None
    HandDropChances: Optional[NBTList[NBTFloat]] = None
    HandItems: Optional[NBTList[NBTCompound]] = None
    LeftHanded: Optional[NBTByte] = None
    leash: Optional[NBTCompound] = None
    LeftHanded: Optional[NBTByte] = None
    NoAI: Optional[NBTByte] = None
    PersistenceRequired: Optional[NBTByte] = None
    SleepingX: Optional[NBTInt] = None
    SleepingY: Optional[NBTInt] = None
    SleepingZ: Optional[NBTInt] = None
    Team: Optional[NBTString] = None


@dataclass
class EventNBT(NBTSerializable):
    """Event data for an allay entity."""

    distance: NBTInt  # Nonnegative integer
    game_event: NBTString  #  A resource location of the game event.
    pos: NBTList[NBTDouble]  # Exact position (x, y, z)
    projectile_owner: Optional[NBTList[NBTInt]]  # UUID of the projectile owner
    source: Optional[NBTList[NBTInt]]  # UUID of the source entity


@dataclass
class AllaySourceNBT(NBTSerializable):
    """Source data for an allay entity."""

    type: NBTString  # A resource location of the source type.
    pos: Optional[NBTList[NBTInt]] = None  # Exact position (x, y, z)
    source_entity: Optional[NBTList[NBTInt]] = None  # UUID of the source entity
    y_offset: Optional[NBTFloat] = None  # Y offset


@dataclass
class AllayListenerNBT(NBTSerializable):
    """Listener data for an allay entity."""

    distance: NBTInt  # Nonnegative integer
    event: Optional[EventNBT] = None
    event_delay: NBTInt = 0  # Nonnegative integer
    event_distance: NBTInt = 0  # Nonnegative integer
    range: NBTInt = 0  # Nonnegative integer


@dataclass
class AllayNBT(LivingEntityNBT):
    """Allay entity data."""

    CanDuplicate: Optional[NBTByte] = None
    DuplicationCooldown: Optional[NBTLong] = None
    Inventory: Optional[NBTList[NBTCompound]] = None
    listener: Optional[NBTCompound] = None


@dataclass
class PlayerNBT(LivingEntityNBT):
    """Player entity data."""

    PlayerGameMode: Optional[NBTInt] = None
    PlayerUUID: NBTList[NBTInt] = None
    PlayerInventory: Optional[NBTList[NBTCompound]] = None
    PlayerEnderChest: Optional[NBTList[NBTCompound]] = None
    PlayerXP: Optional[NBTInt] = None
    PlayerLevel: Optional[NBTInt] = None
    PlayerHealth: NBTFloat = 0.0
    PlayerFoodLevel: Optional[NBTInt] = None


@dataclass
class ArmorStandNBT(EntityNBT):
    """Armor Stand entity."""

    ShowArms: Optional[NBTByte] = None
    Small: Optional[NBTByte] = None
    Marker: Optional[NBTByte] = None
    Invisible: Optional[NBTByte] = None
    NoBasePlate: Optional[NBTByte] = None


@dataclass
class ProjectileNBT(EntityNBT):
    """Represents projectiles like arrows."""

    Critical: Optional[NBTByte] = None
    PiercingLevel: Optional[NBTByte] = None
    OwnerUUID: Optional[NBTList[NBTInt]] = None
    pickup: Optional[NBTByte] = None
