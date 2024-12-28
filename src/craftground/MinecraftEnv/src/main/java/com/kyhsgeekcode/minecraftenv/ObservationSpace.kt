package com.kyhsgeekcode.minecraftenv

import net.minecraft.item.Item

data class ItemStack(
    val rawId: Int,
    val translationKey: String,
    val count: Int,
    val durability: Int,
    val maxDurability: Int,
) {
    constructor(itemStack: net.minecraft.item.ItemStack) : this(
        Item.getRawId(itemStack.item),
        itemStack.item.translationKey,
        itemStack.count,
        itemStack.maxDamage - itemStack.damage,
        itemStack.maxDamage,
    )
}

data class BlockInfo(
    val x: Int,
    val y: Int,
    val z: Int,
    val translationKey: String,
)

data class EntityInfo(
    val uniqueName: String,
    val translationKey: String,
    val x: Double,
    val y: Double,
    val z: Double,
    val yaw: Double,
    val pitch: Double,
    val health: Double,
)

data class HitResult(
    val type: net.minecraft.util.hit.HitResult.Type,
    val targetBlock: BlockInfo? = null,
    val targetEntity: EntityInfo? = null,
)

data class StatusEffect(
    val translationKey: String,
    val duration: Int,
    val amplifier: Int,
)

data class ObservationSpace(
    val image: String = "",
    val x: Double = 0.0,
    val y: Double = 0.0,
    val z: Double = 0.0,
    val yaw: Double = 0.0,
    val pitch: Double = 0.0,
    val health: Double = 20.0,
    val foodLevel: Double = 20.0,
    val saturationLevel: Double = 0.0,
    val isDead: Boolean = false,
    val inventory: List<ItemStack> = listOf(),
    val raycastResult: HitResult = HitResult(net.minecraft.util.hit.HitResult.Type.MISS),
    val soundSubtitles: List<SoundEntry> = listOf(),
    val statusEffects: List<StatusEffect> = listOf(),
)
