package com.kyhsgeekcode.minecraftenv

import com.kyhsgeekcode.minecraftenv.proto.blockInfo
import com.kyhsgeekcode.minecraftenv.proto.entityInfo
import com.kyhsgeekcode.minecraftenv.proto.hitResult
import com.kyhsgeekcode.minecraftenv.proto.itemStack
import com.kyhsgeekcode.minecraftenv.proto.statusEffect
import net.minecraft.block.Block
import net.minecraft.entity.Entity
import net.minecraft.entity.LivingEntity
import net.minecraft.entity.effect.StatusEffectInstance
import net.minecraft.item.Item
import net.minecraft.item.ItemStack
import net.minecraft.util.hit.BlockHitResult
import net.minecraft.util.hit.EntityHitResult
import net.minecraft.util.hit.HitResult
import net.minecraft.util.math.BlockPos
import net.minecraft.world.World

fun Entity.toMessage() =
    entityInfo {
        uniqueName = this@toMessage.uuidAsString
        translationKey = type.translationKey
        x = this@toMessage.x
        y = this@toMessage.y
        z = this@toMessage.z
        yaw = this@toMessage.yaw.toDouble()
        pitch = this@toMessage.pitch.toDouble()
        health = (this@toMessage as? LivingEntity)?.health?.toDouble() ?: 0.0
    }

fun StatusEffectInstance.toMessage() =
    statusEffect {
        translationKey = this@toMessage.translationKey
        duration = this@toMessage.duration
        amplifier = this@toMessage.amplifier
    }

fun HitResult.toMessage(world: World) =
    when (type) {
        HitResult.Type.MISS ->
            hitResult {
                type = com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.HitResult.Type.MISS
            }

        HitResult.Type.BLOCK ->
            hitResult {
                type = com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.HitResult.Type.BLOCK
                val blockPos = (this@toMessage as BlockHitResult).blockPos
                val block = world.getBlockState(blockPos).block
                targetBlock = block.toMessage(blockPos)
            }

        HitResult.Type.ENTITY ->
            hitResult {
                val entity = (this@toMessage as EntityHitResult).entity
                type = com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.HitResult.Type.ENTITY
                targetEntity = entity.toMessage()
            }
    }

fun Block.toMessage(blockPos: BlockPos) =
    blockInfo {
        x = blockPos.x
        y = blockPos.y
        z = blockPos.z
        translationKey = this@toMessage.translationKey
    }

fun ItemStack.toMessage() =
    itemStack {
        rawId = Item.getRawId(this@toMessage.item)
        translationKey = this@toMessage.translationKey
        count = this@toMessage.count
        durability = this@toMessage.maxDamage - this@toMessage.damage
        maxDurability = this@toMessage.maxDamage
    }
