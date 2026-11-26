package com.kyhsgeekcode.minecraftenv

import com.kyhsgeekcode.minecraftenv.proto.blockInfo
import com.kyhsgeekcode.minecraftenv.proto.entityInfo
import com.kyhsgeekcode.minecraftenv.proto.hitResult
import com.kyhsgeekcode.minecraftenv.proto.itemStack
import com.kyhsgeekcode.minecraftenv.proto.statusEffect
import net.minecraft.core.BlockPos
import net.minecraft.world.effect.MobEffectInstance
import net.minecraft.world.entity.Entity
import net.minecraft.world.entity.LivingEntity
import net.minecraft.world.entity.animal.Animal
import net.minecraft.world.item.Item
import net.minecraft.world.item.ItemStack
import net.minecraft.world.level.Level
import net.minecraft.world.level.block.Block
import net.minecraft.world.phys.BlockHitResult
import net.minecraft.world.phys.EntityHitResult
import net.minecraft.world.phys.HitResult

fun Entity.toMessage() =
    entityInfo {
        uniqueName = this@toMessage.stringUUID
        translationKey = type.descriptionId
        x = this@toMessage.x
        y = this@toMessage.y
        z = this@toMessage.z
        yaw = this@toMessage.yRot.toDouble()
        pitch = this@toMessage.xRot.toDouble()
        health = (this@toMessage as? LivingEntity)?.health?.toDouble() ?: 0.0
        inLove = (this@toMessage as? Animal)?.isInLove ?: false
    }

fun MobEffectInstance.toMessage() =
    statusEffect {
        translationKey = this@toMessage.descriptionId
        duration = this@toMessage.duration
        amplifier = this@toMessage.amplifier
    }

fun HitResult.toMessage(world: Level) =
    when (type) {
        HitResult.Type.MISS -> {
            hitResult {
                type = com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.HitResult.Type.MISS
            }
        }

        HitResult.Type.BLOCK -> {
            hitResult {
                type = com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.HitResult.Type.BLOCK
                val blockPos = (this@toMessage as BlockHitResult).blockPos
                val block = world.getBlockState(blockPos).block
                targetBlock = block.toMessage(blockPos)
            }
        }

        HitResult.Type.ENTITY -> {
            hitResult {
                val entity = (this@toMessage as EntityHitResult).entity
                type = com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.HitResult.Type.ENTITY
                targetEntity = entity.toMessage()
            }
        }
    }

fun Block.toMessage(blockPos: BlockPos) =
    blockInfo {
        x = blockPos.x
        y = blockPos.y
        z = blockPos.z
        translationKey = this@toMessage.descriptionId
    }

fun ItemStack.toMessage() =
    itemStack {
        rawId = Item.getId(this@toMessage.item)
        translationKey = this@toMessage.item.descriptionId
        count = this@toMessage.count
        durability = this@toMessage.maxDamage - this@toMessage.damageValue
        maxDurability = this@toMessage.maxDamage
    }
