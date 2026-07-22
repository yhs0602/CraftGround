package com.kyhsgeekcode.minecraftenv.customentity

import net.minecraft.entity.EntityType
import net.minecraft.entity.mob.PathAwareEntity
import net.minecraft.world.World

class RealisticHuman(
    entityType: EntityType<RealisticHuman>,
    world: World,
) : PathAwareEntity(entityType, world)
