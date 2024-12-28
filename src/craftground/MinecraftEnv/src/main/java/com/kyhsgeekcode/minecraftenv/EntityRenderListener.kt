package com.kyhsgeekcode.minecraftenv

import net.minecraft.entity.Entity

interface EntityRenderListener {
    fun onEntityRender(entity: Entity)

    fun clear()
}
