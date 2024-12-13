package com.kyhsgeekcode.minecraft_env

import net.minecraft.entity.Entity

interface EntityRenderListener {
    fun onEntityRender(entity: Entity)

    fun clear()
}
