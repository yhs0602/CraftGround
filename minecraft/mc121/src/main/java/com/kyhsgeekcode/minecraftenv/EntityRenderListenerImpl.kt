package com.kyhsgeekcode.minecraftenv

import net.minecraft.entity.Entity

class EntityRenderListenerImpl(
    renderer: AddListenerInterface,
) : EntityRenderListener {
    private val _entities: MutableSet<Entity> = mutableSetOf()
    val entities: Set<Entity>
        get() = _entities

    override fun onEntityRender(entity: Entity) {
        _entities.add(entity)
    }

    override fun clear() {
        _entities.clear()
    }

    init {
        renderer.addRenderListener(this)
    }
}
