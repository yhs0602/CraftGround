package com.kyhsgeekcode.minecraftenv

import net.minecraft.util.math.Vec3d

/**
 * Singleton object to store Lidar ray data for visualization.
 * The ray data is updated each tick and rendered by WorldRenderer mixin.
 */
object LidarRayVisualizer {
    /**
     * Data class representing a single ray for visualization
     */
    data class VisualRay(
        val start: Vec3d,
        val end: Vec3d,
        val hitType: Int, // 0 = MISS, 1 = BLOCK, 2 = ENTITY
    )

    /**
     * Whether ray visualization is enabled
     */
    var enabled: Boolean = false
        private set

    /**
     * List of rays to visualize
     */
    private val rays: MutableList<VisualRay> = mutableListOf()

    /**
     * Get a copy of the current rays for rendering
     */
    fun getRays(): List<VisualRay> = synchronized(rays) { rays.toList() }

    /**
     * Update the ray data for visualization
     */
    fun updateRays(newRays: List<VisualRay>) {
        synchronized(rays) {
            rays.clear()
            rays.addAll(newRays)
        }
    }

    /**
     * Clear all rays
     */
    fun clear() {
        synchronized(rays) {
            rays.clear()
        }
    }

    /**
     * Enable ray visualization
     */
    fun enable() {
        enabled = true
    }

    /**
     * Disable ray visualization and clear rays
     */
    fun disable() {
        enabled = false
        clear()
    }
}
