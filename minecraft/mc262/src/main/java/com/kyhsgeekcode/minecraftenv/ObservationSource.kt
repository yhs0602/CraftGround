package com.kyhsgeekcode.minecraftenv

import net.minecraft.server.level.ServerPlayer

/**
 * Seam A (26_2_phase2_plan.md §6.3). Numeric observation values must not be read
 * directly off the client player — see §1.3 for why that races with packet delivery.
 * [ServerAuthoritativeSource] reads the authoritative [ServerPlayer] instead, which the
 * `TickSynchronizer` lock's happens-before already makes safe (W11).
 *
 * Only the IntegratedServer-backed implementation exists for now; a client-fallback
 * implementation is deferred to the multiplayer scoping in §6.5 (YAGNI, §6.4).
 */
internal interface ObservationSource {
    val x: Double
    val y: Double
    val z: Double
    val prevX: Double
    val prevY: Double
    val prevZ: Double
    val pitch: Float
    val yaw: Float
    val health: Float
    val foodLevel: Int
    val saturationLevel: Float
    val isDead: Boolean
}

internal class ServerAuthoritativeSource(
    private val serverPlayer: ServerPlayer,
) : ObservationSource {
    override val x: Double get() = serverPlayer.x
    override val y: Double get() = serverPlayer.y
    override val z: Double get() = serverPlayer.z
    override val prevX: Double get() = serverPlayer.xo
    override val prevY: Double get() = serverPlayer.yo
    override val prevZ: Double get() = serverPlayer.zo
    override val pitch: Float get() = serverPlayer.xRot
    override val yaw: Float get() = serverPlayer.yRot
    override val health: Float get() = serverPlayer.health
    override val foodLevel: Int get() = serverPlayer.foodData.foodLevel
    override val saturationLevel: Float get() = serverPlayer.foodData.saturationLevel
    override val isDead: Boolean get() = serverPlayer.isDeadOrDying
}
