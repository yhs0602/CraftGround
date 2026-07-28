package com.kyhsgeekcode.minecraftenv

import net.minecraft.server.level.ServerPlayer

/**
 * Seam A (26_2_phase2_plan.md §6.3), the source of the numeric (non-image) observation values.
 *
 * W11's original intent was that these must not be read off the client player — see §1.3 for the
 * race with packet delivery — and that [ServerAuthoritativeSource] should be used instead, since
 * the step barrier's happens-before makes the server copy safe to read.
 *
 * That is NOT the implementation currently in use. The client-side `ClientPredictedSource` (in the
 * `client` source set, since `LocalPlayer` is client-only) is wired up instead, because a
 * server-sourced value structurally cannot satisfy W1's same-step guarantee: camera rotation is
 * applied by `Minecraft.runTick`'s `handleAccumulatedMovement()` *after* the tick loop that ran
 * `LocalPlayer.sendPosition()`, so the server only learns about it on the following tick. Reading
 * yaw from the server would report the pre-action value and reintroduce the very off-by-one-step
 * observation W1 exists to remove. This also matches mc121's long-shipped semantics.
 *
 * [ServerAuthoritativeSource] is kept so switching is a one-line change if that tradeoff is ever
 * wanted. See the selection site in `MinecraftEnv.sendObservation` for the full rationale.
 */
interface ObservationSource {
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

class ServerAuthoritativeSource(
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
