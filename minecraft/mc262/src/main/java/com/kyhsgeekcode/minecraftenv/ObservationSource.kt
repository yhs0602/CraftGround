package com.kyhsgeekcode.minecraftenv

import net.minecraft.server.level.ServerPlayer

/**
 * Seam A (26_2_phase2_plan.md §6.3), the source of the numeric (non-image) observation values.
 *
 * W11's original intent was that these must not be read off the client player — see §1.3 for the
 * race with packet delivery — and that [ServerAuthoritativeSource] should be used instead, since
 * the step barrier's happens-before makes the server copy safe to read.
 *
 * That is currently NOT the implementation in use. End-to-end testing on 26.2 found the
 * ServerPlayer's state never tracks the client at all (a camera action that rotated the client to
 * yaw=29.85 left serverYaw=0.0 indefinitely), because client→server player sync is not yet working
 * in this port — so a server-sourced observation reports values frozen at spawn. The client-side
 * `ClientPredictedSource` (in the `client` source set, as `LocalPlayer` is client-only) is wired up
 * instead, matching mc121's long-shipped semantics.
 *
 * [ServerAuthoritativeSource] is kept so this can be flipped back in one line once the sync gap is
 * fixed and the switch can actually be re-verified end to end.
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
