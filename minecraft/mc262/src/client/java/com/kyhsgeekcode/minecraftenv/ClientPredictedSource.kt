package com.kyhsgeekcode.minecraftenv

import net.minecraft.client.player.LocalPlayer

/**
 * Client-side [ObservationSource], reading the client-predicted [LocalPlayer] - the same values
 * mc121 has always reported.
 *
 * This lives in the `client` source set rather than next to [ServerAuthoritativeSource] in `main`
 * because [LocalPlayer] is a client-only class.
 *
 * See the comment at the observation-source selection in `MinecraftEnv.sendObservation` for why
 * this, rather than [ServerAuthoritativeSource], is currently the one that gets used.
 */
class ClientPredictedSource(
    private val player: LocalPlayer,
) : ObservationSource {
    override val x: Double get() = player.x
    override val y: Double get() = player.y
    override val z: Double get() = player.z
    override val prevX: Double get() = player.xo
    override val prevY: Double get() = player.yo
    override val prevZ: Double get() = player.zo
    override val pitch: Float get() = player.xRot
    override val yaw: Float get() = player.yRot
    override val health: Float get() = player.health
    override val foodLevel: Int get() = player.foodData.foodLevel
    override val saturationLevel: Float get() = player.foodData.saturationLevel
    override val isDead: Boolean get() = player.isDeadOrDying
}
