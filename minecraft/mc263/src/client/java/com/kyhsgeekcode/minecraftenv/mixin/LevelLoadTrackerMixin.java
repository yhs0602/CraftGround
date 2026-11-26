package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.LevelLoadTracker;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

// 26.2 added a client-loaded handshake that mc121 had no equivalent of, and it silently disables
// the player until it completes. LocalPlayer.tick() wraps its ENTIRE body in
// `if (this.connection.hasClientLoaded())`, so while that flag is false the player never runs
// super.tick() (hence no aiStep() -> no ClientInput.tick(), no movement) and never runs
// sendPosition() (hence the server's copy of the player stays frozen at spawn). That single flag
// accounted for every "actions do nothing" symptom seen end-to-end: held movement keys reached
// KeyMapping.isDown() correctly but the player never moved, and server-side position/rotation
// never tracked the client at all.
//
// ClientPacketListener only sets the flag once LevelLoadTracker.isLevelReady() returns true, and
// that walks a state machine ending in LevelLoadTracker$WaitingForPlayerChunk, whose readiness
// condition is `playerSectionReady` - an AtomicBoolean flipped by the level renderer when the
// chunk section containing the player finishes COMPILING. Gating gameplay on render-thread section
// compilation is meaningless for a headless agent (the window is deliberately iconified), and the
// preceding WaitingForServer state has no tick() override at all, so it only advances when the
// loading packets happen to arrive - the 30s "letting the player into the world anyway" timeout
// lives inside WaitingForPlayerChunk and does not cover it.
//
// So report ready as soon as a level and player actually exist. This reaches the same vanilla
// completion path (ClientPacketListener.notifyPlayerLoaded() -> ServerboundPlayerLoadedPacket +
// setClientLoaded(true)) that the vanilla timeout would eventually reach, just without waiting on
// the renderer. Guarding on level/player being non-null keeps us from announcing "loaded" before
// the world is actually joined.
@Mixin(LevelLoadTracker.class)
public class LevelLoadTrackerMixin {
  @Inject(method = "isLevelReady", at = @At("HEAD"), cancellable = true)
  private void reportReadyOnceWorldExists(CallbackInfoReturnable<Boolean> cir) {
    Minecraft client = Minecraft.getInstance();
    if (client.level != null && client.player != null) {
      cir.setReturnValue(true);
    }
  }
}
