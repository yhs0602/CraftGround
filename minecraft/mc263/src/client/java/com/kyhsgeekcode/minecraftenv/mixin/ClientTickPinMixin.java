package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.DeltaTracker;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

// 26_2_phase2_plan.md W2 / §1.2 (D5'). Minecraft.getTickTargetMillis() floors client tick
// pacing at 50ms (Math.max(defaultTickTargetMillis, ...)) regardless of the server's
// TickRateManager rate, so raising only the server rate leaves the client stuck at
// 20 ticks/s (wall-clock T < 50ms => ticksToDo == 0 most frames) or, on a slow step,
// consuming more than one action per observation (T > 50ms => ticksToDo > 1). Pinning
// advanceGameTime to always report exactly one tick restores the
// 1 action = 1 tick = 1 render = 1 observation invariant mc121's TickSpeedMixin gave us,
// and (as a side effect) zeroing the residual makes getGameTimeDeltaPartialTick() return 0,
// i.e. render exactly on the tick boundary with no interpolation.
@Mixin(DeltaTracker.Timer.class)
public class ClientTickPinMixin {
  @Inject(method = "advanceGameTime", at = @At("HEAD"), cancellable = true)
  private void pinToOneTickPerFrame(long currentMs, CallbackInfoReturnable<Integer> cir) {
    DeltaTrackerTimerAccessor self = (DeltaTrackerTimerAccessor) this;
    self.setDeltaTicks(1.0f);
    self.setLastMs(currentMs);
    self.setDeltaTickResidual(0.0f);
    cir.setReturnValue(1);
  }
}
