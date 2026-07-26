package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.DeltaTracker;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(DeltaTracker.Timer.class)
public interface DeltaTrackerTimerAccessor {
  @Accessor("deltaTicks")
  void setDeltaTicks(float deltaTicks);

  @Accessor("lastMs")
  void setLastMs(long lastMs);

  @Accessor("deltaTickResidual")
  void setDeltaTickResidual(float deltaTickResidual);
}
