package com.kyhsgeekcode.minecraftenv.mixin;

import java.util.Queue;
import net.minecraft.network.PacketProcessor;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

// W12 (docs/26_2_phase2_plan.md): packetsToBeHandled has no public getter or size accessor, but
// W1-a's drain point (MinecraftEnv.kt, END_LEVEL_TICK) needs its size to measure staleness.
@Mixin(PacketProcessor.class)
public interface PacketProcessorQueueAccessor {
  @Accessor("packetsToBeHandled")
  Queue<?> minecraftEnv$getPacketsToBeHandled();
}
