package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.render.RenderTickCounter;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(RenderTickCounter.Dynamic.class)
public interface RenderTickCounterAccessor {
  @Accessor("prevTimeMillis")
  void setPrevTimeMillis(long prevTimeMillis);

  @Accessor("lastFrameDuration")
  void setLastFrameDuration(float lastFrameDuration);

  @Accessor("tickDelta")
  void setTickDelta(float tickDelta);
}
