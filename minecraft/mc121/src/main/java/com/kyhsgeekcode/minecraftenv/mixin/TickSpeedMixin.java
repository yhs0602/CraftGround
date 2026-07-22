package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.render.RenderTickCounter;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

@Mixin(RenderTickCounter.Dynamic.class)
public class TickSpeedMixin {
  @Redirect(
      method = "beginRenderTick(JZ)I",
      at =
          @At(
              value = "INVOKE",
              target =
                  "Lnet/minecraft/client/render/RenderTickCounter$Dynamic;beginRenderTick(J)I"))
  private int beginRenderTick(RenderTickCounter.Dynamic renderTickCounter, long timeMillis) {
    ((RenderTickCounterAccessor) renderTickCounter)
        .setLastFrameDuration(1); // (float)(timeMillis - this.prevTimeMillis) / this.tickTime;
    ((RenderTickCounterAccessor) renderTickCounter).setPrevTimeMillis(timeMillis);
    ((RenderTickCounterAccessor) renderTickCounter)
        .setTickDelta(
            renderTickCounter.getTickDelta(true) + renderTickCounter.getLastFrameDuration());
    int i = (int) renderTickCounter.getTickDelta(true);
    ((RenderTickCounterAccessor) renderTickCounter)
        .setTickDelta((renderTickCounter.getTickDelta(true) - (float) i));
    return i;
  }
}
