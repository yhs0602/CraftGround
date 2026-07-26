package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

// Port of mc121's RenderMixin (see docs/26_2_phase2_plan.md W5). mc121 redirected
// Framebuffer.endWrite()/draw()/Window.swapBuffers() inside MinecraftClient.render(); 26.2's
// Minecraft.renderFrame(boolean) blits the finished frame onto the window surface and blocks on
// present()/vsync instead, and additionally sleeps in FramerateLimiter.limitDisplayFPS() when a
// frame-rate cap is set.
//
// Unlike mc121, this mixin only neutralizes limitDisplayFPS - it does NOT skip
// GpuSurface.blitFromTexture()/present() (an earlier version of this mixin did, and that broke
// world creation: Minecraft.doWorldLoad()'s own blocking wait loop
// (`while (!singleplayerServer.isReady() || gui.overlay() != null)`) also calls
// renderFrame(false) directly on every iteration, and skipping present() there left that wait
// condition unsatisfied forever - confirmed via jstack showing the render thread permanently
// parked inside that loop with RenderMixin's blit/present redirects active, and the smoke test
// passing end-to-end as soon as they were removed). Real, correctness-preserving frame-skipping
// needs the present()-as-capture-hook redesign from W1, not a blind no-op.
@Mixin(Minecraft.class)
public class RenderMixin {
  @Redirect(
      method = "renderFrame",
      at = @At(value = "INVOKE", target = "Lnet/minecraft/client/FramerateLimiter;limitDisplayFPS(I)V"))
  private void skipFramerateLimit(int framerateLimit) {
    // do nothing - a headless step must never sleep waiting for a display refresh
  }
}
