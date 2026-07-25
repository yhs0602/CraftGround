package com.kyhsgeekcode.minecraftenv.mixin;

import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuSurface;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.textures.GpuTextureView;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

// Port of mc121's RenderMixin (see docs/26_2_phase2_plan.md W5). mc121 redirected
// Framebuffer.endWrite()/draw()/Window.swapBuffers() inside MinecraftClient.render(); 26.2's
// Minecraft.renderFrame(boolean) blits the finished frame onto the window surface and blocks on
// present()/vsync instead, and additionally sleeps in FramerateLimiter.limitDisplayFPS() when a
// frame-rate cap is set - all three must be neutralized so a headless step isn't rate-limited or
// blocked waiting on a window that may not even be visible.
//
// NOTE: per plan W1, windowSurface.present() is also meant to become the capture hook point
// (moving sendObservation() off the END_WORLD_TICK callback). That wiring is intentionally not
// done here - it depends on the still-open W11/W12 staleness-measurement decision - so this
// mixin only does the mechanical skip for now.
@Mixin(Minecraft.class)
public class RenderMixin {
  @Redirect(
      method = "renderFrame",
      at =
          @At(
              value = "INVOKE",
              target =
                  "Lcom/mojang/blaze3d/systems/GpuSurface;blitFromTexture(Lcom/mojang/blaze3d/systems/CommandEncoder;Lcom/mojang/blaze3d/textures/GpuTextureView;)V"))
  private void skipBlitFromTexture(GpuSurface instance, CommandEncoder commandEncoder, GpuTextureView textureView) {
    // do nothing
  }

  @Redirect(
      method = "renderFrame",
      at = @At(value = "INVOKE", target = "Lcom/mojang/blaze3d/systems/GpuSurface;present()V"))
  private void skipPresent(GpuSurface instance) {
    // mc121 also cleared a queued-render-call buffer and the immediate-mode Tessellator here;
    // both concepts are gone in 26.2's GPU-command-encoder model, so polling window events is
    // all that's left to replicate.
    RenderSystem.pollEvents();
  }

  @Redirect(
      method = "renderFrame",
      at = @At(value = "INVOKE", target = "Lnet/minecraft/client/FramerateLimiter;limitDisplayFPS(I)V"))
  private void skipFramerateLimit(int framerateLimit) {
    // do nothing - a headless step must never sleep waiting for a display refresh
  }
}
