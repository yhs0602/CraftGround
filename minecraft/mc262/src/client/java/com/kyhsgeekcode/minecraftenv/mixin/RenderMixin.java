package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.MinecraftEnv;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuSurface;
import com.mojang.blaze3d.textures.GpuTextureView;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

// Port of mc121's RenderMixin (see docs/26_2_phase2_plan.md W5/W1). mc121 redirected
// Framebuffer.endWrite()/draw()/Window.swapBuffers() inside MinecraftClient.render(); 26.2's
// Minecraft.renderFrame(boolean) blits the finished frame onto the window surface and presents
// it instead, and additionally sleeps in FramerateLimiter.limitDisplayFPS() when a frame-rate
// cap is set.
//
// An earlier version of this mixin blindly no-op'd GpuSurface.blitFromTexture()/present() (with
// no GLFW event pump) and that broke world creation: Minecraft.doWorldLoad()'s own blocking wait
// loop (`while (!singleplayerServer.isReady() || gui.overlay() != null)`) also calls
// renderFrame(false) directly on every iteration, and skipping present() there left that wait
// condition unsatisfied forever - confirmed via jstack showing the render thread permanently
// parked inside that loop, and the smoke test passing end-to-end as soon as the redirects were
// removed. W1 (phase2_plan.md §2.2) redesigns present() into the actual capture hook instead of
// a blind no-op, and - mirroring mc121's WindowSwapBuffers redirect, which always polled GLFW
// events at the equivalent point - keeps pumping GLFW events on every present() call so
// doWorldLoad()'s loop isn't starved.
@Mixin(Minecraft.class)
public class RenderMixin {
  @Redirect(
      method = "renderFrame",
      at = @At(value = "INVOKE", target = "Lnet/minecraft/client/FramerateLimiter;limitDisplayFPS(I)V"))
  private void skipFramerateLimit(int framerateLimit) {
    // do nothing - a headless step must never sleep waiting for a display refresh
  }

  @Redirect(
      method = "renderFrame",
      at =
          @At(
              value = "INVOKE",
              target =
                  "Lcom/mojang/blaze3d/systems/GpuSurface;blitFromTexture(Lcom/mojang/blaze3d/systems/CommandEncoder;Lcom/mojang/blaze3d/textures/GpuTextureView;)V"))
  private void skipBlitToScreen(GpuSurface instance, CommandEncoder encoder, GpuTextureView texture) {
    // do nothing - headless capture doesn't need the frame blitted onto the window surface.
  }

  @Redirect(
      method = "renderFrame",
      at = @At(value = "INVOKE", target = "Lcom/mojang/blaze3d/systems/GpuSurface;present()V"))
  private void captureInsteadOfPresent(GpuSurface instance) {
    // W1 (26_2_phase2_plan.md §2.2): this replaces the real present() call - the present-redirect
    // frame boundary the capture point moves to. Runs after
    // RenderSystem.getDevice().createCommandEncoder().submit() (earlier in renderFrame), so GPU
    // commands are guaranteed submitted before MinecraftEnv reads the color texture.
    RenderSystemPollEventsInvoker.pollEvents();
    MinecraftEnv.onPresentCapture();
  }
}
