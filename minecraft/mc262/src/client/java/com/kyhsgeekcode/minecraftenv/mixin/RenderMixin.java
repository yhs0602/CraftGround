package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.MinecraftEnv;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

// Port of mc121's RenderMixin (see docs/26_2_phase2_plan.md W5/W1). mc121 redirected
// Framebuffer.endWrite()/draw()/Window.swapBuffers() inside MinecraftClient.render(); 26.2
// restructured this enough that the equivalent "skip the screen presentation" trick is actively
// harmful, so the capture hook is attached differently here. Both facts below were established by
// disassembling the real 26.2 Minecraft.renderFrame(boolean) and by end-to-end testing:
//
//  1. renderFrame()'s ENTIRE body is wrapped in `if (!this.windowSurface.isAcquired())`, and
//     GpuSurface.present() is what RELEASES an acquired surface. So suppressing present() is not
//     the harmless "don't show it on screen" no-op it was in mc121 - it permanently wedges the
//     surface in the acquired state, after which every later renderFrame() call returns
//     immediately without rendering anything at all. That is what broke world creation:
//     Minecraft.doWorldLoad()'s blocking wait loop (`while (!singleplayerServer.isReady() ||
//     gui.overlay() != null)`) drives progress by calling renderFrame(false) itself, so once
//     renderFrame went inert the loop spun forever - reproduced twice and confirmed via jstack
//     showing the render thread permanently parked there. Adding a GLFW pollEvents() call did NOT
//     help, because event starvation was never the cause.
//
//  2. Both `blitFromTexture(...)` and `present()` are themselves guarded by
//     `if (this.windowSurface.isAcquired())`, and the surface is only ever acquired when
//     `!surfaceIsInvalid && !window.isMinimized()`. EnvironmentInitializer iconifies the window on
//     purpose, so in a normal headless run present() is never invoked in the first place - making
//     it useless as a capture hook. This is why exactly one observation was produced (during world
//     load, before the window got iconified) and every subsequent step deadlocked.
//
// Hence: don't touch blitFromTexture()/present() at all - vanilla already skips both while the
// window is minimized - and hang the capture off CommandEncoder.submit() instead. submit() is
// called unconditionally, exactly once per renderFrame(), and injecting immediately AFTER it still
// satisfies W1's ordering requirement (phase2_plan.md §2.2) that GPU commands be submitted before
// MinecraftEnv reads the color texture.
@Mixin(Minecraft.class)
public class RenderMixin {
  @Redirect(
      method = "renderFrame",
      at =
          @At(
              value = "INVOKE",
              target = "Lnet/minecraft/client/FramerateLimiter;limitDisplayFPS(I)V"))
  private void skipFramerateLimit(int framerateLimit) {
    // do nothing - a headless step must never sleep waiting for a display refresh
  }

  @Inject(
      method = "renderFrame",
      at =
          @At(
              value = "INVOKE",
              target = "Lcom/mojang/blaze3d/systems/CommandEncoder;submit()V",
              shift = At.Shift.AFTER))
  private void captureAfterSubmit(boolean advanceGameTime, CallbackInfo ci) {
    // W1 (26_2_phase2_plan.md §2.2): the frame boundary the capture point moves to. Fires on every
    // frame; MinecraftEnv.onPresentCapture() is a no-op unless the preceding END_LEVEL_TICK marked
    // an observation as pending, so world-load frames (and any frame outside a step) cost nothing.
    MinecraftEnv.onPresentCapture();
  }
}
