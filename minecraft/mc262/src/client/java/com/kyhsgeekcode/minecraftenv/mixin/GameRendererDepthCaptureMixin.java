package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.FramebufferCapturer;
import com.kyhsgeekcode.minecraftenv.GameRendererDepthCaptureMixinGetterInterface;
import com.mojang.blaze3d.opengl.GlTexture;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.textures.GpuTexture;
import net.minecraft.client.Camera;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * 26.2 port of mc121's depth-capture mixin (docs/26_2_phase2_plan.md W3, depth path).
 *
 * <p>The injection point is load-bearing and is the reason this can't ride along with the color
 * capture in RenderMixin. GameRenderer.render() calls {@code
 * CommandEncoder.clearDepthTexture(mainRenderTarget().getDepthTexture(), 0.0)} immediately after
 * the level pass and before the GUI pass, so by the time the frame reaches {@code
 * CommandEncoder.submit()} - where color is captured - the world depth is already gone. Depth has
 * to be read while it still exists, right after renderLevel().
 *
 * <p>mc121 wrapped its capture in {@code RenderSystem.recordRenderCall}; that isn't needed here
 * because GameRenderer.render() already runs on the render thread inside Minecraft.renderFrame. The
 * assertion is kept as a cheap guard.
 */
@Mixin(GameRenderer.class)
public class GameRendererDepthCaptureMixin implements GameRendererDepthCaptureMixinGetterInterface {
  @Unique private float[] minecraftEnv$lastDepthBuffer = null;

  @Inject(
      method = "render",
      at =
          @At(
              value = "INVOKE",
              target =
                  "Lnet/minecraft/client/renderer/GameRenderer;renderLevel(Lnet/minecraft/client/DeltaTracker;)V",
              shift = At.Shift.AFTER))
  private void minecraftEnv$captureDepthAfterLevel(
      DeltaTracker deltaTracker, boolean advanceGameTime, CallbackInfo ci) {
    if (!FramebufferCapturer.INSTANCE.getShouldCaptureDepth()) {
      return;
    }
    if (!RenderSystem.isOnRenderThread()) {
      throw new IllegalStateException("Depth capture must run on the render thread");
    }
    if (!FramebufferCapturer.INSTANCE.checkGLEW()) {
      throw new IllegalStateException("GLEW not initialized");
    }

    Minecraft client = Minecraft.getInstance();
    RenderTarget mainRenderTarget = client.gameRenderer.mainRenderTarget();
    GpuTexture depthTexture = mainRenderTarget.getDepthTexture();
    if (!(depthTexture instanceof GlTexture glDepthTexture)) {
      // Same fail-fast rationale as the color path (phase2_plan.md D2/W4): a non-GL backend
      // would silently produce garbage instead of depth.
      throw new IllegalStateException(
          "Expected an OpenGL depth texture on the main render target (got "
              + depthTexture
              + "); depth capture requires the GL backend (see phase2_plan.md D2).");
    }

    // 26.2 computes the level's far plane per frame as max(renderDistance * 4, cloudRange * 16)
    // (Camera.update) and publishes it on the extracted camera state; the near plane is the
    // Camera.PROJECTION_Z_NEAR constant that Camera.setupPerspective passes. mc121 hardcoded
    // viewDistance * 4, which no longer matches when the cloud range dominates.
    float farPlane =
        client.gameRenderer.gameRenderState().levelRenderState.cameraRenderState.depthFar;
    boolean zZeroToOne = RenderSystem.getDevice().getDeviceInfo().isZZeroToOne();

    minecraftEnv$lastDepthBuffer =
        FramebufferCapturer.INSTANCE.captureDepthImpl(
            glDepthTexture.glId(),
            mainRenderTarget.width,
            mainRenderTarget.height,
            FramebufferCapturer.INSTANCE.getRequiresDepthConversion(),
            Camera.PROJECTION_Z_NEAR,
            farPlane,
            zZeroToOne);
  }

  @Override
  public float[] minecraftEnv$getLastDepthBuffer() {
    return minecraftEnv$lastDepthBuffer;
  }
}
