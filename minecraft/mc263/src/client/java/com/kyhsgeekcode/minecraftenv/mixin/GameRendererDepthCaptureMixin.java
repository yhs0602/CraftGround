package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.Blaze3dCapture;
import com.kyhsgeekcode.minecraftenv.FramebufferCapturer;
import com.kyhsgeekcode.minecraftenv.GameRendererDepthCaptureMixinGetterInterface;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.renderpearl.api.textures.GpuTexture;
import com.mojang.renderpearl.backend.opengl.GlTexture;
import net.minecraft.client.Camera;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Read world depth before the 3D HUD clears the main depth texture. */
@Mixin(GameRenderer.class)
public class GameRendererDepthCaptureMixin implements GameRendererDepthCaptureMixinGetterInterface {
  @Unique private float[] minecraftEnv$lastDepthBuffer = null;

  @Inject(
      method = "renderLevel",
      at =
          @At(
              value = "INVOKE",
              target =
                  "Lnet/minecraft/client/renderer/GameRenderer;render3dHud(Lnet/minecraft/client/renderer/state/level/CameraRenderState;Lnet/minecraft/client/renderer/state/level/PlayerRenderState;Lnet/minecraft/client/renderer/state/OptionsRenderState;Z)V"))
  private void minecraftEnv$captureWorldDepth(CallbackInfo ci) {
    if (!FramebufferCapturer.INSTANCE.getShouldCaptureDepth()) {
      return;
    }
    if (!RenderSystem.isOnRenderThread()) {
      throw new IllegalStateException("Depth capture must run on the render thread");
    }

    Minecraft client = Minecraft.getInstance();
    RenderTarget mainRenderTarget = client.gameRenderer.mainRenderTarget();
    GpuTexture depthTexture = mainRenderTarget.getDepthTexture();
    if (depthTexture == null) {
      throw new IllegalStateException("Main render target has no depth texture to capture");
    }

    // 26.2 computes the level's far plane per frame as max(renderDistance * 4, cloudRange * 16)
    // (Camera.update) and publishes it on the extracted camera state; the near plane is the
    // Camera.PROJECTION_Z_NEAR constant that Camera.setupPerspective passes. mc121 hardcoded
    // viewDistance * 4, which no longer matches when the cloud range dominates.
    float farPlane =
        client.gameRenderer.gameRenderState().levelRenderState.cameraRenderState.depthFar;
    boolean zZeroToOne = RenderSystem.getDevice().getDeviceInfo().isZZeroToOne();

    if (Blaze3dCapture.INSTANCE.backendFor(depthTexture) == Blaze3dCapture.CaptureBackend.BLAZE3D) {
      // Backend-neutral path: only *record* the copy here. Reading it would need a fence, and a
      // fence armed mid-frame would name the submission this very command is being written into -
      // which submit() has not issued yet. MinecraftEnv maps it after the frame's submit()
      // instead (Blaze3dCapture.readPendingDepth). The injection point still has to be here,
      // because the copy command has to precede the clearDepthTexture() that follows.
      Blaze3dCapture.INSTANCE.recordDepthReadback(
          depthTexture, Camera.PROJECTION_Z_NEAR, farPlane, zZeroToOne);
      return;
    }

    if (!FramebufferCapturer.INSTANCE.checkGLEW()) {
      throw new IllegalStateException("GLEW not initialized");
    }
    minecraftEnv$lastDepthBuffer =
        FramebufferCapturer.INSTANCE.captureDepthImpl(
            ((GlTexture) depthTexture).glId(),
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
