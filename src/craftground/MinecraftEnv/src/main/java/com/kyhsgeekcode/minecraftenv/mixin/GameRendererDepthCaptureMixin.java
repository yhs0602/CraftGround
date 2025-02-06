package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.FramebufferCapturer;
import com.kyhsgeekcode.minecraftenv.GameRendererDepthCaptureMixinGetterInterface;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.render.RenderTickCounter;
import net.minecraft.client.util.Window;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;


@Mixin(net.minecraft.client.render.GameRenderer.class)
public class GameRendererDepthCaptureMixin implements GameRendererDepthCaptureMixinGetterInterface {
    @Unique
    private float[] lastDepthBuffer = null;

    @Inject(
            method = "renderWorld",
            at = @At(
                    value = "INVOKE",
                    target = "Lnet/minecraft/client/render/WorldRenderer;render(Lnet/minecraft/client/render/RenderTickCounter;ZLnet/minecraft/client/render/Camera;Lnet/minecraft/client/render/GameRenderer;Lnet/minecraft/client/render/LightmapTextureManager;Lorg/joml/Matrix4f;Lorg/joml/Matrix4f;)V",
                    shift = At.Shift.AFTER
            )
    )
    private void afterRenderWorld(
            RenderTickCounter tickCounter,
            CallbackInfo ci
    ) {
        if (FramebufferCapturer.INSTANCE.getShouldCaptureDepth()) {
            MinecraftClient client = MinecraftClient.getInstance();
            Window window = client.getWindow();
            int textureWidth = window.getFramebufferWidth();
            int textureHeight = window.getFramebufferHeight();
            int fbo = client.getFramebuffer().fbo;

            lastDepthBuffer = FramebufferCapturer.INSTANCE.captureDepthImpl(
                    fbo,
                    textureWidth,
                    textureHeight,
                    FramebufferCapturer.INSTANCE.getRequiresDepthConversion(),
                    0.05f,
                    client.options.getViewDistance().getValue() * 4.0f
            );
        }
    }

    @Override
    public float[] minecraftEnv$getLastDepthBuffer() {
        return lastDepthBuffer;
    }
}
