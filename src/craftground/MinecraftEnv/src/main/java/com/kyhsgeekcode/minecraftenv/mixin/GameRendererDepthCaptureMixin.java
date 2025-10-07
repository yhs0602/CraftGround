package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.FramebufferCapturer;
import com.kyhsgeekcode.minecraftenv.GameRendererDepthCaptureMixinGetterInterface;
import com.mojang.blaze3d.systems.RenderSystem;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.render.RenderTickCounter;
import net.minecraft.client.util.Window;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import static com.kyhsgeekcode.minecraftenv.PrintWithTimeKt.printWithTime;


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
            RenderSystem.recordRenderCall(() -> {
                if (FramebufferCapturer.INSTANCE.checkGLEW()) {
                    printWithTime("GLEW initialized");
                } else {
                    printWithTime("GLEW not initialized");
                    throw new RuntimeException("GLEW not initialized");
                }
                MinecraftClient client = MinecraftClient.getInstance();
                Window window = client.getWindow();
                org.lwjgl.opengl.GL.createCapabilities();
                int textureWidth = window.getFramebufferWidth();
                int textureHeight = window.getFramebufferHeight();
                int fbo = client.getFramebuffer().fbo;

                if (!RenderSystem.isOnRenderThread()) {
                    throw new IllegalStateException("Call on render thread");
                }

                lastDepthBuffer = FramebufferCapturer.INSTANCE.captureDepthImpl(
                        fbo,
                        textureWidth,
                        textureHeight,
                        FramebufferCapturer.INSTANCE.getRequiresDepthConversion(),
                        0.05f,
                        client.options.getViewDistance().getValue() * 4.0f
                );
            });
        }
    }

    @Override
    public float[] minecraftEnv$getLastDepthBuffer() {
        return lastDepthBuffer;
    }
}
