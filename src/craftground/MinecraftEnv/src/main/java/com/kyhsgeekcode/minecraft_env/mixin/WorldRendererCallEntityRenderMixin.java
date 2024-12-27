package com.kyhsgeekcode.minecraft_env.mixin;

import com.kyhsgeekcode.minecraft_env.AddListenerInterface;
import com.kyhsgeekcode.minecraft_env.EntityRenderListener;
import net.minecraft.client.render.VertexConsumerProvider;
import net.minecraft.client.util.math.MatrixStack;
import net.minecraft.entity.Entity;
import org.jetbrains.annotations.NotNull;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.ArrayList;
import java.util.List;

@Mixin(net.minecraft.client.render.WorldRenderer.class)
public class WorldRendererCallEntityRenderMixin implements AddListenerInterface {
    List<EntityRenderListener> listeners = new ArrayList<>();

    @Inject(method = "renderEntity", at = @At(value = "RETURN"))
    private void callOnEntityRender(Entity entity, double cameraX, double cameraY, double cameraZ, float tickDelta, MatrixStack matrices, VertexConsumerProvider vertexConsumers, CallbackInfo info) {
        for (EntityRenderListener listener : listeners) {
            listener.onEntityRender(entity);
        }
    }

    @Override
    public void addRenderListener(@NotNull EntityRenderListener listener) {
        listeners.add(listener);
    }

    @Override
    public List<EntityRenderListener> getRenderListeners() {
        return listeners;
    }
}
