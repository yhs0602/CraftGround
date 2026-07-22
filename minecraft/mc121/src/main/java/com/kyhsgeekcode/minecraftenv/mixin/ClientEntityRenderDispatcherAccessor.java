package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.render.entity.EntityRenderDispatcher;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(net.minecraft.client.MinecraftClient.class)
public interface ClientEntityRenderDispatcherAccessor {
  @Accessor("entityRenderDispatcher")
  EntityRenderDispatcher getEntityRenderDispatcher();
}
