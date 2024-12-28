package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.MinecraftClient;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(net.minecraft.client.network.ClientCommonNetworkHandler.class)
public interface ClientCommonNetworkHandlerClientAccessor {
  @Accessor("client")
  MinecraftClient getClient();
}
