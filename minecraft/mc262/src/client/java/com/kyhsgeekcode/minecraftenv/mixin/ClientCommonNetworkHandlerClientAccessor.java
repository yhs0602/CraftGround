package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(net.minecraft.client.multiplayer.ClientCommonPacketListenerImpl.class)
public interface ClientCommonNetworkHandlerClientAccessor {
  @Accessor("minecraft")
  Minecraft getClient();
}
