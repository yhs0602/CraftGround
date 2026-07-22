package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.MinecraftClient;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(MinecraftClient.class)
public interface ClientRenderInvoker {
  @Invoker("render")
  void invokeRender(boolean tick);
}
