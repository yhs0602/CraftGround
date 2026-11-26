package com.kyhsgeekcode.minecraftenv.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(com.mojang.blaze3d.systems.RenderSystem.class)
public interface RenderSystemPollEventsInvoker {
  @Invoker("pollEvents")
  static void pollEvents() {}
}
