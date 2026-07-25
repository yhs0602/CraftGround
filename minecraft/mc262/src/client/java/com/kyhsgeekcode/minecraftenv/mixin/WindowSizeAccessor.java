package com.kyhsgeekcode.minecraftenv.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(com.mojang.blaze3d.platform.Window.class)
public interface WindowSizeAccessor {
  @Accessor("windowedWidth")
  int getWindowedWidth();

  @Accessor("windowedHeight")
  int getWindowedHeight();
}
