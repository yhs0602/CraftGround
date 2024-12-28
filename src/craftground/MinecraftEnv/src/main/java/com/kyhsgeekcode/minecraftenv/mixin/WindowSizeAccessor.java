package com.kyhsgeekcode.minecraftenv.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(net.minecraft.client.util.Window.class)
public interface WindowSizeAccessor {
  @Accessor("windowedWidth")
  int getWindowedWidth();

  @Accessor("windowedHeight")
  int getWindowedHeight();
}
