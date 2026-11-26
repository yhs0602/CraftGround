package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.MouseHandler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(MouseHandler.class)
public interface MouseXYAccessor {
  @Accessor("xpos")
  void setX(double x);

  @Accessor("ypos")
  void setY(double y);
}
