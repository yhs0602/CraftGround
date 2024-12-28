package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.Mouse;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(Mouse.class)
public interface MouseXYAccessor {
  @Accessor("x")
  void setX(double x);

  @Accessor("y")
  void setY(double y);
}
