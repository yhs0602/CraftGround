package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Overwrite;

@Mixin(Minecraft.class)
public class ClientIsWindowFocusedMixin {
  @Overwrite
  public boolean isWindowActive() {
    return true;
  }
}
