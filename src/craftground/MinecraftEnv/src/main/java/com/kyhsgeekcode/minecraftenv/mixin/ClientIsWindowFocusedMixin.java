package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.MinecraftClient;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Overwrite;

@Mixin(MinecraftClient.class)
public class ClientIsWindowFocusedMixin {
  @Overwrite
  public boolean isWindowFocused() {
    return true;
  }
}
