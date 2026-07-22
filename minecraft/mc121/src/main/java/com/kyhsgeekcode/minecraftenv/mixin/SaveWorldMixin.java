package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

@Mixin(MinecraftServer.class)
public class SaveWorldMixin {
  @Redirect(
      method = "tick",
      at = @At(value = "INVOKE", target = "Lnet/minecraft/server/MinecraftServer;saveAll(ZZZ)Z"))
  private boolean saveAll(MinecraftServer server, boolean bl, boolean bl2, boolean bl3) {
    return false; // disable saving
  }
}
