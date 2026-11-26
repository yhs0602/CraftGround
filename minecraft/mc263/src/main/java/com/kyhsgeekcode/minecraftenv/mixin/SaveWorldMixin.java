package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

@Mixin(MinecraftServer.class)
public class SaveWorldMixin {
  // 26.2 moved the saveEverything() call mc121 targeted (inside tick()) into a private
  // autoSave() helper; MinecraftServer.tick() no longer exists at all (renamed/split into
  // tickServer/tickChildren/processPacketsAndTick).
  @Redirect(
      method = "autoSave",
      at = @At(value = "INVOKE", target = "Lnet/minecraft/server/MinecraftServer;saveEverything(ZZZ)Z"))
  private boolean saveEverything(MinecraftServer server, boolean bl, boolean bl2, boolean bl3) {
    return false; // disable saving
  }
}
