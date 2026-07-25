package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.server.network.ServerPlayNetworkHandler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ServerPlayNetworkHandler.class)
public class ServerPlayNetworkHandlerDisableSpamChecker {
  @Inject(method = "checkForSpam", at = @At("HEAD"), cancellable = true)
  private void checkForSpam(CallbackInfo ci) {
    ci.cancel();
  }
}
