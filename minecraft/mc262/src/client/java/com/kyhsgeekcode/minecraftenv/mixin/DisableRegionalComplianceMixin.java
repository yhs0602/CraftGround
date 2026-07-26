package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.PeriodicNotificationManager;
import net.minecraft.server.packs.resources.PreparableReloadListener;
import net.minecraft.server.packs.resources.ReloadableResourceManager;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ReloadableResourceManager.class)
public class DisableRegionalComplianceMixin {
  @Inject(method = "registerReloadListener", at = @At("HEAD"), cancellable = true)
  private void registerReloadListener(PreparableReloadListener reloader, CallbackInfo ci) {
    if (reloader instanceof PeriodicNotificationManager) {
      ci.cancel(); // cancel the reload
    }
  }
}
