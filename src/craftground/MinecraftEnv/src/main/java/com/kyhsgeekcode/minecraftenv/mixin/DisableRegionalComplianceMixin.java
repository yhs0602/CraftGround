package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.resource.PeriodicNotificationManager;
import net.minecraft.resource.ReloadableResourceManagerImpl;
import net.minecraft.resource.ResourceReloader;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ReloadableResourceManagerImpl.class)
public class DisableRegionalComplianceMixin {
  @Inject(method = "registerReloader", at = @At("HEAD"), cancellable = true)
  private void registerReloader(ResourceReloader reloader, CallbackInfo ci) {
    if (reloader instanceof PeriodicNotificationManager) {
      ci.cancel(); // cancel the reload
    }
  }
}
