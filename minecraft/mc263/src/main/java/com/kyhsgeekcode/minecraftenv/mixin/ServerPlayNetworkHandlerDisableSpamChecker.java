package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.server.network.ServerGamePacketListenerImpl;
import net.minecraft.util.TickThrottler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

// mc121's checkForSpam was split into detectChatRateSpam/detectCommandRateSpam, which both
// funnel through this one private method - cancelling here covers both in one hook.
@Mixin(ServerGamePacketListenerImpl.class)
public class ServerPlayNetworkHandlerDisableSpamChecker {
  @Inject(method = "detectRateSpam", at = @At("HEAD"), cancellable = true)
  private void detectRateSpam(TickThrottler throttler, CallbackInfo ci) {
    ci.cancel();
  }
}
