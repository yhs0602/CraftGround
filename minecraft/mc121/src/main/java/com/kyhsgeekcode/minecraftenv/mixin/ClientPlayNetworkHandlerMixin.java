package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.GetMessagesInterface;
import net.minecraft.client.network.ClientPlayNetworkHandler;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.network.packet.s2c.play.DeathMessageS2CPacket;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ClientPlayNetworkHandler.class)
public class ClientPlayNetworkHandlerMixin implements GetMessagesInterface {
  @Shadow private ClientWorld world;

  @Inject(method = "onDeathMessage", at = @At("HEAD"), cancellable = false)
  public void onDeathMessage(DeathMessageS2CPacket packet, CallbackInfo ci) {
    Entity entity = this.world.getEntityById(packet.playerId());
    if (entity == ((ClientCommonNetworkHandlerClientAccessor) this).getClient().player) {
      var message = packet.message();
      this.lastDeathMessage.clear();
      this.lastDeathMessage.add(message.getString());
    }
  }
}
