package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.AddListenerInterface;
import net.minecraft.client.render.WorldRenderer;
import net.minecraft.entity.Entity;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

@Mixin(WorldRenderer.class)
public class ClientWorldMixin {
  @Redirect(
      method = "render",
      at =
          @At(
              value = "INVOKE",
              target = "Lnet/minecraft/client/world/ClientWorld;getEntities()Ljava/lang/Iterable;"))
  private Iterable<Entity> getEntities(net.minecraft.client.world.ClientWorld clientWorld) {
    var listeners = ((AddListenerInterface) this).getRenderListeners();
    for (var listener : listeners) {
      listener.clear();
    }
    return clientWorld.getEntities();
  }
}
