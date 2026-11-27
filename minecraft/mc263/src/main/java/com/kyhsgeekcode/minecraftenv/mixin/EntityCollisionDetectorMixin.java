package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.CollisionListener;
import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.player.Player;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Player.class)
public abstract class EntityCollisionDetectorMixin {
  // Player.touch dispatches to Entity.playerTouch, including overrides such as ItemEntity's.
  @Inject(method = "touch", at = @At("HEAD"))
  private void recordTouch(Entity entity, CallbackInfo ci) {
    String translationKey = entity.getType().getDescriptionId();
    if (CollisionListener.Companion.getEntityCollisionInfoSet().contains(translationKey)) {
      CollisionListener.Companion.getEntityCollisionInfo()
          .add(
              ObservationSpace.EntityCollisionInfo.newBuilder()
                  .setX((float) entity.getX())
                  .setY((float) entity.getY())
                  .setZ((float) entity.getZ())
                  .setEntityName(translationKey)
                  .build());
    }
  }
}
