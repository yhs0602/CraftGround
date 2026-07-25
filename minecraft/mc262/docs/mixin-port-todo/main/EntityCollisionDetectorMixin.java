package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.CollisionListener;
import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityType;
import net.minecraft.entity.player.PlayerEntity;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Entity.class)
public abstract class EntityCollisionDetectorMixin {
    @Shadow
    public abstract EntityType<?> getType();

    @Shadow
    public abstract double getX();

    @Shadow
    public abstract double getY();

    @Shadow
    public abstract double getZ();

    @Inject(method = "onPlayerCollision", at = @At("HEAD"))
    public void onPlayerCollision(PlayerEntity player, CallbackInfo ci) {
//        System.out.println("EntityCollisionDetectorMixin.onPlayerCollision called for " + getType().getTranslationKey());
        // Get the player's position
        String translationKey = getType().getTranslationKey();
        if (CollisionListener.Companion.getEntityCollisionInfoSet().contains(translationKey)) {
            CollisionListener.Companion.getEntityCollisionInfo().add(
                    ObservationSpace.EntityCollisionInfo.newBuilder()
                            .setX((float) getX())
                            .setY((float) getY())
                            .setZ((float) getZ())
                            .setEntityName(translationKey)
                            .build()
            );
        }
    }
}
