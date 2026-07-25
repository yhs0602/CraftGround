package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.CollisionListener;
import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.player.Player;
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

    @Inject(method = "playerTouch", at = @At("HEAD"))
    public void playerTouch(Player player, CallbackInfo ci) {
        String translationKey = getType().getDescriptionId();
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
