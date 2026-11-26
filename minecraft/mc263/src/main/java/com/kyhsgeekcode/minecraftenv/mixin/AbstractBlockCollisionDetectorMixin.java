package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.CollisionListener;
import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace;
import net.minecraft.core.BlockPos;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.InsideBlockEffectApplier;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockBehaviour;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(BlockBehaviour.BlockStateBase.class)
public abstract class AbstractBlockCollisionDetectorMixin {
    @Shadow
    public abstract Block getBlock();

    // 26.2 added InsideBlockEffectApplier + isPrecise params to entityInside; unused here,
    // same as mc121's onEntityCollision which only cared about the block/entity/position.
    @Inject(method = "entityInside", at = @At("HEAD"))
    public void entityInside(
            Level level,
            BlockPos pos,
            Entity entity,
            InsideBlockEffectApplier effectApplier,
            boolean isPrecise,
            CallbackInfo ci
    ) {
        if (!(entity instanceof Player))
            return;

        String blockName = this.getBlock().getDescriptionId();
        if (CollisionListener.Companion.getBlockCollisionInfoSet().contains(blockName)) {
            ObservationSpace.BlockCollisionInfo info = ObservationSpace.BlockCollisionInfo.newBuilder()
                    .setX(pos.getX())
                    .setY(pos.getY())
                    .setZ(pos.getZ())
                    .setBlockName(blockName)
                    .build();
            CollisionListener.Companion.getBlockCollisionInfo().add(info);
        }
    }
}
