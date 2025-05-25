package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.CollisionListener;
import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace;
import net.minecraft.block.AbstractBlock;
import net.minecraft.block.Block;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;


@Mixin(AbstractBlock.AbstractBlockState.class)
public abstract class AbstractBlockCollisionDetectorMixin {
    @Shadow
    public abstract Block getBlock();

    // Hook the onBlockCollision method
    // to add custom collision detection logic
    @Inject(method = "onEntityCollision", at = @At("HEAD"))
    public void onEntityCollision(World world, BlockPos pos, Entity entity, CallbackInfo ci) {
//        System.out.println("onEntityCollision at " + pos + " with entity " + entity);
        if (!entity.isPlayer())
            return;

        // Check if the collision info is needed at the tick and save it for the tick
        String blockName = this.getBlock().getTranslationKey();
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
