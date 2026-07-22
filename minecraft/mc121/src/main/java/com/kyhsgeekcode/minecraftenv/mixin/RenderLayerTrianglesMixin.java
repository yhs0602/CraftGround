package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.render.RenderLayer;
import net.minecraft.client.render.RenderPhase;
import net.minecraft.client.render.VertexFormat;
import net.minecraft.client.render.VertexFormats;
import net.minecraft.util.Identifier;
import net.minecraft.util.Util;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.function.Function;

import static net.minecraft.client.render.RenderPhase.ENABLE_LIGHTMAP;
import static net.minecraft.client.render.RenderPhase.ENABLE_OVERLAY_COLOR;
import static net.minecraft.client.render.RenderPhase.ENTITY_SOLID_PROGRAM;
import static net.minecraft.client.render.RenderPhase.NO_TRANSPARENCY;

@Mixin(RenderLayer.class)
public class RenderLayerTrianglesMixin {
    @Unique
    private static final Function<Identifier, RenderLayer> ENTITY_SOLID_TRIANGLES = Util.memoize(texture -> {
        RenderLayer.MultiPhaseParameters multiPhaseParameters = RenderLayer.MultiPhaseParameters
                .builder()
                .program(ENTITY_SOLID_PROGRAM)
                .texture(new RenderPhase
                                .Texture(
                                texture,
                                false,
                                false
                        )
                )
                .transparency(NO_TRANSPARENCY)
                .lightmap(ENABLE_LIGHTMAP)
                .overlay(ENABLE_OVERLAY_COLOR)
                .build(true
                );
        return RenderLayer.of(
                "entity_solid_triangles",
                VertexFormats.POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL,
                VertexFormat.DrawMode.TRIANGLES,
                1536,
                true,
                false,
                multiPhaseParameters
        );
    });

    @Inject(method = "getEntitySolid", at = @At(value = "HEAD"), cancellable = true)
    private static void getEntitySolid(Identifier texture, CallbackInfoReturnable<RenderLayer> cir) {
        if (texture.getPath().contains("triangles")) {
            cir.setReturnValue(ENTITY_SOLID_TRIANGLES.apply(texture));
        }
    }
}
