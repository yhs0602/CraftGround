package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.LidarRayVisualizer;
import com.mojang.blaze3d.systems.RenderSystem;
import net.minecraft.client.render.*;
import net.minecraft.client.util.math.MatrixStack;
import net.minecraft.util.math.Vec3d;
import org.joml.Matrix4f;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Mixin to render Lidar rays in the world for visualization
 */
@Mixin(WorldRenderer.class)
public class LidarRayVisualizationMixin {

    @Inject(
            method = "render",
            at = @At("RETURN")
    )
    private void renderLidarRays(
            RenderTickCounter tickCounter,
            boolean renderBlockOutline,
            Camera camera,
            GameRenderer gameRenderer,
            LightmapTextureManager lightmapTextureManager,
            Matrix4f positionMatrix,
            Matrix4f projectionMatrix,
            CallbackInfo ci
    ) {
        if (!LidarRayVisualizer.INSTANCE.getEnabled()) {
            return;
        }

        var rays = LidarRayVisualizer.INSTANCE.getRays();
        if (rays.isEmpty()) {
            return;
        }

        // Get camera position for offset calculation
        Vec3d cameraPos = camera.getPos();

        // Setup rendering
        RenderSystem.enableDepthTest();
        RenderSystem.setShader(GameRenderer::getPositionColorProgram);
        RenderSystem.disableCull();
        RenderSystem.enableBlend();
        RenderSystem.defaultBlendFunc();
        RenderSystem.lineWidth(2.0f);

        // Apply the position matrix
        RenderSystem.setShaderColor(1.0f, 1.0f, 1.0f, 1.0f);

        MatrixStack matrixStack = new MatrixStack();
        matrixStack.multiplyPositionMatrix(positionMatrix);

        Tessellator tessellator = Tessellator.getInstance();
        BufferBuilder bufferBuilder = tessellator.begin(
                VertexFormat.DrawMode.DEBUG_LINES,
                VertexFormats.POSITION_COLOR
        );

        Matrix4f matrix = matrixStack.peek().getPositionMatrix();

        for (LidarRayVisualizer.VisualRay ray : rays) {
            // Calculate positions relative to camera
            float startX = (float) (ray.getStart().x - cameraPos.x);
            float startY = (float) (ray.getStart().y - cameraPos.y);
            float startZ = (float) (ray.getStart().z - cameraPos.z);
            float endX = (float) (ray.getEnd().x - cameraPos.x);
            float endY = (float) (ray.getEnd().y - cameraPos.y);
            float endZ = (float) (ray.getEnd().z - cameraPos.z);

            // Color based on hit type
            // 0 = MISS (gray), 1 = BLOCK (green), 2 = ENTITY (red)
            int red, green, blue, alpha;
            switch (ray.getHitType()) {
                case 1: // BLOCK
                    red = 0;
                    green = 255;
                    blue = 0;
                    alpha = 200;
                    break;
                case 2: // ENTITY
                    red = 255;
                    green = 0;
                    blue = 0;
                    alpha = 200;
                    break;
                default: // MISS
                    red = 128;
                    green = 128;
                    blue = 128;
                    alpha = 100;
                    break;
            }

            // Draw line from start to end
            bufferBuilder.vertex(matrix, startX, startY, startZ)
                    .color(red, green, blue, alpha);
            bufferBuilder.vertex(matrix, endX, endY, endZ)
                    .color(red, green, blue, alpha);
        }

        BufferRenderer.drawWithGlobalProgram(bufferBuilder.end());

        // Restore state
        RenderSystem.disableBlend();
        RenderSystem.enableCull();
    }
}

