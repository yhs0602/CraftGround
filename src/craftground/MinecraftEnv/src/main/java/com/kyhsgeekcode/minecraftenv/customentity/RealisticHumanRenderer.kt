package com.kyhsgeekcode.minecraftenv.customentity

import com.kyhsgeekcode.minecraftenv.client.Minecraft_envClient
import net.minecraft.client.render.OverlayTexture
import net.minecraft.client.render.VertexConsumerProvider
import net.minecraft.client.render.entity.EntityRendererFactory
import net.minecraft.client.render.entity.MobEntityRenderer
import net.minecraft.client.util.math.MatrixStack
import net.minecraft.util.Identifier

class RealisticHumanRenderer(
    context: EntityRendererFactory.Context,
) : MobEntityRenderer<RealisticHuman, RealisticHumanModel>(
        context,
        RealisticHumanModel(context.getPart(Minecraft_envClient.MODEL_CUBE_LAYER)),
        0.5f,
    ) {
//    override fun getTexture(entity: RealisticHuman?): Identifier = Identifier.of("minecraft", "textures/entity/blaze.png")

    override fun getTexture(entity: RealisticHuman?): Identifier =
        Identifier.of("minecraftenv", "textures/entity/human_texture_triangle.png")

    override fun render(
        livingEntity: RealisticHuman?,
        f: Float,
        g: Float,
        matrixStack: MatrixStack,
        vertexConsumerProvider: VertexConsumerProvider?,
        light: Int,
    ) {
        ModelCache.load()
        matrixStack.push()
//        matrixStack.translate(0.0, 1.0, 0.0)
        matrixStack.scale(0.3f, 0.3f, 0.3f)

        val buffer = vertexConsumerProvider?.getBuffer(model.getLayer(getTexture(livingEntity)))

        if (buffer == null) {
            println("Error: Buffer is null")
            return
        }
        val entry = matrixStack.peek()
        val matrix = entry.positionMatrix

//        RenderSystem.setShader(GameRenderer::getPositionTexProgram)
//        RenderSystem.setShaderTexture(0, ModelCache.sphereTexture)

//        val tesselator = Tessellator.getInstance()
//        val bufferBuilder = tesselator.begin(VertexFormat.DrawMode.TRIANGLES, VertexFormats.POSITION_TEXTURE)

        val modelData = ModelCache.modelData
        for (i in ModelCache.modelData.vertices.indices step 3) {
            val x = modelData.vertices[i]
            val y = modelData.vertices[i + 1]
            val z = modelData.vertices[i + 2]
            val u = modelData.uvs[i / 3 * 2]
            val v = modelData.uvs[i / 3 * 2 + 1]
            val normalX = modelData.normals[i]
            val normalY = modelData.normals[i + 1]
            val normalZ = modelData.normals[i + 2]
            buffer
                .vertex(matrix, x, y, z)
                .texture(u, v)
                .normal(matrixStack.peek(), normalX, normalY, normalZ)
                .color(0x30FF00FF)
                .overlay(OverlayTexture.DEFAULT_UV)
                .light(light)
        }

//        BufferRenderer.drawWithGlobalProgram(bufferBuilder.end())

//        sphereBuffer.draw(
//            matrixStack.peek().positionMatrix,
//            RenderSystem.getProjectionMatrix(),
//            GameRenderer.getPositionTexProgram(),
//        )

        matrixStack.pop()
        super.render(livingEntity, f, g, matrixStack, vertexConsumerProvider, light)
    }
}
