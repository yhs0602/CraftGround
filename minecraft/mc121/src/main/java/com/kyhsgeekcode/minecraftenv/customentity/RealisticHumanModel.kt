package com.kyhsgeekcode.minecraftenv.customentity

import net.minecraft.client.model.ModelData
import net.minecraft.client.model.ModelPart
import net.minecraft.client.model.ModelPartBuilder
import net.minecraft.client.model.ModelTransform
import net.minecraft.client.model.TexturedModelData
import net.minecraft.client.render.VertexConsumer
import net.minecraft.client.render.entity.model.EntityModel
import net.minecraft.client.render.entity.model.EntityModelPartNames
import net.minecraft.client.util.math.MatrixStack
import org.joml.Matrix4f
import org.joml.Vector3f

class RealisticHumanModel(
    modelPart: ModelPart,
) : EntityModel<RealisticHuman>() {
    private var base: ModelPart = modelPart.getChild(EntityModelPartNames.CUBE)

    override fun render(
        matrices: MatrixStack,
        vertices: VertexConsumer?,
        light: Int,
        overlay: Int,
        color: Int,
    ) {
        ModelCache.load()
        val entry = matrices.peek()
        val matrix: Matrix4f = entry.positionMatrix
        val vector3f = Vector3f()
        val modelData = ModelCache.modelData
        var count = 0
        for (i in ModelCache.modelData.vertices.indices step 3) {
            val x = modelData.vertices[i] / 16.0f
            val y = modelData.vertices[i + 1] / 16.0f
            val z = modelData.vertices[i + 2] / 16.0f
            val u = modelData.uvs[i / 3 * 2]
            val v = modelData.uvs[i / 3 * 2 + 1]
            val normalX = modelData.normals[i]
            val normalY = modelData.normals[i + 1]
            val normalZ = modelData.normals[i + 2]
            val vector3f2 = entry.transformNormal(normalX, normalY, normalZ, vector3f)
            val vector3f3 = matrix.transformPosition(x, y, z, vector3f)
            vertices?.vertex(
                vector3f3.x(),
                vector3f3.y(),
                vector3f3.z(),
                color,
                u,
                v,
                overlay,
                light,
                vector3f2.x(),
                vector3f2.y(),
                vector3f2.z(),
            )
            count++
        }

        println("Rendered $count vertices")

//        ImmutableList.of(this.base).forEach { modelRenderer ->
//            modelRenderer.render(matrices, vertices, light, overlay, color)
//        }
    }

    override fun setAngles(
        entity: RealisticHuman?,
        limbAngle: Float,
        limbDistance: Float,
        animationProgress: Float,
        headYaw: Float,
        headPitch: Float,
    ) {
    }

    companion object {
        @JvmStatic
        fun getTexturedModelData(): TexturedModelData {
            val modelData = ModelData()
            val modelPartData = modelData.root
            modelPartData.addChild(
                EntityModelPartNames.CUBE,
                ModelPartBuilder.create().uv(0, 0).cuboid(-6f, 12f, -6f, 12f, 12f, 12f),
                ModelTransform.pivot(0f, 0f, 0f),
            )
            return TexturedModelData.of(modelData, 64, 64)
        }
    }
}
