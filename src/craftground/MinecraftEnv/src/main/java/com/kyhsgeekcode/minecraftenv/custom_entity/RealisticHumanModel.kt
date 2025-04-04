package com.kyhsgeekcode.minecraftenv.custom_entity

import com.google.common.collect.ImmutableList
import net.minecraft.client.model.ModelData
import net.minecraft.client.model.ModelPart
import net.minecraft.client.model.ModelPartBuilder
import net.minecraft.client.model.ModelTransform
import net.minecraft.client.model.TexturedModelData
import net.minecraft.client.render.VertexConsumer
import net.minecraft.client.render.entity.model.EntityModel
import net.minecraft.client.render.entity.model.EntityModelPartNames
import net.minecraft.client.util.math.MatrixStack

class RealisticHumanModel(
    modelPart: ModelPart,
) : EntityModel<RealisticHuman>() {
    private var base: ModelPart = modelPart.getChild(EntityModelPartNames.CUBE)

    override fun render(
        matrices: MatrixStack?,
        vertices: VertexConsumer?,
        light: Int,
        overlay: Int,
        color: Int,
    ) {
        ImmutableList.of(this.base).forEach { modelRenderer ->
            modelRenderer.render(matrices, vertices, light, overlay, color)
        }
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
