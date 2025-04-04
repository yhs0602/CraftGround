package com.kyhsgeekcode.minecraftenv.custom_entity

import com.kyhsgeekcode.minecraftenv.client.Minecraft_envClient
import net.minecraft.client.render.entity.EntityRendererFactory
import net.minecraft.client.render.entity.MobEntityRenderer
import net.minecraft.util.Identifier

class RealisticHumanRenderer(
    context: EntityRendererFactory.Context,
) : MobEntityRenderer<RealisticHuman, RealisticHumanModel>(
        context,
        RealisticHumanModel(context.getPart(Minecraft_envClient.MODEL_CUBE_LAYER)),
        0.5f,
    ) {
    override fun getTexture(entity: RealisticHuman?): Identifier = Identifier.of("minecraft", "textures/entity/blaze.png")
}
