package com.kyhsgeekcode.minecraftenv.client;

import com.kyhsgeekcode.minecraftenv.MinecraftEnv;
import com.kyhsgeekcode.minecraftenv.customentity.RealisticHumanModel;
import com.kyhsgeekcode.minecraftenv.customentity.RealisticHumanRenderer;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import net.fabricmc.fabric.api.client.rendering.v1.EntityModelLayerRegistry;
import net.fabricmc.fabric.api.client.rendering.v1.EntityRendererRegistry;
import net.minecraft.client.render.entity.model.EntityModelLayer;
import net.minecraft.util.Identifier;

@Environment(EnvType.CLIENT)
public class Minecraft_envClient implements ClientModInitializer {
    public static final EntityModelLayer MODEL_CUBE_LAYER = new EntityModelLayer(Identifier.of("entitytesting", "cube"), "main");

    @Override
    public void onInitializeClient() {
        System.out.println("Hello Fabric world! client");
        var ld_preload = System.getenv("LD_PRELOAD");
        if (ld_preload != null) {
            System.out.println("LD_PRELOAD is set: " + ld_preload);
        } else {
            System.out.println("LD_PRELOAD is not set");
        }
        EntityRendererRegistry.register(
                MinecraftEnv.Companion.getREALISTIC_HUMAN(),
                RealisticHumanRenderer::new
        );
        EntityModelLayerRegistry.registerModelLayer(MODEL_CUBE_LAYER, RealisticHumanModel::getTexturedModelData);
    }
}
