package com.kyhsgeekcode.minecraftenv.customentity

import net.minecraft.client.MinecraftClient
import net.minecraft.util.Identifier

object ModelCache {
    lateinit var sphereTexture: Identifier
    lateinit var modelData: OBJLoader.ModelData
    var initialized = false

    fun load() {
        if (initialized) {
            return
        }
        val resourceManager = MinecraftClient.getInstance().resourceManager
        // https://web.mit.edu/djwendel/www/weblogo/shapes/basic-shapes/sphere/sphere.obj
        val identifier = Identifier.of("minecraftenv", "models/external/human.obj")
        if (identifier != null) {
            println("Loading model from $identifier")
        } else {
            println("Model identifier is null")
        }
        if (resourceManager == null) {
            println("Resource manager is null")
        } else {
            println("Resource manager is not null")
        }
        modelData = OBJLoader.load(resourceManager, identifier)
//        val tesselator = Tessellator.getInstance()
//        val bufferBuilder = tesselator.begin(VertexFormat.DrawMode.TRIANGLES, VertexFormats.POSITION_TEXTURE)
//
//        for (i in modelData.vertices.indices step 3) {
//            val x = modelData.vertices[i]
//            val y = modelData.vertices[i + 1]
//            val z = modelData.vertices[i + 2]
//            val u = modelData.uvs[i / 3 * 2]
//            val v = modelData.uvs[i / 3 * 2 + 1]
//            bufferBuilder.vertex(x, y, z).texture(u, v)
//        }
//
//        sphereBuffer = bufferBuilder.end()
        sphereTexture = Identifier.of("minecraftenv", "textures/entity/skull_and_roses_triangles.png")
        initialized = true
    }
}
