package com.kyhsgeekcode.minecraftenv.customentity

import net.minecraft.resource.ResourceManager
import net.minecraft.util.Identifier

object OBJLoader {
    data class ModelData(
        val vertices: FloatArray,
        val uvs: FloatArray,
        val normals: FloatArray,
    )

    fun load(
        resourceManager: ResourceManager,
        id: Identifier,
    ): ModelData {
        val vertices = mutableListOf<Float>()
        val uvs = mutableListOf<Float>()
        val normals = mutableListOf<Float>()

        val outVertices = mutableListOf<Float>()
        val outUvs = mutableListOf<Float>()
        val outNormals = mutableListOf<Float>()

        val inputStream = resourceManager.getResource(id).get().inputStream
        inputStream.bufferedReader().useLines { lines ->
            for (line in lines) {
                val tokens = line.trim().split("\\s+".toRegex())
                when (tokens.firstOrNull()) {
                    "v" -> vertices.addAll(tokens.drop(1).map { it.toFloat() })
                    "vt" -> uvs.addAll(tokens.drop(1).map { it.toFloat() })
                    "vn" -> normals.addAll(tokens.drop(1).map { it.toFloat() })

                    "f" -> {
                        for (i in 1 until tokens.size) {
                            val parts = tokens[i].split("/")
                            val vIndex = parts.getOrNull(0)?.toIntOrNull()?.minus(1) ?: continue
                            val vtIndex = parts.getOrNull(1)?.toIntOrNull()?.minus(1) ?: 0
                            val vnIndex = parts.getOrNull(2)?.toIntOrNull()?.minus(1) ?: 0

                            // vertex
                            outVertices.add(vertices[vIndex * 3])
                            outVertices.add(vertices[vIndex * 3 + 1])
                            outVertices.add(vertices[vIndex * 3 + 2])

                            // uv
                            if (uvs.isNotEmpty()) {
                                outUvs.add(uvs[vtIndex * 2])
                                outUvs.add(1 - uvs[vtIndex * 2 + 1]) // flip V
                            } else {
                                outUvs.add(0f)
                                outUvs.add(0f)
                            }

                            // normal
                            if (normals.isNotEmpty()) {
                                outNormals.add(normals[vnIndex * 3])
                                outNormals.add(normals[vnIndex * 3 + 1])
                                outNormals.add(normals[vnIndex * 3 + 2])
                            } else {
                                outNormals.add(0f)
                                outNormals.add(1f)
                                outNormals.add(0f)
                            }
                        }
                    }
                }
            }
        }

        return ModelData(
            vertices = outVertices.toFloatArray(),
            uvs = outUvs.toFloatArray(),
            normals = outNormals.toFloatArray(),
        )
    }
}
