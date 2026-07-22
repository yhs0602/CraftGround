package com.kyhsgeekcode.minecraftenv

import net.minecraft.util.math.BlockPos
import net.minecraft.util.math.ChunkPos
import net.minecraft.world.Heightmap
import net.minecraft.world.World

class HeightMapProvider {
    // Returns the height map of the given position with the given radius in chunks
    fun getHeightMap(
        world: World,
        pos: BlockPos,
        radiusInChunks: Int,
    ): List<HeightMapInfo> {
        val heightMapInfoList = mutableListOf<HeightMapInfo>()
        for (dx in -radiusInChunks..radiusInChunks) {
            for (dz in -radiusInChunks..radiusInChunks) {
                val chunkPos = ChunkPos(pos.add(dx * 16, 0, dz * 16))
                val heightMap = world.getChunk(chunkPos.x, chunkPos.z).getHeightmap(Heightmap.Type.WORLD_SURFACE)
                for (x in 0..15) {
                    for (z in 0..15) {
                        val blockPos = BlockPos(chunkPos.startX + x, pos.y, chunkPos.startZ + z)
                        val height = heightMap[x, z]
                        val blockName = world.getBlockState(blockPos).block.translationKey
                        heightMapInfoList.add(HeightMapInfo(blockPos.x, blockPos.z, height, blockName))
                    }
                }
            }
        }
        return heightMapInfoList
    }
}

data class HeightMapInfo(
    val x: Int,
    val z: Int,
    val height: Int,
    val blockName: String,
)
