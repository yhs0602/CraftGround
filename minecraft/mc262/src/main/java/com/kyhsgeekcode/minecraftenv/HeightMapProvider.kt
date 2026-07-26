package com.kyhsgeekcode.minecraftenv

import net.minecraft.core.BlockPos
import net.minecraft.world.level.ChunkPos
import net.minecraft.world.level.Level
import net.minecraft.world.level.levelgen.Heightmap

class HeightMapProvider {
    // Returns the height map of the given position with the given radius in chunks
    fun getHeightMap(
        world: Level,
        pos: BlockPos,
        radiusInChunks: Int,
    ): List<HeightMapInfo> {
        val heightMapInfoList = mutableListOf<HeightMapInfo>()
        for (dx in -radiusInChunks..radiusInChunks) {
            for (dz in -radiusInChunks..radiusInChunks) {
                val chunkPos = ChunkPos.containing(pos.offset(dx * 16, 0, dz * 16))
                val heightMap = world.getChunk(chunkPos.x, chunkPos.z).getOrCreateHeightmapUnprimed(Heightmap.Types.WORLD_SURFACE)
                for (x in 0..15) {
                    for (z in 0..15) {
                        val blockPos = BlockPos(chunkPos.minBlockX + x, pos.y, chunkPos.minBlockZ + z)
                        val height = heightMap.getFirstAvailable(x, z)
                        val blockName = world.getBlockState(blockPos).block.descriptionId
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
