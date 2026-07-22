package com.kyhsgeekcode.minecraftenv

import net.minecraft.registry.entry.RegistryEntry
import net.minecraft.server.world.ServerWorld
import net.minecraft.util.math.BlockPos
import net.minecraft.util.math.ChunkPos
import net.minecraft.world.biome.Biome
import net.minecraft.world.biome.source.BiomeCoords

class BiomeCenterFinder(
    val world: ServerWorld,
) {
    // 주어진 좌표에서 반경 내에 있는 바이옴의 중심을 계산
    fun calculateBiomeCenter(
        startPos: BlockPos,
        radius: Int,
        targetBiome: RegistryEntry<Biome>,
    ): BlockPos? {
        // 바이옴 경계 좌표를 저장하기 위한 셋
        val biomeBoundaryPositions: MutableSet<BlockPos> = HashSet()

        // 주어진 반경 내의 청크를 반복
        for (dx in -radius..radius) {
            for (dz in -radius..radius) {
                val chunkPos = ChunkPos(startPos.add(dx * 16, 0, dz * 16))
                // 해당 청크 내에서 바이옴을 검사하여 경계를 찾음
                for (x in 0..15) {
                    for (z in 0..15) {
                        val blockPos = BlockPos(chunkPos.startX + x, startPos.y, chunkPos.startZ + z)
                        val biome =
                            world.getGeneratorStoredBiome(
                                BiomeCoords.fromBlock(blockPos.x),
                                BiomeCoords.fromBlock(blockPos.y),
                                BiomeCoords.fromBlock(blockPos.z),
                            )
                        // 목표 바이옴과 일치하는 경우 경계 좌표로 간주
                        if (biome == targetBiome) {
                            biomeBoundaryPositions.add(blockPos)
                        }
                    }
                }
            }
        }

        // 경계 좌표가 없는 경우 null 반환
        if (biomeBoundaryPositions.isEmpty()) {
            return null
        }

        // 모든 경계 좌표의 중심을 계산
        return getAveragePosition(biomeBoundaryPositions)
    }

    // 주어진 좌표들의 평균 위치를 계산하는 메서드
    private fun getAveragePosition(positions: Set<BlockPos>): BlockPos {
        var sumX: Long = 0
        var sumY: Long = 0
        var sumZ: Long = 0
        val count = positions.size

        for (pos in positions) {
            sumX += pos.x.toLong()
            sumY += pos.y.toLong()
            sumZ += pos.z.toLong()
        }

        return BlockPos((sumX / count).toInt(), (sumY / count).toInt(), (sumZ / count).toInt())
    }

    fun getNearbyBiomes(
        startPos: BlockPos,
        radiusInChunks: Int,
    ): List<NearbyBiome> {
        val nearbyBiomes = mutableListOf<NearbyBiome>()
        // 주어진 반경 내의 청크를 반복
        for (dx in -radiusInChunks..radiusInChunks) {
            for (dz in -radiusInChunks..radiusInChunks) {
                val chunkPos = ChunkPos(startPos.add(dx * 16, 0, dz * 16))
                // 해당 청크 내에서 바이옴을 검사하여 경계를 찾음
                for (x in 0..15) {
                    for (z in 0..15) {
                        val blockPos = BlockPos(chunkPos.startX + x, startPos.y, chunkPos.startZ + z)
                        val biome =
                            world.getGeneratorStoredBiome(
                                BiomeCoords.fromBlock(blockPos.x),
                                BiomeCoords.fromBlock(blockPos.y),
                                BiomeCoords.fromBlock(blockPos.z),
                            )
                        nearbyBiomes.add(
                            NearbyBiome(
                                blockPos.x,
                                blockPos.y,
                                blockPos.z,
                                biome,
                            ),
                        )
                    }
                }
            }
        }
        return nearbyBiomes
    }
}

data class NearbyBiome(
    val x: Int,
    val y: Int,
    val z: Int,
    val biome: RegistryEntry<Biome>,
)
