package com.kyhsgeekcode.minecraftenv

import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.BlockCollisionInfo
import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.EntityCollisionInfo

class CollisionListener {
    companion object {
        val blockCollisionInfoSet: HashSet<String> = HashSet()
        val blockCollisionInfo: ArrayList<BlockCollisionInfo> = ArrayList()

        val entityCollisionInfoSet: HashSet<String> = HashSet()
        val entityCollisionInfo: ArrayList<EntityCollisionInfo> = ArrayList()
    }
}
