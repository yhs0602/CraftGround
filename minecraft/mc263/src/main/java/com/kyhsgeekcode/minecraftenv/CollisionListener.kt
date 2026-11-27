package com.kyhsgeekcode.minecraftenv

import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.BlockCollisionInfo
import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.EntityCollisionInfo
import java.util.Collections

class CollisionListener {
    companion object {
        val blockCollisionInfoSet: HashSet<String> = HashSet()
        val blockCollisionInfo: MutableList<BlockCollisionInfo> = Collections.synchronizedList(ArrayList())

        val entityCollisionInfoSet: HashSet<String> = HashSet()
        val entityCollisionInfo: MutableList<EntityCollisionInfo> = Collections.synchronizedList(ArrayList())
    }
}
