package com.kyhsgeekcode.minecraftenv

/**
 * Seam B (26_2_phase2_plan.md §6.3). Tick event handlers call these three points and must
 * not know whether a step is lockstep-synchronized or free-running — that decision is
 * `LockStepBarrier` vs `NoOpBarrier`. A future `DistributedBarrier` (§6.5) plugs in here
 * without touching the handlers. Do not add a third implementation speculatively (§6.4).
 */
internal interface StepBarrier {
    /** Called at END_WORLD_TICK: let the server proceed, then block for its tick to finish. */
    fun onClientTickEnd()

    /** Called at START_SERVER_TICK: block until the client has applied this step's action. */
    fun onServerTickStart()

    /** Called at END_SERVER_TICK: let the waiting client proceed to send its observation. */
    fun onServerTickEnd()

    fun terminate()
}

/** Wraps the existing [TickSynchronizer] condvar rendezvous — the current sync-mode behavior. */
internal class LockStepBarrier(
    private val skipSync: () -> Boolean,
) : StepBarrier {
    private val synchronizer = TickSynchronizer()

    override fun onClientTickEnd() {
        synchronizer.notifyServerTickStart()
        if (!skipSync()) {
            synchronizer.waitForServerTickCompletion()
        }
    }

    override fun onServerTickStart() {
        if (!skipSync()) {
            synchronizer.waitForClientAction()
        }
    }

    override fun onServerTickEnd() {
        synchronizer.notifyClientSendObservation()
    }

    override fun terminate() {
        synchronizer.terminate()
    }
}

/** Async mode (W13): no rendezvous at all. Client and server tick independently. */
internal class NoOpBarrier : StepBarrier {
    override fun onClientTickEnd() {}

    override fun onServerTickStart() {}

    override fun onServerTickEnd() {}

    override fun terminate() {}
}
