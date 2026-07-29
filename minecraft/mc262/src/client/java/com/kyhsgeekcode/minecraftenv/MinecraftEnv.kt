@file:OptIn(ExperimentalPathApi::class)

package com.kyhsgeekcode.minecraftenv

import com.google.protobuf.ByteString
import com.kyhsgeekcode.minecraftenv.mixin.PacketProcessorQueueAccessor
import com.kyhsgeekcode.minecraftenv.proto.ActionSpace.ActionSpaceMessageV2
import com.kyhsgeekcode.minecraftenv.proto.InitialEnvironment
import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace
import com.kyhsgeekcode.minecraftenv.proto.biomeInfo
import com.kyhsgeekcode.minecraftenv.proto.blockInfo
import com.kyhsgeekcode.minecraftenv.proto.chatMessageInfo
import com.kyhsgeekcode.minecraftenv.proto.entitiesWithinDistance
import com.kyhsgeekcode.minecraftenv.proto.heightInfo
import com.kyhsgeekcode.minecraftenv.proto.hitResult
import com.kyhsgeekcode.minecraftenv.proto.lidarRay
import com.kyhsgeekcode.minecraftenv.proto.lidarResult
import com.kyhsgeekcode.minecraftenv.proto.nearbyBiome
import com.kyhsgeekcode.minecraftenv.proto.observationSpaceMessage
import net.fabricmc.api.ClientModInitializer
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents
import net.minecraft.client.Minecraft
import net.minecraft.client.gui.screens.DeathScreen
import net.minecraft.client.multiplayer.ClientLevel
import net.minecraft.client.player.LocalPlayer
import net.minecraft.core.BlockPos
import net.minecraft.core.registries.BuiltInRegistries
import net.minecraft.network.protocol.game.ServerboundClientCommandPacket
import net.minecraft.resources.Identifier
import net.minecraft.server.MinecraftServer
import net.minecraft.stats.Stats
import net.minecraft.tags.FluidTags
import net.minecraft.world.entity.EntityTypes
import net.minecraft.world.entity.player.Player
import net.minecraft.world.entity.projectile.ProjectileUtil
import net.minecraft.world.level.ClipContext
import net.minecraft.world.level.block.state.BlockState
import net.minecraft.world.phys.AABB
import net.minecraft.world.phys.BlockHitResult
import net.minecraft.world.phys.EntityHitResult
import net.minecraft.world.phys.HitResult
import net.minecraft.world.phys.Vec3
import net.minecraft.world.phys.shapes.BooleanOp
import net.minecraft.world.phys.shapes.Shapes
import java.io.IOException
import java.net.InetSocketAddress
import java.net.SocketTimeoutException
import java.net.StandardProtocolFamily
import java.net.UnixDomainSocketAddress
import java.nio.channels.ServerSocketChannel
import java.nio.channels.SocketChannel
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.ExperimentalPathApi
import kotlin.io.path.deleteRecursively
import kotlin.math.cos
import kotlin.math.sin
import kotlin.system.exitProcess

enum class ResetPhase {
    WAIT_PLAYER_DEATH,
    WAIT_PLAYER_RESPAWN,
    WAIT_INIT_ENDS,
    END_RESET,
}

enum class IOPhase {
    BEGINNING,
    GOT_INITIAL_ENVIRONMENT_SHOULD_SEND_OBSERVATION,
    GOT_INITIAL_ENVIRONMENT_SENT_OBSERVATION_SKIP_SEND_OBSERVATION,
    READ_ACTION_SHOULD_SEND_OBSERVATION,
    SENT_OBSERVATION_SHOULD_READ_ACTION,
}

/**
 * Port of mc121's MinecraftEnv onto 26.2. See 26_2_phase2_plan.md for the design this follows:
 *
 * - W6: "/tick rate 10000" (unchanged from mc121 - still backed by the vanilla
 *   TickRateManager in 26.2) is issued from EnvironmentInitializer.setUnlimitedTPS(), not here.
 * - W2: client tick pacing is pinned to 1 tick/frame by ClientTickPinMixin
 *   (DeltaTracker.Timer), not by anything in this file.
 * - D4: mc121's REALISTIC_HUMAN custom-entity registration is dropped - out of scope
 *   (see phase2_plan.md §4).
 *
 * Color and depth capture work on both of 26.2's rendering backends. The OpenGL backend keeps the
 * original native glReadPixels path; anything else (i.e. Vulkan) goes through Blaze3D's own
 * backend-neutral readback - see Blaze3dCapture.CaptureBackend and
 * docs/26_2_vulkan_capture.md.
 *
 * ZEROCOPY_TORCH is implemented on both backends: OpenGL via FramebufferCapturer's texture-based
 * IOSurface path (verified end to end), Vulkan via VulkanMetalZerocopy's VK_EXT_metal_objects
 * import (opt-in behind -Dcraftground.enableMetalObjects=true; see docs/26_2_vulkan_capture.md for
 * its verification status). EnvironmentInitializer fails loudly rather than silently producing
 * garbage if ZEROCOPY_TORCH is requested on Vulkan without that flag. Single-eye and stereo color
 * capture and depth capture are ported and verified end to end on both backends.
 */
class MinecraftEnv :
    ClientModInitializer,
    CommandExecutor {
    companion object {
        @Volatile
        private var instance: MinecraftEnv? = null

        // W1 (26_2_phase2_plan.md §2.2): called from RenderMixin's present()-redirect. Static so
        // the Java mixin (running in Minecraft's static context) can reach the running
        // MinecraftEnv instance without holding a reference of its own.
        @JvmStatic
        fun onPresentCapture() {
            instance?.handlePresentCapture()
        }

        // Called from RenderMixin immediately *before* the frame's CommandEncoder.submit(). Only
        // the backend-neutral capture path needs it: Blaze3D fences name the submission that is
        // current when they are created, so both the readback command and its fence have to be in
        // place before the submit they ride on (see docs/26_2_vulkan_capture.md).
        @JvmStatic
        fun onBeforeSubmitCapture() {
            instance?.handleBeforeSubmitCapture()
        }
    }

    private lateinit var initialEnvironment: InitialEnvironment.InitialEnvironmentMessage
    private var soundListener: MinecraftSoundListener? = null
    private var entityListener: EntityRenderListenerImpl? =
        null // tracks the entities rendered in the last tick
    private var resetPhase: ResetPhase = ResetPhase.END_RESET
    private var deathMessageCollector: GetMessagesInterface? = null

    private var skipSync = false

    // W13 (26_2_phase2_plan.md Seam B): tick handlers call stepBarrier, not TickSynchronizer
    // directly, so a future DistributedBarrier can be swapped in without touching them.
    private val stepBarrier: StepBarrier = LockStepBarrier(skipSync = { skipSync })

    // W12 (26_2_phase2_plan.md): the staleness instrumentation logs through this logger, so it
    // has to be switchable without a rebuild. CRAFTGROUND_JAVA_LOG enables the plain log,
    // CRAFTGROUND_JAVA_PROFILE additionally enables the profileStartPrint/profileEndPrint spans.
    // Both default to off - this sits on the per-step hot path.
    private val csvLogger =
        CsvLogger(
            "java_log.csv",
            enabled = System.getenv("CRAFTGROUND_JAVA_LOG") != null,
            profile = System.getenv("CRAFTGROUND_JAVA_PROFILE") != null,
        )

    private val variableCommandsAfterReset = mutableListOf<String>()
    private var ioPhase = IOPhase.BEGINNING
    private var useSharedMemory = false

    // W11 (26_2_phase2_plan.md Seam A): the integrated server, kept so numeric observations can
    // be read from the authoritative ServerPlayer instead of the client-predicted LocalPlayer.
    private var minecraftServer: MinecraftServer? = null

    // W12 staleness instrumentation.
    private var lastStepStartNanos: Long = 0L
    private var lastServerTickCount: Long = -1L

    // W1 (26_2_phase2_plan.md §2.2): capture+send moved out of END_LEVEL_TICK into the
    // present-redirect hook (RenderMixin.captureInsteadOfPresent -> onPresentCapture ->
    // handlePresentCapture below). END_LEVEL_TICK only decides *whether* to capture (based on
    // ioPhase, same as before) and records the world to capture; messageIO is promoted to a
    // field so the present-hook (which runs outside the END_LEVEL_TICK closure) can reach it.
    private lateinit var messageIO: MessageIO

    @Volatile
    private var pendingObservationWorld: ClientLevel? = null

    override fun onInitializeClient() {
        instance = this
        val isLdPreloadSet = System.getenv("LD_PRELOAD")
        if (isLdPreloadSet != null) {
            println("LD_PRELOAD is set: $isLdPreloadSet")
        } else {
            println("LD_PRELOAD is not set")
        }
        val socket: SocketChannel
        try {
            val portStr = System.getenv("PORT")
            val port = portStr?.toInt() ?: 8000
            val verbose =
                when (val verboseStr = System.getenv("VERBOSE")) {
                    "1" -> true
                    "0" -> false
                    else -> verboseStr?.toBoolean() ?: false
                }
            doPrintWithTime = verbose

            useSharedMemory =
                when (val useSharedMemoryStr = System.getenv("USE_SHARED_MEMORY")) {
                    "1" -> true
                    "0" -> false
                    else -> useSharedMemoryStr?.toBoolean() ?: false
                }

            if (useSharedMemory) {
                messageIO = SharedMemoryMessageIO(port)
            } else {
                val isWindows = System.getProperty("os.name").lowercase().contains("win")
                if (isWindows) {
                    val serverSocket = ServerSocketChannel.open()
                    serverSocket.bind(InetSocketAddress("127.0.0.1", port))
                    csvLogger.profileStartPrint("Minecraft_env/onInitialize/Accept")
                    socket = serverSocket.accept()
                    csvLogger.profileEndPrint("Minecraft_env/onInitialize/Accept")
                    messageIO = DomainSocketMessageIO(socket)
                } else {
                    val socketFilePath = Path.of("/tmp/minecraftrl_$port.sock")
                    socketFilePath.toFile().deleteOnExit()
                    csvLogger.log("Connecting to $port")
                    printWithTime("Connecting to $port")
                    Files.deleteIfExists(socketFilePath)
                    val serverSocket =
                        ServerSocketChannel
                            .open(StandardProtocolFamily.UNIX)
                            .bind(UnixDomainSocketAddress.of(socketFilePath))
                    csvLogger.profileStartPrint("Minecraft_env/onInitialize/Accept")
                    socket = serverSocket.accept()
                    csvLogger.profileEndPrint("Minecraft_env/onInitialize/Accept")
                    messageIO = DomainSocketMessageIO(socket)
                }
            }
        } catch (e: IOException) {
            throw RuntimeException(e)
        }
        skipSync = true
        csvLogger.log("Hello Fabric world!")

        csvLogger.profileStartPrint("Minecraft_env/onInitialize/readInitialEnvironment")
        initialEnvironment = messageIO.readInitialEnvironment()
        FramebufferCapturer.shouldCaptureDepth = initialEnvironment.requiresDepth
        FramebufferCapturer.requiresDepthConversion = initialEnvironment.requiresDepthConversion
        csvLogger.profileEndPrint("Minecraft_env/onInitialize/readInitialEnvironment")

        // Check the collision info dict and override collision dynamically
        for (blockCollisionKey in initialEnvironment.blockCollisionKeysList) {
            CollisionListener.blockCollisionInfoSet.add(blockCollisionKey)
        }
        for (entityCollisionKey in initialEnvironment.entityCollisionKeysList) {
            CollisionListener.entityCollisionInfoSet.add(entityCollisionKey)
        }

        ioPhase = IOPhase.GOT_INITIAL_ENVIRONMENT_SHOULD_SEND_OBSERVATION
        resetPhase = ResetPhase.WAIT_INIT_ENDS
        csvLogger.log("Initial environment read; $ioPhase $resetPhase")
        val initializer = EnvironmentInitializer(initialEnvironment, csvLogger)
        ClientTickEvents.START_CLIENT_TICK.register(
            ClientTickEvents.StartTick { client: Minecraft ->
                printWithTime("Start Client tick")
                csvLogger.profileStartPrint("Minecraft_env/onInitialize/ClientTick")
                initializer.onClientTick(client)
                if (soundListener == null) {
                    soundListener = MinecraftSoundListener(client.soundManager)
                }
                if (entityListener == null) {
                    entityListener =
                        EntityRenderListenerImpl(
                            client.levelExtractor as AddListenerInterface,
                        )
                }
                if (deathMessageCollector == null) {
                    deathMessageCollector = client.connection as GetMessagesInterface?
                }
                csvLogger.profileEndPrint("Minecraft_env/onInitialize/ClientTick")
            },
        )
        ClientTickEvents.START_LEVEL_TICK.register(
            ClientTickEvents.StartLevelTick { world: ClientLevel ->
                // read input
                printWithTime("Start client World tick")
                csvLogger.log("Start World tick")
                csvLogger.profileStartPrint("Minecraft_env/onInitialize/ClientWorldTick")
                onStartWorldTick(initializer, world, messageIO)
                csvLogger.profileEndPrint("Minecraft_env/onInitialize/ClientWorldTick")
                csvLogger.log("End World tick")
            },
        )
        ClientTickEvents.END_LEVEL_TICK.register(
            ClientTickEvents.EndLevelTick { world: ClientLevel ->
                lastStepStartNanos = System.nanoTime()
                // allow server to start tick, then wait until it ends
                csvLogger.profileStartPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/WaitServerTickEnds",
                )
                if (skipSync) {
                    csvLogger.log("Skip waiting server world tick ends")
                } else {
                    csvLogger.log("Wait server world tick ends")
                }
                stepBarrier.onClientTickEnd()
                csvLogger.profileEndPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/WaitServerTickEnds",
                )

                // W1-a (26_2_phase2_plan.md §1.1, C안): drain packets queued during the server
                // tick we just waited on, so ClientLevel reflects that tick's authoritative state
                // before capture. Mixin unnecessary - Minecraft.packetProcessor() and
                // PacketProcessor.processQueuedPackets() are both public.
                // W12: measure the queue depth we drained - evidence for whether W1-b's
                // marker-packet barrier (currently deferred) is actually needed.
                val client = Minecraft.getInstance()
                val packetProcessor = client.packetProcessor()
                val queuedPacketCount =
                    (packetProcessor as PacketProcessorQueueAccessor).`minecraftEnv$getPacketsToBeHandled`().size
                packetProcessor.processQueuedPackets()
                val tickCountDelta = lastServerTickCount - world.gameTime
                csvLogger.log(
                    "W12 packetQueueDepthAtDrain=$queuedPacketCount " +
                        "serverTick=$lastServerTickCount clientReflectedTick=${world.gameTime} " +
                        "tickCountDelta=$tickCountDelta",
                )

                // W1: capture+send is no longer done here - only decide *whether* this step
                // should capture (same ioPhase check as before) and hand the world off to the
                // present-redirect hook (handlePresentCapture), which fires later in this same
                // runTick once the frame has actually been rendered (phase2_plan.md §2.2).
                if (ioPhase ==
                    IOPhase.GOT_INITIAL_ENVIRONMENT_SENT_OBSERVATION_SKIP_SEND_OBSERVATION ||
                    ioPhase == IOPhase.SENT_OBSERVATION_SHOULD_READ_ACTION
                ) {
                    csvLogger.log("Skip send observation; $ioPhase")
                    pendingObservationWorld = null
                } else {
                    csvLogger.log("Will send observation at present hook; $ioPhase")
                    pendingObservationWorld = world
                }
            },
        )
        ServerTickEvents.START_SERVER_TICK.register(
            ServerTickEvents.StartTick { server: MinecraftServer ->
                minecraftServer = server
                // wait until client tick ends
                printWithTime("Wait client world tick ends")
                csvLogger.profileStartPrint(
                    "Minecraft_env/onInitialize/StartServerTick/WaitClientAction",
                )
                if (skipSync) {
                    csvLogger.log("Server tick start; skip waiting client world tick ends")
                    printWithTime("Server tick start; skip waiting client world tick ends")
                } else {
                    csvLogger.log("Real Wait client world tick ends")
                    printWithTime("Real Wait client world tick ends")
                }
                stepBarrier.onServerTickStart()
                csvLogger.profileEndPrint(
                    "Minecraft_env/onInitialize/StartServerTick/WaitClientAction",
                )
            },
        )
        ServerTickEvents.END_SERVER_TICK.register(
            ServerTickEvents.EndTick { server: MinecraftServer ->
                minecraftServer = server
                lastServerTickCount = server.tickCount.toLong()
                // allow client to end tick
                printWithTime("Notify server tick completion")
                csvLogger.log("Notify server tick completion")
                csvLogger.profileStartPrint(
                    "Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation",
                )
                stepBarrier.onServerTickEnd()
                csvLogger.profileEndPrint(
                    "Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation",
                )
            },
        )
    }

    private fun onStartWorldTick(
        initializer: EnvironmentInitializer,
        world: ClientLevel,
        messageIO: MessageIO,
    ) {
        val client = Minecraft.getInstance()
        soundListener!!.onTick()
        if (client.isPaused) return
        val player = client.player ?: return
        if (!player.isDeadOrDying && client.gui.screen() is DeathScreen) {
            sendSetScreenNull(client)
        }
        initializer.onWorldTick(client.getSingleplayerServer(), client.gui.hud.chat, this, emptyList())

        when (resetPhase) {
            ResetPhase.WAIT_PLAYER_DEATH -> {
                printWithTime("Waiting for player death")
                csvLogger.log("Waiting for player death")
                if (player.isDeadOrDying) {
                    player.respawn()
                    resetPhase = ResetPhase.WAIT_PLAYER_RESPAWN
                }
                return
            }

            ResetPhase.WAIT_PLAYER_RESPAWN -> {
                printWithTime("Waiting for player respawn")
                csvLogger.log("Waiting for player respawn")
                if (!player.isDeadOrDying) {
                    initializer.reset(client.gui.hud.chat, this, variableCommandsAfterReset)
                    variableCommandsAfterReset.clear()
                    resetPhase = ResetPhase.WAIT_INIT_ENDS
                }
                return
            }

            ResetPhase.WAIT_INIT_ENDS -> {
                printWithTime("Waiting for the initialization ends")
                csvLogger.log("Waiting for the initialization ends")
                if (initializer.initWorldFinished) {
                    sendSetScreenNull(client) // clear death screen
                    resetPhase = ResetPhase.END_RESET
                }
                return
            }

            ResetPhase.END_RESET -> {
                printWithTime("Reset end")
                csvLogger.log("Reset end")
            }
        }
        try {
            csvLogger.log("Will Read action")
            csvLogger.profileStartPrint("Minecraft_env/onInitialize/ClientWorldTick/ReadAction")
            val action = messageIO.readAction()
            csvLogger.profileEndPrint("Minecraft_env/onInitialize/ClientWorldTick/ReadAction")
            ioPhase = IOPhase.READ_ACTION_SHOULD_SEND_OBSERVATION
            csvLogger.log("Read action done; $ioPhase")
            skipSync = false
            val commands = action.commandsList

            if (commands.isNotEmpty()) {
                for (command in commands) {
                    if (handleCommand(command, client, player)) {
                        return
                    }
                }
            }
            if (player.isDeadOrDying) {
                return
            } else if (client.gui.screen() is DeathScreen) {
                sendSetScreenNull(client)
            }
            if (applyAction(action, player, client)) return
        } catch (e: SocketTimeoutException) {
            printWithTime("Timeout")
            csvLogger.log("Timeout")
        } catch (e: IOException) {
            stepBarrier.terminate()
            e.printStackTrace()
            exitProcess(-1)
        } catch (e: Exception) {
            stepBarrier.terminate()
            e.printStackTrace()
            exitProcess(-2)
        }
    }

    private fun sendSetScreenNull(client: Minecraft) {
        client.gui.setScreen(null)
    }

    // Returns: Should ignore action
    private fun handleCommand(
        command: String,
        client: Minecraft,
        player: LocalPlayer,
    ): Boolean {
        if (command == "respawn") {
            if (client.gui.screen() is DeathScreen && player.isDeadOrDying) {
                player.respawn()
                sendSetScreenNull(client)
            }
            return true
        } else if (command.startsWith("fastreset")) {
            printWithTime("Fast resetting")
            csvLogger.log("Fast resetting")
            val extraCommand = command.substringAfter("fastreset ").trim()
            if (extraCommand.isNotEmpty()) {
                val commands = extraCommand.split(";")
                printWithTime("Extra commands: $commands")
                variableCommandsAfterReset.addAll(commands)
            }
            resetPhase = ResetPhase.WAIT_PLAYER_DEATH
            runCommand(player, "/kill @p") // kill player
            runCommand(player, "/tp @e[type=!player] ~ -500 ~") // send to void
            return true
        } else if (command.startsWith("random-summon")) {
            printWithTime("Random summon")
            csvLogger.log("Random summon")
            val arguments = command.substringAfter("random-summon ").trim()
            val argumentsList = arguments.split(" ")
            val entityName = argumentsList[0]
            val x = argumentsList[1].toInt()
            val y = argumentsList[2].toInt()
            val z = argumentsList[3].toInt()
            return false
        } else if (command == "exit") {
            printWithTime("Will terminate")
            csvLogger.log("Will terminate")
            stepBarrier.terminate()
            // remove the world file
            client.getSingleplayerServer()?.getWorldPath(net.minecraft.world.level.storage.LevelResource.ROOT)?.let {
                try {
                    it.deleteRecursively()
                    printWithTime("Successfully deleted the world $it")
                } catch (e: IOException) {
                    printWithTime("Failed to delete the world $it")
                    e.printStackTrace()
                }
            }
            exitProcess(0)
        } else {
            runCommand(player, command)
            printWithTime("Executed command: $command")
            csvLogger.log("Executed command: $command")
            return false
        }
    }

    private fun applyAction(
        actionDict: ActionSpaceMessageV2,
        player: LocalPlayer,
        client: Minecraft,
    ): Boolean {
        csvLogger.profileStartPrint(
            "Minecraft_env/onInitialize/ClientWorldTick/ReadAction/ApplyAction",
        )
        if (actionDict.cameraYaw != 0.0f || actionDict.cameraPitch != 0.0f) {
            val dy = actionDict.cameraPitch * 20.0 / 3
            val dx = actionDict.cameraYaw * 20.0 / 3
            MouseInfo.moveMouseBy(dx.toInt(), dy.toInt())
        }

        // Handle key press
        KeyboardInfo.onAction(actionDict)
        val currentScreen = client.gui.screen()
        if (currentScreen != null && currentScreen is DeathScreen) {
            // Disable disconnect button
            return false
        }
        MouseInfo.onAction(actionDict)
        csvLogger.profileEndPrint(
            "Minecraft_env/onInitialize/ClientWorldTick/ReadAction/ApplyAction",
        )
        return false
    }

    /**
     * Records this frame's color readback and arms the fence, on the backend-neutral capture path
     * only. Skipped for stereo, which discards this frame and re-renders per eye
     * ([renderEyeAndCapture] does its own record/arm/submit). For ZEROCOPY_TORCH on Vulkan, this
     * records [VulkanMetalZerocopy.recordCopy] instead of a CPU readback; on OpenGL, ZEROCOPY_TORCH
     * doesn't go through this hook at all (see [FramebufferCapturer.captureFramebuffer]).
     */
    private fun handleBeforeSubmitCapture() {
        if (pendingObservationWorld == null) return
        val client = Minecraft.getInstance()
        val colorTexture = client.gameRenderer.mainRenderTarget().colorTexture ?: return
        if (Blaze3dCapture.backendFor(colorTexture) != Blaze3dCapture.CaptureBackend.BLAZE3D) {
            return
        }
        if (initialEnvironment.screenEncodingMode == FramebufferCapturer.ZEROCOPY_TORCH) {
            val device = com.mojang.blaze3d.systems.RenderSystem.getDevice() as com.mojang.blaze3d.vulkan.VulkanDevice
            VulkanMetalZerocopy.recordCopy(
                device,
                colorTexture as com.mojang.blaze3d.vulkan.VulkanGpuTexture,
                initialEnvironment.imageSizeX,
                initialEnvironment.imageSizeY,
            )
            Blaze3dCapture.armFence()
            return
        }
        if (initialEnvironment.eyeDistance <= 0) {
            Blaze3dCapture.recordColorReadback(colorTexture)
        }
        // Armed even in the stereo case, because the depth mixin may already have recorded a depth
        // copy into this same submission.
        Blaze3dCapture.armFence()
    }

    // W1 (26_2_phase2_plan.md §2.2): called from RenderMixin.captureAfterSubmit via
    // onPresentCapture() - after RenderSystem.getDevice().createCommandEncoder().submit() in the
    // same renderFrame() call that followed this step's END_LEVEL_TICK. Runs on the client render
    // thread, same as END_LEVEL_TICK did, so no additional synchronization is needed around
    // pendingObservationWorld beyond @Volatile (only one frame's worth of state is ever pending at
    // a time - W2 pins ticksToDo to 1, so exactly one END_LEVEL_TICK precedes each renderFrame()
    // call).
    private fun handlePresentCapture() {
        val world = pendingObservationWorld ?: return
        pendingObservationWorld = null
        // No-op unless handleBeforeSubmitCapture armed a fence; the submission it names has now
        // been issued by the submit() this hook sits behind.
        Blaze3dCapture.awaitPendingFence()
        csvLogger.profileStartPrint(
            "Minecraft_env/onInitialize/EndWorldTick/SendObservation",
        )
        sendObservation(messageIO, world)
        csvLogger.profileEndPrint(
            "Minecraft_env/onInitialize/EndWorldTick/SendObservation",
        )
        val stepDurationMs = (System.nanoTime() - lastStepStartNanos) / 1_000_000.0
        csvLogger.log("W12 stepDurationMs=$stepDurationMs")
    }

    /**
     * Re-renders the level with the camera moved to [eye] and captures the result.
     *
     * mc121 did this by overwriting only the player's *previous* position (prevX/prevY/prevZ,
     * `xo`/`yo`/`zo` in Mojmap) and re-rendering. That works because `Camera.alignWithEntity`
     * places the camera at `Mth.lerp(partialTicks, entity.xo, entity.getX())` - and under
     * ClientTickPinMixin (W2) `deltaTickResidual` is pinned to 0, so `partialTicks` is exactly 0
     * and the lerp returns `xo` verbatim. The camera therefore lands precisely on [eye] without
     * touching the player's real position, which would have to be undone before the next tick and
     * could leak into collision/chunk state.
     *
     * The 26.2 equivalent of mc121's `render(client)` helper is update() + extract() + render():
     * `update()` re-runs `Camera.update` so the shifted position is picked up, `extract()` rebuilds
     * the render state (including frustum culling) against that camera, and `render()` draws it
     * into the main render target.
     *
     * On the OpenGL path `submit()` is deliberately *not* called - the GL backend issues commands
     * eagerly and the capture's `glReadPixels` is itself a sync point, whereas an extra `submit()`
     * would advance GlCommandEncoder's frame-fence ring out of step with the real frame. The
     * backend-neutral path has no such eager sync point: its readback only completes when the
     * submission carrying it does, so there it *must* submit (once per eye) - see
     * [Blaze3dCapture.captureNow] and docs/26_2_vulkan_capture.md.
     */
    private fun renderEyeAndCapture(
        client: Minecraft,
        player: LocalPlayer,
        eye: Vec3,
        captureBackend: Blaze3dCapture.CaptureBackend,
    ): ByteString {
        val originalXo = player.xo
        val originalYo = player.yo
        val originalZo = player.zo
        player.xo = eye.x
        player.yo = eye.y
        player.zo = eye.z
        try {
            val deltaTracker = client.deltaTracker
            client.gameRenderer.update(deltaTracker)
            client.gameRenderer.extract(deltaTracker, true)
            client.gameRenderer.render(deltaTracker, true)
            val mainRenderTarget = client.gameRenderer.mainRenderTarget()
            val colorTexture =
                mainRenderTarget.colorTexture
                    ?: throw IllegalStateException("Main render target has no color texture to capture")
            return if (captureBackend == Blaze3dCapture.CaptureBackend.BLAZE3D) {
                Blaze3dCapture.captureNow(
                    colorTexture,
                    initialEnvironment.imageSizeX,
                    initialEnvironment.imageSizeY,
                    initialEnvironment.screenEncodingMode,
                    MouseInfo.showCursor,
                    MouseInfo.mouseX.toInt(),
                    MouseInfo.mouseY.toInt(),
                )
            } else {
                FramebufferCapturer.captureFramebuffer(
                    (colorTexture as com.mojang.blaze3d.opengl.GlTexture).glId(),
                    0,
                    mainRenderTarget.width,
                    mainRenderTarget.height,
                    initialEnvironment.imageSizeX,
                    initialEnvironment.imageSizeY,
                    initialEnvironment.screenEncodingMode,
                    false,
                    MouseInfo.showCursor,
                    MouseInfo.mouseX.toInt(),
                    MouseInfo.mouseY.toInt(),
                )
            }
        } finally {
            player.xo = originalXo
            player.yo = originalYo
            player.zo = originalZo
        }
    }

    private fun sendObservation(
        messageIO: MessageIO,
        world: ClientLevel,
    ) {
        printWithTime("send Observation")
        csvLogger.log("send Observation")
        val client = Minecraft.getInstance()
        val player = client.player
        if (player == null) {
            printWithTime("Player is null")
            csvLogger.log("Player is null")
            return
        }
        val mainRenderTarget = client.gameRenderer.mainRenderTarget()
        val colorTexture =
            mainRenderTarget.colorTexture
                ?: throw IllegalStateException("Main render target has no color texture to capture")
        val captureBackend = Blaze3dCapture.backendFor(colorTexture)
        // Only the native GL readback needs GLEW; on the backend-neutral path there may not even be
        // a GL context, so initializing it would fail for no reason.
        if (captureBackend == Blaze3dCapture.CaptureBackend.OPENGL) {
            if (FramebufferCapturer.checkGLEW()) {
                printWithTime("GLEW initialized")
            } else {
                printWithTime("GLEW not initialized")
                throw RuntimeException("GLEW not initialized")
            }
        }
        if (initialEnvironment.screenEncodingMode == FramebufferCapturer.ZEROCOPY_TORCH) {
            // EnvironmentInitializer.checkRenderBackend already fails fast unless this is OpenGL,
            // or Vulkan with VK_EXT_metal_objects/VK_EXT_external_memory_metal actually enabled.
            // Both initialize() calls are guarded by their own ipcHandle, so this only does real
            // work once.
            if (captureBackend == Blaze3dCapture.CaptureBackend.BLAZE3D) {
                VulkanMetalZerocopy.initialize(
                    com.mojang.blaze3d.systems.RenderSystem.getDevice() as com.mojang.blaze3d.vulkan.VulkanDevice,
                    initialEnvironment.imageSizeX,
                    initialEnvironment.imageSizeY,
                    initialEnvironment.pythonPid,
                )
            } else {
                FramebufferCapturer.initializeZeroCopy(
                    initialEnvironment.imageSizeX,
                    initialEnvironment.imageSizeY,
                    initialEnvironment.pythonPid,
                )
            }
        }

        // W11 (26_2_phase2_plan.md §1.3/§6.3 Seam A) proposed reading numeric observations off the
        // authoritative ServerPlayer rather than the client-predicted LocalPlayer, on the theory
        // that the StepBarrier lock's happens-before makes the server copy race-free.
        //
        // We read the client player instead, because a server-sourced value structurally cannot
        // satisfy W1's same-step guarantee. LocalPlayer.sendPosition() runs inside
        // LocalPlayer.tick(), but camera rotation is applied by Minecraft.runTick's
        // mouseHandler.handleAccumulatedMovement() call, which happens AFTER the tick loop. So the
        // rotation produced by this step's action only reaches the server on the following tick -
        // whereas the observation W1 captures at the end of this same step already reflects it,
        // both in the rendered frame and in the client player's own yaw (verified end to end: a
        // +30 deg camera action yields yaw delta 29.85 and a changed image within one step).
        // Sourcing yaw from the server would report the pre-action value and reintroduce exactly
        // the off-by-one-step observation that W1 exists to eliminate. This also matches mc121's
        // long-shipped semantics.
        //
        // (Until the LevelLoadTrackerMixin added alongside this, the ServerPlayer was not merely
        // one tick behind but permanently frozen at spawn, since LocalPlayer.tick() - and with it
        // sendPosition() - never ran at all. That is fixed now; the choice above is about
        // same-step semantics, not about the sync being broken.)
        //
        // The ObservationSource seam and ServerAuthoritativeSource are kept, so switching to
        // server-authoritative numerics is a one-line change if that tradeoff is ever wanted.
        val observationSource: ObservationSource = ClientPredictedSource(player)

        // request stats from server
        csvLogger.profileStartPrint(
            "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Prepare",
        )
        client.connection?.send(
            ServerboundClientCommandPacket(ServerboundClientCommandPacket.Action.REQUEST_STATS),
        )
        try {
            val imageByteString1: ByteString
            val imageByteString2: ByteString
            val pos = Vec3(observationSource.x, observationSource.y, observationSource.z)
            if (initialEnvironment.eyeDistance > 0) {
                // Stereo capture, ported from mc121 (26_2_phase2_plan.md W1/W3/W5). The frame
                // that has just been rendered is discarded; the level is re-rendered once from
                // each eye and captured, exactly as mc121 did.
                val eyeWidth = initialEnvironment.eyeDistance
                val yawRadians = Math.toRadians(observationSource.yaw.toDouble())
                val left =
                    pos.add(eyeWidth * -sin(yawRadians), 0.0, eyeWidth * cos(yawRadians))
                val right =
                    pos.add(eyeWidth * sin(yawRadians), 0.0, eyeWidth * -cos(yawRadians))

                csvLogger.profileStartPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Prepare/StereoEye/ByteString",
                )
                imageByteString1 = renderEyeAndCapture(client, player, left, captureBackend)
                imageByteString2 = renderEyeAndCapture(client, player, right, captureBackend)
                csvLogger.profileEndPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Prepare/StereoEye/ByteString",
                )
            } else {
                csvLogger.profileStartPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Prepare/SingleEye/ByteString",
                )
                imageByteString1 =
                    if (captureBackend == Blaze3dCapture.CaptureBackend.BLAZE3D) {
                        // The copy was already recorded before this frame's submit() and waited on
                        // in handlePresentCapture(); this only maps and converts it.
                        Blaze3dCapture.readColor(
                            initialEnvironment.imageSizeX,
                            initialEnvironment.imageSizeY,
                            initialEnvironment.screenEncodingMode,
                            MouseInfo.showCursor,
                            MouseInfo.mouseX.toInt(),
                            MouseInfo.mouseY.toInt(),
                        )
                    } else {
                        FramebufferCapturer.captureFramebuffer(
                            (colorTexture as com.mojang.blaze3d.opengl.GlTexture).glId(),
                            // frameBufferId: unused - both RAW/PNG and ZEROCOPY_TORCH are
                            // texture-based on 26.2's OpenGL backend (see W3).
                            0,
                            mainRenderTarget.width,
                            mainRenderTarget.height,
                            initialEnvironment.imageSizeX,
                            initialEnvironment.imageSizeY,
                            initialEnvironment.screenEncodingMode,
                            false,
                            MouseInfo.showCursor,
                            MouseInfo.mouseX.toInt(),
                            MouseInfo.mouseY.toInt(),
                        )
                    }
                imageByteString2 = ByteString.EMPTY
                csvLogger.profileEndPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Prepare/SingleEye/ByteString",
                )
            }

            csvLogger.profileStartPrint(
                "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Prepare/Message",
            )
            val observationSpaceMessage =
                observationSpaceMessage {
                    image = imageByteString1
                    x = pos.x
                    y = pos.y
                    z = pos.z
                    pitch = observationSource.pitch.toDouble()
                    yaw = observationSource.yaw.toDouble()
                    health = observationSource.health.toDouble()
                    foodLevel = observationSource.foodLevel.toDouble()
                    saturationLevel = observationSource.saturationLevel.toDouble()
                    isDead = observationSource.isDead
                    val allItems =
                        sequenceOf(
                            player.inventory.nonEquipmentItems.asSequence(),
                            sequenceOf(
                                player.getItemBySlot(net.minecraft.world.entity.EquipmentSlot.HEAD),
                                player.getItemBySlot(net.minecraft.world.entity.EquipmentSlot.CHEST),
                                player.getItemBySlot(net.minecraft.world.entity.EquipmentSlot.LEGS),
                                player.getItemBySlot(net.minecraft.world.entity.EquipmentSlot.FEET),
                                player.getItemBySlot(net.minecraft.world.entity.EquipmentSlot.OFFHAND),
                            ),
                        ).flatten()
                    inventory.addAll(
                        allItems.map { it.toMessage() }.asIterable(),
                    )

                    if (initialEnvironment.requestRaycast) {
                        raycastResult = player.pick(100.0, 1.0f, false).toMessage(world)
                    } else {
                        // Optimized: dummy hit result
                        raycastResult = hitResult { type = ObservationSpace.HitResult.Type.MISS }
                    }
                    soundSubtitles.addAll(
                        soundListener!!.entries.map { it.toMessage() },
                    )
                    statusEffects.addAll(
                        player.activeEffects.map { it.toMessage() },
                    )
                    for (killStatKey in initialEnvironment.killedStatKeysList) {
                        val key = BuiltInRegistries.ENTITY_TYPE.getValue(Identifier.parse(killStatKey))
                        val stat = player.stats.getValue(Stats.ENTITY_KILLED.get(key))
                        killedStatistics[killStatKey] = stat
                    }
                    for (mineStatKey in initialEnvironment.minedStatKeysList) {
                        val key = BuiltInRegistries.BLOCK.getValue(Identifier.fromNamespaceAndPath("minecraft", mineStatKey))
                        val stat = player.stats.getValue(Stats.BLOCK_MINED.get(key))
                        minedStatistics[mineStatKey] = stat
                    }
                    for (miscStatKey in initialEnvironment.miscStatKeysList) {
                        val key = BuiltInRegistries.CUSTOM_STAT.getValue(Identifier.fromNamespaceAndPath("minecraft", miscStatKey))!!
                        miscStatistics[miscStatKey] =
                            player.stats.getValue(Stats.CUSTOM.get(key))
                    }
                    entityListener?.run {
                        for (entity in entities) {
                            // notify where entity is, what it is (supervised)
                            visibleEntities.add(entity.toMessage())
                        }
                    }
                    for (distance in initialEnvironment.surroundingEntityDistancesList) {
                        val distanceDouble = distance.toDouble()
                        val entitiesWithinDistanceMessage =
                            entitiesWithinDistance {
                                world
                                    .getEntities(
                                        player,
                                        player.boundingBox.inflate(
                                            distanceDouble,
                                            distanceDouble,
                                            distanceDouble,
                                        ),
                                    ) { true }
                                    .forEach { entities.add(it.toMessage()) }
                            }
                        surroundingEntities[distance] = entitiesWithinDistanceMessage
                    }
                    bobberThrown = player.fishing != null
                    experience = player.totalExperience
                    worldTime = world.gameTime // world tick, monotonic increasing
                    lastDeathMessage = deathMessageCollector?.lastDeathMessage?.firstOrNull() ?: ""
                    image2 = imageByteString2
                    ipcHandle = FramebufferCapturer.ipcHandle

                    if (initialEnvironment.requiresSurroundingBlocks) {
                        val blocks = mutableListOf<ObservationSpace.BlockInfo>()
                        val pBlockPos = player.blockPosition()
                        for (i in (pBlockPos.x - 1)..(pBlockPos.x + 1)) {
                            for (j in pBlockPos.y - 1..pBlockPos.y + 1) {
                                for (k in pBlockPos.z - 1..pBlockPos.z + 1) {
                                    val block = world.getBlockState(BlockPos(i, j, k))
                                    blocks.add(
                                        blockInfo {
                                            x = i
                                            y = j
                                            z = k
                                            translationKey = block.block.descriptionId
                                        },
                                    )
                                }
                            }
                        }
                        surroundingBlocks.addAll(blocks)
                    }
                    suffocating = player.isInWall
                    eyeInBlock = player.checkIfCameraBlocked()
                    for (chat in chatList) {
                        chatMessages.add(
                            chatMessageInfo {
                                message = chat.message
                                addedTime = chat.addedTime.toLong()
                                indicator = ""
                            },
                        )
                    }
                    chatList.clear()

                    // Populate biome info if needed
                    if (initialEnvironment.requiresBiomeInfo) {
                        println("Get world")
                        val serverWorld = client.getSingleplayerServer()!!.overworld()
                        println("End Get world")
                        val playerBlockPos = player.blockPosition()
                        val currentPlayerBiome =
                            serverWorld.getUncachedNoiseBiome(
                                net.minecraft.core.QuartPos
                                    .fromBlock(playerBlockPos.x),
                                net.minecraft.core.QuartPos
                                    .fromBlock(playerBlockPos.y),
                                net.minecraft.core.QuartPos
                                    .fromBlock(playerBlockPos.z),
                            )
                        println("Current player biome: $currentPlayerBiome")
                        val biomeCenterFinder = BiomeCenterFinder(serverWorld)
                        val biomeCenter =
                            biomeCenterFinder.calculateBiomeCenter(
                                playerBlockPos,
                                4,
                                currentPlayerBiome,
                            )
                        println("Finish biome center")
                        if (biomeCenter != null) {
                            biomeInfo =
                                biomeInfo {
                                    centerX = biomeCenter.x
                                    centerY = biomeCenter.y
                                    centerZ = biomeCenter.z
                                    biomeName = currentPlayerBiome.unwrapKey().map { it.identifier().toString() }.orElse("")
                                }
                        }
                        val nearbyBiomes1 = biomeCenterFinder.getNearbyBiomes(playerBlockPos, 2)
                        for (biomePos in nearbyBiomes1) {
                            nearbyBiomes.add(
                                nearbyBiome {
                                    x = biomePos.x
                                    y = biomePos.y
                                    z = biomePos.z
                                    biomeName =
                                        biomePos.biome
                                            .unwrapKey()
                                            .map { it.identifier().toString() }
                                            .orElse("")
                                },
                            )
                        }
                        println("Finish nearby biome")
                    }

                    submergedInWater = player.isUnderWater
                    isInLava = player.isInLava
                    submergedInLava = player.isEyeInFluid(FluidTags.LAVA)

                    if (initialEnvironment.requiresHeightmap) {
                        val heightMapProvider = HeightMapProvider()
                        val heightMap = heightMapProvider.getHeightMap(world, player.blockPosition(), 1)
                        for (heightMapInfo in heightMap) {
                            heightInfo.add(
                                heightInfo {
                                    x = heightMapInfo.x
                                    z = heightMapInfo.z
                                    height = heightMapInfo.height
                                    blockName = heightMapInfo.blockName
                                },
                            )
                        }
                    }
                    isOnGround = player.onGround()
                    isTouchingWater = player.isInWater
                    if (initialEnvironment.screenEncodingMode == FramebufferCapturer.ZEROCOPY_TORCH) {
                        ipcHandle =
                            if (captureBackend == Blaze3dCapture.CaptureBackend.BLAZE3D) {
                                VulkanMetalZerocopy.ipcHandle
                            } else {
                                FramebufferCapturer.ipcHandle
                            }
                    }
                    if (initialEnvironment.requiresDepth) {
                        // On the backend-neutral path the depth mixin only *recorded* the copy;
                        // this is where it is finally mapped and linearized. On the OpenGL path
                        // readPendingDepth() returns null and the mixin already has the array.
                        val depthBuffer =
                            Blaze3dCapture.readPendingDepth(
                                FramebufferCapturer.requiresDepthConversion,
                            ) ?: (client.gameRenderer as GameRendererDepthCaptureMixinGetterInterface)
                                .`minecraftEnv$getLastDepthBuffer`()
                        depth.addAll(depthBuffer?.asIterable() ?: emptyList())
                    }
                    if (initialEnvironment.blockCollisionKeysCount > 0) {
                        blockCollisions.addAll(CollisionListener.blockCollisionInfo)
                        CollisionListener.blockCollisionInfo.clear()
                    }
                    if (initialEnvironment.entityCollisionKeysCount > 0) {
                        entityCollisions.addAll(CollisionListener.entityCollisionInfo)
                        CollisionListener.entityCollisionInfo.clear()
                    }

                    val playerVelocity = player.deltaMovement
                    velocityX = playerVelocity.x
                    velocityY = playerVelocity.y
                    velocityZ = playerVelocity.z

                    // Lidar raycast
                    if (initialEnvironment.hasLidarConfig()) {
                        val lidarConfig = initialEnvironment.lidarConfig
                        val horizontalRays = lidarConfig.horizontalRays
                        val maxDistance = lidarConfig.maxDistance.toDouble()
                        val verticalRays = lidarConfig.verticalRays
                        val verticalFov = lidarConfig.verticalFov

                        // Calculate vertical angles
                        val verticalAngles =
                            if (verticalRays == 1) {
                                listOf(lidarConfig.verticalAngle)
                            } else {
                                val halfFov = verticalFov / 2.0f
                                (0 until verticalRays).map { i ->
                                    lidarConfig.verticalAngle - halfFov + (i * verticalFov / (verticalRays - 1))
                                }
                            }
                        val resultRays = mutableListOf<ObservationSpace.LidarRay>()
                        for (verticalAngle in verticalAngles) {
                            for (i in 0 until horizontalRays) {
                                val horizontalAngle = (i.toFloat() * 360.0f / horizontalRays)

                                // Calculate ray direction based on player's yaw and pitch
                                val yaw = player.yRot + horizontalAngle
                                val pitch = player.xRot + verticalAngle

                                // Perform raycast with calculated direction
                                val raycastResult =
                                    performDirectionalRaycast(
                                        player,
                                        world,
                                        yaw.toDouble(),
                                        pitch.toDouble(),
                                        maxDistance,
                                    )

                                resultRays.add(
                                    lidarRay {
                                        distance = raycastResult.distance
                                        hitType = raycastResult.hitType
                                        blockName = raycastResult.blockName
                                        entityName = raycastResult.entityName
                                        angleHorizontal = horizontalAngle
                                        angleVertical = verticalAngle
                                    },
                                )
                            }
                        }
                        lidarResult =
                            lidarResult {
                                this.horizontalRays = horizontalRays
                                this.verticalRays = verticalRays
                                this.maxDistance = maxDistance.toFloat()
                                this.rays.addAll(resultRays)
                            }
                    }
                }
            if (ioPhase == IOPhase.GOT_INITIAL_ENVIRONMENT_SHOULD_SEND_OBSERVATION) {
                ioPhase = IOPhase.GOT_INITIAL_ENVIRONMENT_SENT_OBSERVATION_SKIP_SEND_OBSERVATION
            } else if (ioPhase == IOPhase.READ_ACTION_SHOULD_SEND_OBSERVATION) {
                ioPhase = IOPhase.SENT_OBSERVATION_SHOULD_READ_ACTION
            }
            csvLogger.profileEndPrint(
                "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Prepare/Message",
            )
            csvLogger.profileStartPrint(
                "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Write",
            )
            messageIO.writeObservation(observationSpaceMessage)
            csvLogger.profileEndPrint(
                "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Write",
            )
        } catch (e: IOException) {
            e.printStackTrace()
            stepBarrier.terminate()
            client.stop()

            val threadGroup = Thread.currentThread().threadGroup
            val threads = arrayOfNulls<Thread>(threadGroup.activeCount())
            threadGroup.enumerate(threads)

            for (thread in threads) {
                if (thread == null) {
                    continue
                }
                if (thread != Thread.currentThread()) {
                    thread.interrupt()
                }
            }
            printWithTime("Will exitprocess -3")
            exitProcess(-3)
        }
    }

    override fun runCommand(
        player: LocalPlayer,
        command: String,
    ) {
        var command = command
        printWithTime("Running command: $command")
        csvLogger.log("Running command: $command")
        if (command.startsWith("/")) {
            command = command.substring(1)
        }
        player.connection.sendCommand(command)
        printWithTime("End send command: $command")
        csvLogger.log("End send command: $command")
    }
}

fun Player.checkIfCameraBlocked(): Boolean {
    val f: Float = EntityTypes.PLAYER.dimensions.width() * 0.8f
    val box = AABB.ofSize(this.eyePosition, f.toDouble(), 1.0E-6, f.toDouble())
    return BlockPos.betweenClosedStream(box).anyMatch { pos: BlockPos ->
        val blockState: BlockState = this.level().getBlockState(pos)
        !blockState.isAir &&
            Shapes.joinIsNotEmpty(
                blockState
                    .getCollisionShape(
                        this.level(),
                        pos,
                    ).move(pos.x.toDouble(), pos.y.toDouble(), pos.z.toDouble()),
                Shapes.create(box),
                BooleanOp.AND,
            )
    }
}

data class LidarRayResult(
    val distance: Float,
    val hitType: Int,
    val blockName: String,
    val entityName: String,
)

fun performDirectionalRaycast(
    player: LocalPlayer,
    world: ClientLevel,
    yaw: Double,
    pitch: Double,
    maxDistance: Double,
): LidarRayResult {
    // Calculate ray direction from yaw and pitch
    val yawRad = Math.toRadians(yaw)
    val pitchRad = Math.toRadians(pitch)

    val xDir = -Math.sin(yawRad) * Math.cos(pitchRad)
    val yDir = -Math.sin(pitchRad)
    val zDir = Math.cos(yawRad) * Math.cos(pitchRad)

    val direction = Vec3(xDir, yDir, zDir)
    val start = player.getEyePosition(1.0f)
    val end = start.add(direction.scale(maxDistance))

    // Perform raycast
    val blockHitResult =
        world.clip(
            ClipContext(
                start,
                end,
                ClipContext.Block.OUTLINE,
                ClipContext.Fluid.NONE,
                player,
            ),
        )

    // Check for entity hits
    val entityHitResult =
        ProjectileUtil.getEntityHitResult(
            player,
            start,
            end,
            player.boundingBox.expandTowards(direction.scale(maxDistance)).inflate(1.0),
            { entity -> !entity.isSpectator && entity.isPickable },
            maxDistance * maxDistance,
        )

    // Determine which hit is closer
    val blockDistance =
        if (blockHitResult.type == HitResult.Type.BLOCK) {
            start.distanceTo(blockHitResult.location)
        } else {
            maxDistance
        }

    val entityDistance =
        if (entityHitResult != null) {
            start.distanceTo(entityHitResult.location)
        } else {
            maxDistance
        }

    return when {
        entityDistance < blockDistance -> {
            // Entity hit
            val entity = (entityHitResult as EntityHitResult).entity
            LidarRayResult(
                distance = entityDistance.toFloat(),
                hitType = 2, // ENTITY
                blockName = "",
                entityName = entity.type.descriptionId,
            )
        }

        blockDistance < maxDistance -> {
            // Block hit
            val blockPos = (blockHitResult as BlockHitResult).blockPos
            val block = world.getBlockState(blockPos).block
            LidarRayResult(
                distance = blockDistance.toFloat(),
                hitType = 1, // BLOCK
                blockName = block.descriptionId,
                entityName = "",
            )
        }

        else -> {
            // Miss
            LidarRayResult(
                distance = maxDistance.toFloat(),
                hitType = 0, // MISS
                blockName = "",
                entityName = "",
            )
        }
    }
}
