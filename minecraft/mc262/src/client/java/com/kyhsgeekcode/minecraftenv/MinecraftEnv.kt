@file:OptIn(ExperimentalPathApi::class)

package com.kyhsgeekcode.minecraftenv

import com.google.protobuf.ByteString
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
import net.minecraft.world.entity.EntityType
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
 * NOT yet applied in this file (left for the tasks below - porting them together with this
 * file would have meant guessing at APIs this port couldn't verify without them):
 * - W1 (moving capture to the present()-redirect frame boundary) and W1-a (PacketProcessor
 *   drain) - sendObservation() below still captures directly in END_WORLD_TICK, mc121-style.
 * - W11 (ObservationSource/ServerAuthoritativeSource) - numeric fields below are still read
 *   from the client LocalPlayer, not the server-authoritative ServerPlayer, even though
 *   ObservationSource/ServerAuthoritativeSource already exist (see ObservationSource.kt).
 * - W3/W5 (texture-based capture, present-hook, FramerateLimiter) - the eye-distance ("stereo")
 *   render path below is disabled (throws) rather than ported, because it depends on a
 *   `render(client)` helper that doesn't have a faithful 26.2 equivalent yet: GameRenderer's
 *   immediate-mode re-render (RenderSystem.clear, Framebuffer.beginWrite/endWrite/draw) was
 *   restructured into the extract/render/present pipeline discussed in phase2_plan.md §2.
 *   Single-eye capture (eyeDistance == 0, the common case) does not need that helper and is
 *   fully ported.
 */
class MinecraftEnv :
    ClientModInitializer,
    CommandExecutor {
    private lateinit var initialEnvironment: InitialEnvironment.InitialEnvironmentMessage
    private var soundListener: MinecraftSoundListener? = null
    private var entityListener: EntityRenderListenerImpl? =
        null // tracks the entities rendered in the last tick
    private var resetPhase: ResetPhase = ResetPhase.END_RESET
    private var deathMessageCollector: GetMessagesInterface? = null

    private val tickSynchronizer = TickSynchronizer()
    private val csvLogger = CsvLogger("java_log.csv", enabled = false, profile = false)

    private val variableCommandsAfterReset = mutableListOf<String>()
    private var skipSync = false
    private var ioPhase = IOPhase.BEGINNING
    private var useSharedMemory = false

    override fun onInitializeClient() {
        val isLdPreloadSet = System.getenv("LD_PRELOAD")
        if (isLdPreloadSet != null) {
            println("LD_PRELOAD is set: $isLdPreloadSet")
        } else {
            println("LD_PRELOAD is not set")
        }
        val socket: SocketChannel
        val messageIO: MessageIO
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
                            client.levelRenderer as AddListenerInterface,
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
                // allow server to start tick
                tickSynchronizer.notifyServerTickStart()
                // wait until server tick ends
                csvLogger.profileStartPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/WaitServerTickEnds",
                )
                if (skipSync) {
                    csvLogger.log("Skip waiting server world tick ends")
                } else {
                    csvLogger.log("Wait server world tick ends")
                    tickSynchronizer.waitForServerTickCompletion()
                }
                csvLogger.profileEndPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/WaitServerTickEnds",
                )
                csvLogger.profileStartPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/SendObservation",
                )
                if (ioPhase ==
                    IOPhase.GOT_INITIAL_ENVIRONMENT_SENT_OBSERVATION_SKIP_SEND_OBSERVATION ||
                    ioPhase == IOPhase.SENT_OBSERVATION_SHOULD_READ_ACTION
                ) {
                    // pass
                    csvLogger.log("Skip send observation; $ioPhase")
                } else {
                    csvLogger.log("Real send observation; $ioPhase")
                    sendObservation(messageIO, world)
                }
                csvLogger.profileEndPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/SendObservation",
                )
            },
        )
        ServerTickEvents.START_SERVER_TICK.register(
            ServerTickEvents.StartTick { server: MinecraftServer ->
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
                    tickSynchronizer.waitForClientAction()
                }
                csvLogger.profileEndPrint(
                    "Minecraft_env/onInitialize/StartServerTick/WaitClientAction",
                )
            },
        )
        ServerTickEvents.END_SERVER_TICK.register(
            ServerTickEvents.EndTick { server: MinecraftServer ->
                // allow client to end tick
                printWithTime("Notify server tick completion")
                csvLogger.log("Notify server tick completion")
                csvLogger.profileStartPrint(
                    "Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation",
                )
                tickSynchronizer.notifyClientSendObservation()
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
            tickSynchronizer.terminate()
            e.printStackTrace()
            exitProcess(-1)
        } catch (e: Exception) {
            tickSynchronizer.terminate()
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
            tickSynchronizer.terminate()
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
        if (FramebufferCapturer.checkGLEW()) {
            printWithTime("GLEW initialized")
        } else {
            printWithTime("GLEW not initialized")
            throw RuntimeException("GLEW not initialized")
        }
        val mainRenderTarget = client.gameRenderer.mainRenderTarget()
        val glColorTexture = mainRenderTarget.colorTexture as? com.mojang.blaze3d.opengl.GlTexture
        if (glColorTexture == null) {
            // 26_2_phase2_plan.md D2/W4: fail fast rather than silently produce garbage
            // frames if the GL backend isn't in use.
            throw IllegalStateException(
                "Expected an OpenGL color texture on the main render target " +
                    "(got ${mainRenderTarget.colorTexture}); capture requires the GL backend (see phase2_plan.md D2).",
            )
        }
        val colorTextureId = glColorTexture.glId()
        if (initialEnvironment.screenEncodingMode == FramebufferCapturer.ZEROCOPY) {
            // TODO(26_2_phase2_plan.md W3): initializeZeroCopy still expects the mc121-era
            // FBO-attachment-based signature (colorAttachment/depthAttachment ints from
            // Framebuffer, which no longer exists). Needs the texture-based native rewrite.
            printWithTime("ZEROCOPY mode requested but not yet ported for 26.2 (W3 pending)")
        }

        // request stats from server
        // TODO(W11): use ObservationSource.ServerAuthoritativeSource instead of the client
        // player directly, per 26_2_phase2_plan.md §1.3 - not yet wired into this file.
        csvLogger.profileStartPrint(
            "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Prepare",
        )
        client.connection?.send(
            ServerboundClientCommandPacket(ServerboundClientCommandPacket.Action.REQUEST_STATS),
        )
        try {
            val imageByteString1: ByteString
            val imageByteString2: ByteString
            val oldX = player.x
            val oldY = player.y
            val oldZ = player.z
            val pos = Vec3(oldX, oldY, oldZ)
            if (initialEnvironment.eyeDistance > 0) {
                // TODO(26_2_phase2_plan.md W1/W3/W5): the stereo (eyeDistance>0) capture path
                // needs the extract/render/present redesign described in phase2_plan.md §2 -
                // mc121's render(client) helper (RenderSystem.clear + Framebuffer.beginWrite/
                // endWrite/draw + GameRenderer.render(RenderTickCounter, boolean)) has no
                // faithful 26.2 equivalent; that machinery was restructured into
                // extract()/render()/present(). Fail loud instead of silently capturing the
                // wrong thing.
                throw UnsupportedOperationException(
                    "eyeDistance > 0 (stereo capture) is not yet supported on mc262 - " +
                        "pending the RenderMixin/present-hook redesign (phase2_plan.md W1/W5).",
                )
            } else {
                csvLogger.profileStartPrint(
                    "Minecraft_env/onInitialize/EndWorldTick/SendObservation/Prepare/SingleEye/ByteString",
                )
                imageByteString1 =
                    FramebufferCapturer.captureFramebuffer(
                        colorTextureId,
                        // TODO(W3): no FBO int exists on 26.2's RenderTarget anymore; 0 is a
                        // placeholder until the native capture path is rewritten to be
                        // texture-based (native owning its own FBO, see phase2_plan.md §2.3).
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
                    pitch = player.xRot.toDouble()
                    yaw = player.yRot.toDouble()
                    health = player.health.toDouble()
                    foodLevel = player.foodData.foodLevel.toDouble()
                    saturationLevel = player.foodData.saturationLevel.toDouble()
                    isDead = player.isDeadOrDying
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
                                net.minecraft.core.QuartPos.fromBlock(playerBlockPos.x),
                                net.minecraft.core.QuartPos.fromBlock(playerBlockPos.y),
                                net.minecraft.core.QuartPos.fromBlock(playerBlockPos.z),
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
                                    biomeName = biomePos.biome.unwrapKey().map { it.identifier().toString() }.orElse("")
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
                    if (initialEnvironment.screenEncodingMode == FramebufferCapturer.ZEROCOPY) {
                        ipcHandle = FramebufferCapturer.ipcHandle
                    }
                    if (initialEnvironment.requiresDepth) {
                        depth.addAll(
                            (client.gameRenderer as GameRendererDepthCaptureMixinGetterInterface)
                                .`minecraftEnv$getLastDepthBuffer`()
                                ?.asIterable()
                                ?: emptyList(),
                        )
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
            tickSynchronizer.terminate()
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
