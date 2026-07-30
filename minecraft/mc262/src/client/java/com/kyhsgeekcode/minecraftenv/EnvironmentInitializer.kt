package com.kyhsgeekcode.minecraftenv

import com.kyhsgeekcode.minecraftenv.mixin.ChatVisibleMessageAccessor
import com.kyhsgeekcode.minecraftenv.mixin.WindowSizeAccessor
import com.kyhsgeekcode.minecraftenv.proto.InitialEnvironment
import com.kyhsgeekcode.minecraftenv.proto.InitialEnvironment.InitialEnvironmentMessage
import net.minecraft.client.Minecraft
import net.minecraft.client.NarratorStatus
import net.minecraft.client.gui.components.Button
import net.minecraft.client.gui.components.CycleButton
import net.minecraft.client.gui.components.EditBox
import net.minecraft.client.gui.components.TabButton
import net.minecraft.client.gui.components.events.GuiEventListener
import net.minecraft.client.gui.components.tabs.TabNavigationBar
import net.minecraft.client.gui.screens.AccessibilityOnboardingScreen
import net.minecraft.client.gui.screens.GenericMessageScreen
import net.minecraft.client.gui.screens.PresetFlatWorldScreen
import net.minecraft.client.gui.screens.TitleScreen
import net.minecraft.client.gui.screens.worldselection.CreateWorldScreen
import net.minecraft.client.gui.screens.worldselection.SelectWorldScreen
import net.minecraft.client.gui.screens.worldselection.WorldSelectionList
import net.minecraft.client.input.KeyEvent
import net.minecraft.client.player.LocalPlayer
import net.minecraft.client.tutorial.TutorialSteps
import net.minecraft.server.MinecraftServer
import net.minecraft.sounds.SoundSource
import net.minecraft.world.level.GameType
import net.minecraft.world.level.storage.LevelResource
import org.lwjgl.glfw.GLFW
import java.nio.file.Files
import java.util.concurrent.CompletableFuture
import kotlin.io.path.Path
import kotlin.io.path.copyTo

interface CommandExecutor {
    fun runCommand(
        server: LocalPlayer,
        command: String,
    )
}

// mc121's ButtonWidget.onPress() took no arguments; 26.2's AbstractButton.onPress() now takes
// an InputWithModifiers (real presses come from a KeyEvent/MouseButtonEvent). For a synthetic
// GUI-automation click we don't have a real input event, so we build a no-modifier KeyEvent -
// CycleButton.onPress() reads hasShiftDown() to decide cycle direction, so modifiers=0 is load
// bearing here (keeps the forward-cycle-until-match loops below behaving as before).
private val SYNTHETIC_PRESS = KeyEvent(0, 0, 0)

// Referenced from EnvironmentInitializer.onWorldTick; declared here (rather than in
// MinecraftEnv.kt, which doesn't exist yet in mc262 - see 26_2_phase2_plan.md task #4) since
// it's conceptually part of chat-message bookkeeping for the init-done handshake.
internal val chatList = mutableListOf<ChatMessageRecord>()

/**
 * Port of mc121's EnvironmentInitializer onto 26.2 (Mojmap). GUI class/method renames are
 * confirmed against the decompiled 26.2 sources (see 26_2_phase2_plan.md / mixin deep dive).
 *
 * IMPORTANT CAVEAT: the screen-walking automation below (createNewWorldAndEnterUsingGUI,
 * createEmptyWorldAndEnterUsingGUI) could not be runtime-verified in this environment - this
 * sandbox's native build is blocked by a pre-existing CMake/JNI detection issue, so the game
 * has only been launched far enough to reach the title screen, not far enough to click
 * through world creation. CreateWorldScreen's internals also changed more than a rename in
 * 26.2: it's now driven by a typed `WorldCreationUiState` (getGameMode()/setAllowCommands()/
 * getWorldType() etc.) rather than mc121's find-widget-by-button-text approach. The
 * text-matching traversal here is preserved from mc121 as the best available port, but should
 * be re-verified against an actual running 26.2 client, and is a good candidate to replace
 * with direct WorldCreationUiState calls once that's done.
 */
class EnvironmentInitializer(
    private val initialEnvironment: InitialEnvironmentMessage,
    private val csvLogger: CsvLogger,
) {
    private var hasRunInitWorld: Boolean = false
        private set
    var initWorldFinished: Boolean = false
        private set

    private lateinit var minecraftServer: MinecraftServer
    private lateinit var player: LocalPlayer
    private var hasMinimizedWindow: Boolean = false

    private var initializedClient = false
    private var finishedEnteringWorld = false
    private var shouldReloadResourcePack = false
    private var reloadResourcePackFuture: CompletableFuture<Void>? = null

    fun onClientTick(client: Minecraft) {
        if (finishedEnteringWorld && initializedClient) {
            return
        }
        if (shouldReloadResourcePack) {
            shouldReloadResourcePack = false
        }
        csvLogger.profileStartPrint("Minecraft_env/onInitialize/ClientTick/EnvironmentInitializer/onClientTick")
        disableNarrator(client)
        val currentScreenBeforeEnteringWorld = client.gui.screen()
        if (currentScreenBeforeEnteringWorld is AccessibilityOnboardingScreen) {
            // Fresh options.txt files make vanilla show this screen before the title
            // screen. Our GUI-driving logic below doesn't know this screen, so left
            // unhandled it blocks startup forever. Dismiss it the same way Escape/Done
            // would, which also persists onboardAccessibility=false so it won't reappear.
            currentScreenBeforeEnteringWorld.onClose()
            csvLogger.profileEndPrint("Minecraft_env/onInitialize/ClientTick/EnvironmentInitializer/onClientTick")
            return
        }
        if (!initialEnvironment.levelDisplayNameToPlay.isNullOrEmpty()) {
            enterExistingWorldUsingGUI(client, initialEnvironment.levelDisplayNameToPlay)
        } else {
            createNewWorldAndEnterUsingGUI(client)
        }
        val window = Minecraft.getInstance().window
        val windowSizeGetter = (window as WindowSizeAccessor)
        val glfwScaleWidth = FloatArray(1)
        val glfwScaleHeight = FloatArray(1)
        GLFW.glfwGetWindowContentScale(window.handle(), glfwScaleWidth, glfwScaleHeight)
        val desiredWindowWidth = (initialEnvironment.imageSizeX / glfwScaleWidth[0]).toInt()
        val desiredWindowHeight = (initialEnvironment.imageSizeY / glfwScaleHeight[0]).toInt()
        if (windowSizeGetter.windowedWidth != desiredWindowWidth ||
            windowSizeGetter.windowedHeight != desiredWindowHeight
        ) {
            window.setWindowed(desiredWindowWidth, desiredWindowHeight)
            // mc121 followed a manual window resize with an explicit onResolutionChanged()
            // call; no public equivalent was found on 26.2's Minecraft/Window. Window's own
            // GLFW framebuffer/resize callbacks (onFramebufferResize/onResize) are private and
            // GLFW-callback-driven, so setWindowed() should trigger them on its own - but this
            // is unverified without a runtime check.
        }
        if (!hasMinimizedWindow) {
            GLFW.glfwIconifyWindow(window.handle())
            hasMinimizedWindow = true
        }
        disablePauseOnLostFocus(client)
        disableOnboardAccessibility(client)
        setRenderDistance(client, initialEnvironment.renderDistance)
        setSimulationDistance(client, initialEnvironment.simulationDistance)
        disableVSync(client)
        disableSound(client)
        disableTutorial(client)
        setMaxFPSToUnlimited(client)
        if (initialEnvironment.noFovEffect) {
            setFovEffectDisabled(client)
        }
        checkRenderBackend(client)
        initializedClient = true
        csvLogger.profileEndPrint("Minecraft_env/onInitialize/ClientTick/EnvironmentInitializer/onClientTick")
    }

    // W4 (26_2_phase2_plan.md D2): resolve and report the capture path once at initialization,
    // rather than discovering a bad combination on the first captureFramebuffer() call in
    // MinecraftEnv.sendObservation(). Checked here (once, right before initializedClient is set)
    // rather than on the very first client tick, because the main render target's color texture
    // isn't guaranteed to exist yet before a world has been entered - by this point in
    // onClientTick a world is already up and the render pipeline has run many frames.
    //
    // Both backends are supported now (docs/26_2_vulkan_capture.md). ZEROCOPY_TORCH works on
    // OpenGL unconditionally, and on Vulkan only when one of the cross-API interop extension pairs
    // actually got enabled at device-creation time (see VulkanBackendInteropExtensionMixin):
    // VK_EXT_metal_objects + VK_EXT_external_memory_metal (-Dcraftground.enableMetalObjects=true,
    // verified on Apple Silicon), or VK_KHR_external_memory_fd/win32
    // (-Dcraftground.enableCudaInterop=true, NOT verified - no CUDA/Linux hardware in this dev
    // environment) - otherwise VulkanZerocopy has nothing to import a shared surface/buffer into.
    private fun checkRenderBackend(client: Minecraft) {
        val backendName =
            com.mojang.blaze3d.systems.RenderSystem
                .getDevice()
                .deviceInfo
                .backendName()
        val colorTexture =
            client.gameRenderer.mainRenderTarget().colorTexture
                ?: throw IllegalStateException(
                    "Main render target has no color texture; cannot determine the capture backend " +
                        "(rendering backend is '$backendName').",
                )
        val captureBackend = Blaze3dCapture.backendFor(colorTexture)
        println(
            "CraftGround: rendering backend '$backendName' -> capture path $captureBackend " +
                "(color texture ${colorTexture.javaClass.simpleName})",
        )
        if (captureBackend != Blaze3dCapture.CaptureBackend.OPENGL &&
            initialEnvironment.screenEncodingMode == FramebufferCapturer.ZEROCOPY_TORCH
        ) {
            val vulkanZerocopySupported =
                captureBackend == Blaze3dCapture.CaptureBackend.BLAZE3D &&
                    (VulkanMetalObjectsState.metalObjectsEnabled || VulkanCudaObjectsState.cudaInteropEnabled)
            if (!vulkanZerocopySupported) {
                throw IllegalStateException(
                    "ZEROCOPY_TORCH capture requires the OpenGL rendering backend, or Vulkan with " +
                        "-Dcraftground.enableMetalObjects=true on a device supporting " +
                        "VK_EXT_metal_objects + VK_EXT_external_memory_metal, or " +
                        "-Dcraftground.enableCudaInterop=true on a device supporting " +
                        "VK_KHR_external_memory_fd/win32 (active backend '$backendName'). Use RAW " +
                        "or PNG otherwise (see docs/26_2_vulkan_capture.md).",
                )
            }
        }
    }

    private fun enterExistingWorldUsingGUI(
        client: Minecraft,
        levelDisplayName: String,
    ) {
        val screen = client.gui.screen() ?: return
        println("Entering existing world: $levelDisplayName")
        when (screen) {
            is TitleScreen -> {
                screen
                    .children()
                    .find {
                        it is Button && it.message.string == "Singleplayer"
                    }?.let {
                        it as Button
                        it.onPress(SYNTHETIC_PRESS)
                        return
                    }
            }

            is SelectWorldScreen -> {
                // search for the world to open
                var levelList: WorldSelectionList? = null
                for (child in screen.children()) {
                    if (child is WorldSelectionList) {
                        levelList = child
                        break
                    }
                }
                if (levelList != null) {
                    for (child in levelList.children()) {
                        if (child is WorldSelectionList.LoadingHeader) {
                            return
                        }
                        if (child is WorldSelectionList.WorldListEntry) {
                            if (!child.levelSummary.primaryActionActive()) {
                                continue
                            }
                            if (child.levelName == levelDisplayName) {
                                child.joinWorld()
                                finishedEnteringWorld = true
                                return
                            } else {
                                println("Level display name: ${child.levelName}!= $levelDisplayName")
                            }
                        }
                    }
                } else {
                    println("Level list not found")
                }
            }

            is GenericMessageScreen -> {
                println("Message screen: ${screen.title.string}")
            }

            is CreateWorldScreen -> {
            }

            else -> {
                println("Unknown screen: $screen")
            }
        }
    }

    private fun createNewWorldAndEnterUsingGUI(client: Minecraft) {
        val screen = client.gui.screen() ?: return
        when (screen) {
            is TitleScreen -> {
                screen
                    .children()
                    .find {
                        it is Button && it.message.string == "Singleplayer"
                    }?.let {
                        it as Button
                        it.onPress(SYNTHETIC_PRESS)
                        return
                    }
            }

            is SelectWorldScreen -> {
                var createButton: Button? = null
                for (child in screen.children()) {
                    if (child is Button && child.message.string == "Create New World") {
                        createButton = child
                    }
                }
                createButton?.onPress(SYNTHETIC_PRESS)
            }

            is CreateWorldScreen -> {
                var createButton: Button? = null
                val cheatRequested = true
                var indexOfWorldSettingTab = -1
                var cheatButton: CycleButton<*>? = null
                var settingTabWidget: TabNavigationBar? = null
                var worldTypeButton: CycleButton<*>? = null
                for (child in screen.children()) {
                    // search for tab navigation widget, to find index of world settings tab
                    if (indexOfWorldSettingTab == -1 && child is TabNavigationBar) {
                        settingTabWidget = child
                        for (i in child.children().indices) {
                            val tabChild: GuiEventListener = child.children()[i]
                            if (tabChild is TabButton) {
                                if (tabChild.message.string == "World") {
                                    indexOfWorldSettingTab = i
                                }
                            }
                        }
                    }
                    // search for create button
                    if (createButton == null && child is Button) {
                        if (child.message.string == "Create New World") {
                            createButton = child
                        }
                    }
                    // search for cheat button
                    if (cheatButton == null && child is CycleButton<*>) {
                        if (child.message.string.startsWith("Allow Commands")) {
                            cheatButton = child
                        } else {
                            println("Cheat button is not found, and the text is ${child.message.string}")
                        }
                    }
                }
                // Set allow cheats to requested
                if (cheatButton != null) {
                    setupAllowCheats(cheatButton, cheatRequested)
                } else {
                    println("Cheat button not found")
                    throw Exception("Cheat button not found")
                }
                // Select world settings tab
                settingTabWidget!!.selectTab(indexOfWorldSettingTab, false)
                // Search for seed input
                if (initialEnvironment.seed.isNotEmpty()) {
                    for (child in screen.children()) {
                        if (child is EditBox) {
                            child.setValue(initialEnvironment.seed.toString())
                        }
                    }
                }
                if (initialEnvironment.worldType == InitialEnvironment.WorldType.SUPERFLAT) {
                    for (child in screen.children()) {
                        if (worldTypeButton == null && child is CycleButton<*>) {
                            if (child.message.string.startsWith("World Type")) {
                                worldTypeButton = child
                            }
                        }
                    }
                    if (worldTypeButton != null) {
                        while (!worldTypeButton.message.string.endsWith("flat")) {
                            worldTypeButton.onPress(SYNTHETIC_PRESS)
                        }
                    }
                }
                createButton?.onPress(SYNTHETIC_PRESS)
                finishedEnteringWorld = true
            }
        }
    }

    private fun createEmptyWorldAndEnterUsingGUI(client: Minecraft) {
        when (val screen = client.gui.screen()) {
            is TitleScreen -> {
                screen
                    .children()
                    .find {
                        it is Button && it.message.string == "Singleplayer"
                    }?.let {
                        it as Button
                        it.onPress(SYNTHETIC_PRESS)
                        return
                    }
            }

            is SelectWorldScreen -> {
                var createButton: Button? = null
                for (child in screen.children()) {
                    if (child is Button && child.message.string == "Create New World") {
                        createButton = child
                    }
                }
                createButton?.onPress(SYNTHETIC_PRESS)
            }

            is CreateWorldScreen -> {
                var createButton: Button? = null
                val cheatRequested = true
                var indexOfWorldSettingTab = -1
                var cheatButton: CycleButton<*>? = null
                var settingTabWidget: TabNavigationBar? = null
                var worldTypeButton: CycleButton<*>? = null
                var customizeFlatmapButton: Button? = null
                for (child in screen.children()) {
                    if (indexOfWorldSettingTab == -1 && child is TabNavigationBar) {
                        settingTabWidget = child
                        for (i in child.children().indices) {
                            val tabChild: GuiEventListener = child.children()[i]
                            if (tabChild is TabButton) {
                                if (tabChild.message.string == "World") {
                                    indexOfWorldSettingTab = i
                                }
                            }
                        }
                    }
                    if (createButton == null && child is Button) {
                        if (child.message.string == "Create New World") {
                            createButton = child
                        }
                    }
                    if (cheatButton == null && child is CycleButton<*>) {
                        if (child.message.string.startsWith("Allow Commands")) {
                            cheatButton = child
                        } else {
                            println("Cheat button is not found, and the text is ${child.message.string}")
                        }
                    }
                }
                if (cheatButton != null) {
                    setupAllowCheats(cheatButton, cheatRequested)
                } else {
                    println("Cheat button not found")
                    throw Exception("Cheat button not found")
                }
                settingTabWidget!!.selectTab(indexOfWorldSettingTab, false)
                if (initialEnvironment.seed.isNotEmpty()) {
                    for (child in screen.children()) {
                        if (child is EditBox) {
                            child.setValue(initialEnvironment.seed.toString())
                        }
                    }
                }
                if (initialEnvironment.worldType == InitialEnvironment.WorldType.SUPERFLAT) {
                    for (child in screen.children()) {
                        if (worldTypeButton == null &&
                            child is CycleButton<*> &&
                            child.message.string.startsWith("World Type")
                        ) {
                            worldTypeButton = child
                        }
                        if (customizeFlatmapButton == null && child is Button && child.message.string.startsWith("Customize")) {
                            customizeFlatmapButton = child
                        }
                    }
                    if (worldTypeButton != null) {
                        while (!worldTypeButton.message.string.endsWith("flat")) {
                            worldTypeButton.onPress(SYNTHETIC_PRESS)
                        }
                    }
                    if (customizeFlatmapButton != null) {
                        customizeFlatmapButton.onPress(SYNTHETIC_PRESS)
                    }
                }
                createButton?.onPress(SYNTHETIC_PRESS)
            }

            is PresetFlatWorldScreen -> {
            }

            else -> {}
        }
    }

    private fun disableSound(client: Minecraft) {
        client.options?.let {
            it.getSoundSourceOptionInstance(SoundSource.MASTER).set(0.0)
        }
    }

    private fun disableNarrator(client: Minecraft) {
        val options = client.options
        if (options != null) {
            if (options.narrator().get() != NarratorStatus.OFF) {
                options.narrator().set(NarratorStatus.OFF)
                options.save()
                println("Disabled narrator")
            }
        }
    }

    private fun disableTutorial(client: Minecraft) {
        client.tutorial?.setStep(TutorialSteps.NONE)
    }

    private fun disableVSync(client: Minecraft) {
        val options = client.options
        if (options != null) {
            if (options.enableVsync().get()) {
                options.enableVsync().set(false)
                client.options.save()
                println("Disabled VSync")
            }
        }
    }

    private fun setSimulationDistance(
        client: Minecraft,
        simulationDistance: Int,
    ) {
        val options = client.options
        if (options != null) {
            if (options.simulationDistance().get() != simulationDistance) {
                options.simulationDistance().set(simulationDistance)
                client.options.save()
                println("Set simulation distance to $simulationDistance")
            }
        }
    }

    private fun setRenderDistance(
        client: Minecraft,
        renderDistance: Int,
    ) {
        val options = client.options
        if (options != null) {
            if (options.renderDistance().get() != renderDistance) {
                options.renderDistance().set(renderDistance)
                client.options.save()
                println("Set render distance to $renderDistance")
            }
        }
    }

    fun reset(
        chatComponent: net.minecraft.client.gui.components.ChatComponent,
        commandExecutor: CommandExecutor,
        variableCommandAfterReset: List<String>,
    ) {
        println("Resetting...")
        hasRunInitWorld = false
        initWorldFinished = false
        chatComponent.clearMessages(true)
        onWorldTick(null, chatComponent, commandExecutor, variableCommandAfterReset)
    }

    fun onWorldTick(
        minecraftServer: MinecraftServer?,
        chatComponent: net.minecraft.client.gui.components.ChatComponent,
        commandExecutor: CommandExecutor,
        variableCommandsAfterReset: List<String>,
    ) {
        player = Minecraft.getInstance().player ?: return

        // Get the chat messages to check if the initialization is done, and clear the chat
        val messages = ArrayList((chatComponent as ChatVisibleMessageAccessor).allMessages)
        val hasInitFinishMessage =
            messages.find {
                it.content().string.contains("Initialization Done")
            } != null
        initWorldFinished = (initWorldFinished || hasInitFinishMessage) && reloadResourcePackFuture?.isDone != false
        // TODO: Do not clear the chat, and delete only the message related to the initialization.
        // Do not clear the chat related to the advancements
        messages.forEach { it ->
            val content = it.content().string
            chatList.add(
                ChatMessageRecord(
                    it.addedTime(),
                    content,
                ),
            )
        }
        chatComponent.clearMessages(true)
        if (hasRunInitWorld) {
            return
        }
        // copy the path to world file
        minecraftServer?.getWorldPath(LevelResource.GENERATED_DIR)?.let { path ->
            println("World path: $path")
            // path / minecraft / structures / name.nbt
            val structuresPath = path.resolve("minecraft").resolve("structures")
            if (!Files.exists(structuresPath)) {
                Files.createDirectories(structuresPath)
            }
            for (structure in initialEnvironment.structurePathsList) {
                val structureName = structure.substringAfterLast('/')
                val targetPath = structuresPath.resolve(structureName)
                val sourcePath = Path(structure)
                println("Copying structure file: $sourcePath to $targetPath")
                sourcePath.copyTo(targetPath, true)
            }
        } ?: run {
            println("World path not found; server: $minecraftServer")
        }

        // TODO: 26.2's world-specific resource pack loading path (mc121's
        // client.serverResourcePackProvider / IntegratedServerLoader.WORLD_PACK_ID) could not
        // be located in the decompiled 26.2 sources during this port - ServerPackManager
        // exists but its API surface for a world's own resourcepacks/resources.zip wasn't
        // confirmed. The zip is still copied to LevelResource.MAP_RESOURCE_FILE below; only
        // the "tell the client to actually load it" step is missing.
        minecraftServer?.getWorldPath(LevelResource.MAP_RESOURCE_FILE)?.let { targetZipPath ->
            println("Copying resource zip file to: $targetZipPath")
            val sourcePath = Path(initialEnvironment.resourceZipPath)
            if (!Files.exists(sourcePath)) {
                println("Resource zip path not found: $sourcePath")
                return@let
            }
            println("Copying resource zip file: $sourcePath to $targetZipPath")
            // A freshly created world has no "resourcepacks/" directory yet (vanilla only
            // creates it lazily elsewhere) - copyTo doesn't create the target's parent dirs,
            // so without this a fresh world throws NoSuchFileException here every time.
            Files.createDirectories(targetZipPath.parent)
            sourcePath.copyTo(targetZipPath, true)
            println("(gap) Not yet reloading the world resource pack - see TODO above")
        } ?: run {
            println("Resource zip path not found; server: $minecraftServer")
        }

        minecraftServer?.getWorldPath(LevelResource.ROOT)?.let { rootPath ->
            val dataPath = rootPath.resolve("data")
            println("Copying resource zip file to: $dataPath")
            val mapSrcPath = Path(initialEnvironment.mapDirPath)
            if (!Files.exists(mapSrcPath)) {
                println("Map directory path not found: $mapSrcPath")
                return@let
            }
            println("Copying map directory: $mapSrcPath to $dataPath")
            Files.createDirectories(dataPath)
            mapSrcPath.toFile().listFiles()?.forEach { file ->
                if (file.isDirectory) {
                    return@forEach
                } else {
                    val targetPath = dataPath.resolve(file.name)
                    file.copyTo(targetPath.toFile(), true)
                    println("Copied file: ${file.name} to $targetPath")
                }
            }
            println("Copied all files from $mapSrcPath to $dataPath")
        } ?: run {
            println("Root path not found; server: $minecraftServer")
        }

        // NOTE: should be called only once when initial environment is set
        val myCommandExecutor = { player: LocalPlayer, c: String ->
            commandExecutor.runCommand(player, c)
        }
        setUnlimitedTPS(myCommandExecutor)
        for (command in initialEnvironment.initialExtraCommandsList) {
            commandExecutor.runCommand(this.player, "/$command")
        }
        for (command in variableCommandsAfterReset) {
            commandExecutor.runCommand(this.player, "/$command")
        }
        commandExecutor.runCommand(this.player, "/say Initialization Done")
        initWorldFinished = false
        hasRunInitWorld = true
    }

    // Set the TPS to virtually unlimited (26_2_phase2_plan.md W6 - the vanilla /tick rate
    // command backs onto TickRateManager.setTickRate() in 26.2 same as it did in mc121, so
    // this needs no version-specific change).
    private fun setUnlimitedTPS(commandExecutor: (LocalPlayer, String) -> Unit) {
        commandExecutor(player, "/tick rate 10000")
    }

    private fun setupAllowCheats(
        cheatButton: CycleButton<*>,
        cheatRequested: Boolean,
    ) {
        val testString = if (cheatRequested) "ON" else "OFF"
        while (!cheatButton.message.string.endsWith(testString)) {
            cheatButton.onPress(SYNTHETIC_PRESS)
        }
    }

    private fun setupGameMode(
        gameModeButton: CycleButton<*>,
        gameModeRequested: GameType,
    ) {
        val testString = gameModeRequested.name
        while (!gameModeButton.message.string.endsWith(testString)) {
            gameModeButton.onPress(SYNTHETIC_PRESS)
        }
    }

    private fun setupNoWeatherCycle(commandExecutor: (LocalPlayer, String) -> Unit) {
        commandExecutor(
            player,
            "/gamerule doWeatherCycle false",
        )
    }

    private fun disablePauseOnLostFocus(client: Minecraft) {
        val options = client.options
        if (options != null) {
            if (options.pauseOnLostFocus) {
                println("Disabled pause on lost focus")
                options.pauseOnLostFocus = false
                client.options.save()
            }
        }
    }

    private fun disableOnboardAccessibility(client: Minecraft) {
        val options = client.options
        if (options != null) {
            if (options.onboardAccessibility) {
                println("Disabled onboardAccessibility")
                options.onboardAccessibility = false
                client.options.save()
            }
        }
    }

    // TODO: 26.2's HUD-hidden toggle (mc121's Options.hudHidden) could not be located in the
    // decompiled sources during this port. Not wired up yet - if the HUD (crosshair/hotbar/
    // health bar) renders over captured frames, this needs to be found and reinstated.
    private fun setHudHidden(
        client: Minecraft,
        hudHidden: Boolean,
    ) {
    }

    private fun setMaxFPSToUnlimited(client: Minecraft) {
        val options = client.options
        if (options != null) {
            if (options.framerateLimit().get() < 260) { // unlimited
                options.framerateLimit().set(260)
                client.options.save()
                println("Set max fps to 260")
            }
        }
    }

    private fun setFovEffectDisabled(client: Minecraft) {
        val options = client.options
        if (options != null) {
            if (options.fovEffectScale().get() != 0.0) {
                options.fovEffectScale().set(0.0)
                client.options.save()
                println("Disabled fov effect")
            }
        }
    }
}
