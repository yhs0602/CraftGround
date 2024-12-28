package com.kyhsgeekcode.minecraftenv

import com.kyhsgeekcode.minecraftenv.mixin.ChatVisibleMessageAccessor
import com.kyhsgeekcode.minecraftenv.mixin.WindowSizeAccessor
import com.kyhsgeekcode.minecraftenv.proto.InitialEnvironment
import com.kyhsgeekcode.minecraftenv.proto.InitialEnvironment.InitialEnvironmentMessage
import net.minecraft.client.MinecraftClient
import net.minecraft.client.gui.Element
import net.minecraft.client.gui.hud.ChatHud
import net.minecraft.client.gui.screen.MessageScreen
import net.minecraft.client.gui.screen.TitleScreen
import net.minecraft.client.gui.screen.world.CreateWorldScreen
import net.minecraft.client.gui.screen.world.CustomizeFlatLevelScreen
import net.minecraft.client.gui.screen.world.SelectWorldScreen
import net.minecraft.client.gui.screen.world.WorldListWidget
import net.minecraft.client.gui.widget.ButtonWidget
import net.minecraft.client.gui.widget.CyclingButtonWidget
import net.minecraft.client.gui.widget.NarratedMultilineTextWidget
import net.minecraft.client.gui.widget.TabButtonWidget
import net.minecraft.client.gui.widget.TabNavigationWidget
import net.minecraft.client.gui.widget.TextFieldWidget
import net.minecraft.client.network.ClientPlayerEntity
import net.minecraft.client.option.NarratorMode
import net.minecraft.client.tutorial.TutorialStep
import net.minecraft.server.MinecraftServer
import net.minecraft.sound.SoundCategory
import net.minecraft.util.WorldSavePath
import net.minecraft.world.GameMode
import org.lwjgl.glfw.GLFW
import java.nio.file.Files
import kotlin.io.path.Path
import kotlin.io.path.copyTo

interface CommandExecutor {
    fun runCommand(
        server: ClientPlayerEntity,
        command: String,
    )
}

class EnvironmentInitializer(
    private val initialEnvironment: InitialEnvironmentMessage,
    private val csvLogger: CsvLogger,
) {
    var hasRunInitWorld: Boolean = false
        private set
    var initWorldFinished: Boolean = false
        private set

    private lateinit var minecraftServer: MinecraftServer
    private lateinit var player: ClientPlayerEntity
    var hasMinimizedWindow: Boolean = false

    fun onClientTick(client: MinecraftClient) {
        csvLogger.profileStartPrint("Minecraft_env/onInitialize/ClientTick/EnvironmentInitializer/onClientTick")
        disableNarrator(client)
        if (!initialEnvironment.levelDisplayNameToPlay.isNullOrEmpty()) {
            enterExistingWorldUsingGUI(client, initialEnvironment.levelDisplayNameToPlay)
        } else {
            createNewWorldAndEnterUsingGUI(client)
        }
        val window = MinecraftClient.getInstance().window
        val windowSizeGetter = (window as WindowSizeAccessor)
        if (windowSizeGetter.windowedWidth != initialEnvironment.imageSizeX ||
            windowSizeGetter.windowedHeight != initialEnvironment.imageSizeY
        ) {
            window.setWindowedSize(initialEnvironment.imageSizeX, initialEnvironment.imageSizeY)
        }
        if (!hasMinimizedWindow) {
            GLFW.glfwIconifyWindow(window.handle)
            hasMinimizedWindow = true
        }
        disablePauseOnLostFocus(client)
        disableOnboardAccessibility(client)
        setHudHidden(client, initialEnvironment.hudHidden)
        setRenderDistance(client, initialEnvironment.renderDistance)
        setSimulationDistance(client, initialEnvironment.simulationDistance)
        disableVSync(client)
        disableSound(client)
        disableTutorial(client)
        setMaxFPSToUnlimited(client)
        if (initialEnvironment.noFovEffect) {
            setFovEffectDisabled(client)
        }
        csvLogger.profileEndPrint("Minecraft_env/onInitialize/ClientTick/EnvironmentInitializer/onClientTick")
    }

    private fun enterExistingWorldUsingGUI(
        client: MinecraftClient,
        levelDisplayName: String,
    ) {
        if (client.currentScreen == null) {
            return
        }
        println("Entering existing world: $levelDisplayName")
        when (val screen = client.currentScreen) {
            is TitleScreen -> {
                screen
                    .children()
                    .find {
                        it is ButtonWidget && it.message.string == "Singleplayer"
                    }?.let {
                        it as ButtonWidget
                        it.onPress()
                        return
                    }
            }

            is SelectWorldScreen -> {
                // search for the world to open
                var levelList: WorldListWidget? = null
                for (child in screen.children()) {
                    if (child is WorldListWidget) {
                        levelList = child
                        break
                    }
                }
                if (levelList != null) {
                    for (child in levelList.children()) {
                        if (child is WorldListWidget.LoadingEntry) {
                            return
                        }
                        if (child is WorldListWidget.WorldEntry) {
                            if (!child.isLevelSelectable) {
                                continue
                            }
                            if (child.levelDisplayName == levelDisplayName) {
                                child.play()
                                return
                            } else {
                                println("Level display name: ${child.levelDisplayName}!= $levelDisplayName")
                            }
                        }
                    }
                } else {
                    println("Level list not found")
                }
            }

            is MessageScreen -> {
                for (child in screen.children()) {
                    println("Message screen child: $child")
                    if (child is NarratedMultilineTextWidget) {
                        println("Button widget: ${child.message.string}")
                    }
                }
            }

            is CreateWorldScreen -> {
            }

            else -> {
                println("Unknown screen: $screen")
            }
        }
    }

    private fun createNewWorldAndEnterUsingGUI(client: MinecraftClient) {
//        println("Creating new world")
        if (client.currentScreen == null) {
            return
        }
        when (val screen = client.currentScreen) {
            is TitleScreen -> {
                screen
                    .children()
                    .find {
                        it is ButtonWidget && it.message.string == "Singleplayer"
                    }?.let {
                        it as ButtonWidget
                        it.onPress()
                        return
                    }
            }

            is SelectWorldScreen -> {
                //                println("Select world screen1")
                var widget: WorldListWidget? = null
                var deleteButton: ButtonWidget? = null
                var createButton: ButtonWidget? = null
                for (child in screen.children()) {
                    //                    println(child)
                    if (child is WorldListWidget) {
                        widget = child
                    } else if (child is ButtonWidget) {
                        if (child.message.string == "Delete Selected World") {
                            deleteButton = child
                        } else if (child.message.string == "Create New World") {
                            createButton = child
                        }
                    }
                }
                createButton?.onPress()
            }

            is CreateWorldScreen -> {
                //                println("Create world screen")
                var createButton: ButtonWidget? = null
                val cheatRequested = true
                var indexOfWorldSettingTab = -1
                var cheatButton: CyclingButtonWidget<*>? = null
                var settingTabWidget: TabNavigationWidget? = null
                var worldTypeButton: CyclingButtonWidget<*>? = null
                for (child in screen.children()) {
                    // search for tab navigation widget, to find index of world settings tab
                    if (indexOfWorldSettingTab == -1 && child is TabNavigationWidget) {
                        settingTabWidget = child
                        for (i in child.children().indices) {
                            val tabChild: Element = child.children()[i]
                            if (tabChild is TabButtonWidget) {
                                if (tabChild.message.string == "World") {
                                    indexOfWorldSettingTab = i
                                }
                            }
                        }
                    }
                    // search for create button
                    if (createButton == null && child is ButtonWidget) {
                        if (child.message.string == "Create New World") {
                            createButton = child
                        }
                    }
                    // search for cheat button
                    if (cheatButton == null && child is CyclingButtonWidget<*>) {
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
                        //                        println(child)
                        if (child is TextFieldWidget) {
                            //                            println("Found text field")
                            child.text = initialEnvironment.seed.toString()
                        }
                    }
                }
                if (initialEnvironment.worldType == InitialEnvironment.WorldType.SUPERFLAT) {
                    for (child in screen.children()) {
                        //                        println(child)
                        if (worldTypeButton == null && child is CyclingButtonWidget<*>) {
                            if (child.message.string.startsWith("World Type")) {
                                worldTypeButton = child
                            }
                        }
                    }
                    if (worldTypeButton != null) {
                        while (!worldTypeButton.message.string.endsWith("flat")) {
                            worldTypeButton.onPress()
                        }
                    }
                }
                createButton?.onPress()
            }
        }
    }

    private fun createEmptyWorldAndEnterUsingGUI(client: MinecraftClient) {
        when (val screen = client.currentScreen) {
            is TitleScreen -> {
                screen
                    .children()
                    .find {
                        it is ButtonWidget && it.message.string == "Singleplayer"
                    }?.let {
                        it as ButtonWidget
                        it.onPress()
                        return
                    }
            }

            is SelectWorldScreen -> {
                //                println("Select world screen1")
                var widget: WorldListWidget? = null
                var deleteButton: ButtonWidget? = null
                var createButton: ButtonWidget? = null
                for (child in screen.children()) {
                    //                    println(child)
                    if (child is WorldListWidget) {
                        widget = child
                    } else if (child is ButtonWidget) {
                        if (child.message.string == "Delete Selected World") {
                            deleteButton = child
                        } else if (child.message.string == "Create New World") {
                            createButton = child
                        }
                    }
                }
                createButton?.onPress()
            }

            is CreateWorldScreen -> {
                //                println("Create world screen")
                var createButton: ButtonWidget? = null
                val cheatRequested = true
                var indexOfWorldSettingTab = -1
                var cheatButton: CyclingButtonWidget<*>? = null
                var settingTabWidget: TabNavigationWidget? = null
                var worldTypeButton: CyclingButtonWidget<*>? = null
                var customizeFlatmapButton: ButtonWidget? = null
                for (child in screen.children()) {
                    // search for tab navigation widget, to find index of world settings tab
                    if (indexOfWorldSettingTab == -1 && child is TabNavigationWidget) {
                        settingTabWidget = child
                        for (i in child.children().indices) {
                            val tabChild: Element = child.children()[i]
                            if (tabChild is TabButtonWidget) {
                                if (tabChild.message.string == "World") {
                                    indexOfWorldSettingTab = i
                                }
                            }
                        }
                    }
                    // search for create button
                    if (createButton == null && child is ButtonWidget) {
                        if (child.message.string == "Create New World") {
                            createButton = child
                        }
                    }
                    // search for cheat button
                    if (cheatButton == null && child is CyclingButtonWidget<*>) {
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
                if (initialEnvironment.seed != null) {
                    for (child in screen.children()) {
                        //                        println(child)
                        if (child is TextFieldWidget) {
                            //                            println("Found text field")
                            child.text = initialEnvironment.seed.toString()
                        }
                    }
                }
                if (initialEnvironment.worldType == InitialEnvironment.WorldType.SUPERFLAT) {
                    for (child in screen.children()) {
                        //                        println(child)
                        if (worldTypeButton == null &&
                            child is CyclingButtonWidget<*> &&
                            child.message.string.startsWith("World Type")
                        ) {
                            worldTypeButton = child
                        }
                        if (customizeFlatmapButton == null && child is ButtonWidget && child.message.string.startsWith("Customize")) {
                            customizeFlatmapButton = child
                        }
                    }
                    if (worldTypeButton != null) {
                        while (!worldTypeButton.message.string.endsWith("flat")) {
                            worldTypeButton.onPress()
                        }
                    }
                    if (customizeFlatmapButton != null) {
                        customizeFlatmapButton.onPress()
                    }
                }
                createButton?.onPress()
            }

            is CustomizeFlatLevelScreen -> {
            }
        }
    }

    private fun disableSound(client: MinecraftClient) {
        client.options?.let {
            it.getSoundVolumeOption(SoundCategory.MASTER).value = 0.0
        }
    }

    private fun disableNarrator(client: MinecraftClient) {
        val options = client.options
        if (options != null) {
            if (options.narrator.value != NarratorMode.OFF) {
                options.narrator.value = NarratorMode.OFF
                options.write()
                println("Disabled narrator")
            }
        }
    }

    private fun disableTutorial(client: MinecraftClient) {
        client.tutorialManager?.setStep(TutorialStep.NONE)
    }

    private fun disableVSync(client: MinecraftClient) {
        val options = client.options
        if (options != null) {
            if (options.enableVsync.value) {
                options.enableVsync.value = false
                client.options.write()
                println("Disabled VSync")
            }
        }
    }

    private fun setSimulationDistance(
        client: MinecraftClient,
        simulationDistance: Int,
    ) {
        val options = client.options
        if (options != null) {
            if (options.simulationDistance.value != simulationDistance) {
                options.simulationDistance.value = simulationDistance
                client.options.write()
                println("Set simulation distance to $simulationDistance")
            }
        }
    }

    private fun setRenderDistance(
        client: MinecraftClient,
        renderDistance: Int,
    ) {
        val options = client.options
        if (options != null) {
            if (options.viewDistance.value != renderDistance) {
                options.viewDistance.value = renderDistance
                client.options.write()
                println("Set render distance to $renderDistance")
            }
        }
    }

    fun reset(
        chatHud: ChatHud,
        commandExecutor: CommandExecutor,
        variableCommandAfterReset: List<String>,
    ) {
        println("Resetting...")
        hasRunInitWorld = false
        initWorldFinished = false
        chatHud.clear(true)
        onWorldTick(null, chatHud, commandExecutor, variableCommandAfterReset)
    }

    fun onWorldTick(
        minecraftServer: MinecraftServer?,
        chatHud: ChatHud,
        commandExecutor: CommandExecutor,
        variableCommandsAfterReset: List<String>,
    ) {
        player = MinecraftClient.getInstance().player ?: return

        // Get the chat messages to check if the initialization is done, and clear the chat
        val messages = ArrayList((chatHud as ChatVisibleMessageAccessor).visibleMessages)
        val hasInitFinishMessage =
            messages.find {
                val text = it.content
                val builder = StringBuilder()
                text.accept { index, style, codePoint ->
                    val ch = codePoint.toChar()
                    builder.append(ch)
                    true
                }
                val content = builder.toString()
                content.contains("Initialization Done")
            } != null
        initWorldFinished = (initWorldFinished || hasInitFinishMessage)
//        println("has init finish message: $hasInitFinishMessage, has run init world: $hasRunInitWorld, init world finished: $initWorldFinished")
        // TODO: Do not clear the chat, and delete only the message related to the initialization.
        // Do not clear the chat related to the advancements
        messages.forEach { it ->
            val text = it.content
            val builder = StringBuilder()
            text.accept { index, style, codePoint ->
                val ch = codePoint.toChar()
                builder.append(ch)
                true
            }
            val content = builder.toString()
            chatList.add(
                ChatMessageRecord(
                    it.addedTime,
                    content,
                ),
            )
        }
        chatHud.clear(true)
        if (hasRunInitWorld) {
            return
        }
        // copy the path to world file
        minecraftServer?.getSavePath(WorldSavePath.GENERATED)?.let { path ->
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
                // copy
                println("Copying structure file: $sourcePath to $targetPath")
                sourcePath.copyTo(targetPath, true)
            }
        } ?: run {
            println("World path not found; server: $minecraftServer")
        }
        // NOTE: should be called only once when initial environment is set
        val myCommandExecutor = { player: ClientPlayerEntity, c: String ->
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

    // Set the TPS to virtually unlimited
    private fun setUnlimitedTPS(commandExecutor: (ClientPlayerEntity, String) -> Unit) {
        commandExecutor(player, "/tick rate 10000")
    }

    private fun setupAllowCheats(
        cheatButton: CyclingButtonWidget<*>,
        cheatRequested: Boolean,
    ) {
        val testString = if (cheatRequested) "ON" else "OFF"
        while (!cheatButton.message.string.endsWith(testString)) {
            cheatButton.onPress()
        }
    }

    private fun setupGameMode(
        gameModeButton: CyclingButtonWidget<*>,
        gameModeRequested: GameMode,
    ) {
        val testString = gameModeRequested.name
        while (!gameModeButton.message.string.endsWith(testString)) {
            gameModeButton.onPress()
        }
    }

    private fun setupNoWeatherCycle(commandExecutor: (ClientPlayerEntity, String) -> Unit) {
        commandExecutor(
            player,
            "/gamerule doWeatherCycle false",
        )
    }

    private fun disablePauseOnLostFocus(client: MinecraftClient) {
        val options = client.options
        if (options != null) {
            if (options.pauseOnLostFocus) {
                println("Disabled pause on lost focus")
                options.pauseOnLostFocus = false
                client.options.write()
            }
        }
    }

    private fun disableOnboardAccessibility(client: MinecraftClient) {
        val options = client.options
        if (options != null) {
            if (options.onboardAccessibility) {
                println("Disabled onboardAccessibility")
                options.onboardAccessibility = false
                client.options.write()
            }
        }
    }

    private fun setHudHidden(
        client: MinecraftClient,
        hudHidden: Boolean,
    ) {
        val options = client.options
        if (options != null) {
            if (options.hudHidden != hudHidden) {
                options.hudHidden = hudHidden
                client.options.write()
                if (hudHidden) {
                    println("Hid hud")
                } else {
                    println("Showed hud")
                }
            }
        }
    }

    private fun setMaxFPSToUnlimited(client: MinecraftClient) {
        val options = client.options
        if (options != null) {
            if (options.maxFps.value < 260) { // unlimited
                options.maxFps.value = 260
                client.options.write()
                println("Set max fps to 260")
            }
        }
    }

    private fun setFovEffectDisabled(client: MinecraftClient) {
        val options = client.options
        if (options != null) {
            if (options.fovEffectScale.value != 0.0) {
                options.fovEffectScale.value = 0.0
                client.options.write()
                println("Disabled fov effect")
            }
        }
    }
}
