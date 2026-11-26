package com.kyhsgeekcode.minecraftenv

import com.kyhsgeekcode.minecraftenv.proto.ActionSpace
import com.mojang.blaze3d.platform.InputConstants
import net.minecraft.client.Minecraft
import net.minecraft.client.input.KeyEvent
import org.lwjgl.sdl.SDLKeyboard.SDL_GetKeyFromScancode

object KeyboardInfo {
    var currentState: MutableMap<Int, Boolean> = mutableMapOf()

    val keyMappings =
        mapOf(
            "W" to InputConstants.KEY_W,
            "A" to InputConstants.KEY_A,
            "S" to InputConstants.KEY_S,
            "D" to InputConstants.KEY_D,
            "LShift" to InputConstants.KEY_LSHIFT,
            "Ctrl" to InputConstants.KEY_LCONTROL,
            "Space" to InputConstants.KEY_SPACE,
            "E" to InputConstants.KEY_E,
            "Q" to InputConstants.KEY_Q,
            "Hotbar1" to InputConstants.KEY_1,
            "Hotbar2" to InputConstants.KEY_2,
            "Hotbar3" to InputConstants.KEY_3,
            "Hotbar4" to InputConstants.KEY_4,
            "Hotbar5" to InputConstants.KEY_5,
            "Hotbar6" to InputConstants.KEY_6,
            "Hotbar7" to InputConstants.KEY_7,
            "Hotbar8" to InputConstants.KEY_8,
            "Hotbar9" to InputConstants.KEY_9,
        )

    fun onAction(actionDict: ActionSpace.ActionSpaceMessageV2) {
        val actions =
            mapOf(
                "W" to actionDict.forward,
                "A" to actionDict.left,
                "S" to actionDict.back,
                "D" to actionDict.right,
                "LShift" to actionDict.sneak,
                "Ctrl" to actionDict.sprint,
                "Space" to actionDict.jump,
                "E" to actionDict.inventory,
                "Q" to actionDict.drop,
                "Hotbar1" to actionDict.hotbar1,
                "Hotbar2" to actionDict.hotbar2,
                "Hotbar3" to actionDict.hotbar3,
                "Hotbar4" to actionDict.hotbar4,
                "Hotbar5" to actionDict.hotbar5,
                "Hotbar6" to actionDict.hotbar6,
                "Hotbar7" to actionDict.hotbar7,
                "Hotbar8" to actionDict.hotbar8,
                "Hotbar9" to actionDict.hotbar9,
            )

        val client = Minecraft.getInstance()
        for ((key, scanCode) in keyMappings) {
            val held = actions.getValue(key)
            val wasHeld = currentState.put(scanCode, held) == true
            val action =
                when {
                    held && wasHeld -> InputConstants.REPEAT
                    held -> InputConstants.PRESS
                    wasHeld -> InputConstants.RELEASE
                    else -> continue
                }
            val keycode = SDL_GetKeyFromScancode(scanCode, 0, false)
            client.keyboardHandler.keyPress(client.window.handle(), action, KeyEvent(scanCode, keycode, 0))
        }
    }

    fun isKeyPressed(key: Int): Boolean = currentState[key] ?: false
}
