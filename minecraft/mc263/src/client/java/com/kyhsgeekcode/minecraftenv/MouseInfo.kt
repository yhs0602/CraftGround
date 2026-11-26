package com.kyhsgeekcode.minecraftenv

import com.kyhsgeekcode.minecraftenv.mixin.MouseXYAccessor
import com.kyhsgeekcode.minecraftenv.proto.ActionSpace
import com.mojang.blaze3d.platform.InputConstants
import net.minecraft.client.Minecraft
import net.minecraft.client.input.MouseButtonInfo

object MouseInfo {
    var mouseX: Double = 0.0
    var mouseY: Double = 0.0
    var showCursor: Boolean = false
    var currentState: MutableMap<Int, Boolean> = mutableMapOf()

    val buttonMappings =
        mapOf(
            "use" to InputConstants.MOUSE_BUTTON_RIGHT,
            "attack" to InputConstants.MOUSE_BUTTON_LEFT,
        )

    fun onAction(actionDict: ActionSpace.ActionSpaceMessageV2) {
        val actions =
            mapOf(
                "use" to actionDict.use,
                "attack" to actionDict.attack,
            )
        val shift = KeyboardInfo.isKeyPressed(InputConstants.KEY_LSHIFT)
        val mods = if (shift) InputConstants.MOD_SHIFT else 0
        val client = Minecraft.getInstance()
        for ((action, button) in buttonMappings) {
            val held = actions.getValue(action)
            val wasHeld = currentState.put(button, held) == true
            if (held != wasHeld) {
                val state = if (held) InputConstants.PRESS else InputConstants.RELEASE
                client.mouseHandler.onButton(client.window.handle(), MouseButtonInfo(button, mods), state)
            }
        }
    }

    fun getMousePos(): Pair<Double, Double> = Pair(mouseX, mouseY)

    fun setCursorPos(
        x: Double,
        y: Double,
    ) {
        mouseX = x
        mouseY = y
        // Do not call the callback
        val client = Minecraft.getInstance()
        (client?.mouseHandler as? MouseXYAccessor)?.setX(x)
        (client?.mouseHandler as? MouseXYAccessor)?.setY(y)
    }

    fun setCursorShown(show: Boolean) {
        showCursor = show
    }

    fun moveMouseBy(
        dx: Int,
        dy: Int,
    ) {
        val client = Minecraft.getInstance()
        // SDL ignores its first motion event after grabbing the mouse. Establish the
        // current position before submitting the whole relative movement for this step.
        client.mouseHandler.onMove(client.window.handle(), mouseX, mouseY, 0.0, 0.0)
        mouseX += dx
        mouseY += dy
        client.mouseHandler.onMove(client.window.handle(), mouseX, mouseY, dx.toDouble(), dy.toDouble())
    }
}
