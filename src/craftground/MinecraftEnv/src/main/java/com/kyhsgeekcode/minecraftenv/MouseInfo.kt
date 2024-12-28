package com.kyhsgeekcode.minecraftenv

import com.kyhsgeekcode.minecraftenv.mixin.MouseXYAccessor
import com.kyhsgeekcode.minecraftenv.proto.ActionSpace
import net.minecraft.client.MinecraftClient
import org.lwjgl.glfw.GLFW.GLFW_KEY_LEFT_SHIFT
import org.lwjgl.glfw.GLFW.GLFW_MOD_SHIFT
import org.lwjgl.glfw.GLFW.GLFW_MOUSE_BUTTON_LEFT
import org.lwjgl.glfw.GLFW.GLFW_MOUSE_BUTTON_RIGHT
import org.lwjgl.glfw.GLFW.GLFW_PRESS
import org.lwjgl.glfw.GLFW.GLFW_RELEASE
import org.lwjgl.glfw.GLFWCursorPosCallbackI
import org.lwjgl.glfw.GLFWMouseButtonCallbackI

object MouseInfo {
    var handle: Long = 0
    var cursorPosCallback: GLFWCursorPosCallbackI? = null
    var mouseButtonCallback: GLFWMouseButtonCallbackI? = null
    var mouseX: Double = 0.0
    var mouseY: Double = 0.0
    var showCursor: Boolean = false
    var currentState: MutableMap<Int, Boolean> = mutableMapOf()

    val buttonMappings =
        mapOf(
            "use" to GLFW_MOUSE_BUTTON_RIGHT,
            "attack" to GLFW_MOUSE_BUTTON_LEFT,
        )

    fun onAction(actionDict: ActionSpace.ActionSpaceMessageV2) {
        val actions =
            mapOf(
                "use" to actionDict.use,
                "attack" to actionDict.attack,
            )
        val shift = KeyboardInfo.isKeyPressed(GLFW_KEY_LEFT_SHIFT)
        val mods = if (shift) GLFW_MOD_SHIFT else 0
        // 각 마우스 버튼 상태를 비교하여 변화가 있으면 mouseCallback 호출
        for ((action, glfwButton) in buttonMappings) {
            val previousState = currentState[glfwButton] ?: false
            val currentState = actions[action] ?: false

            if (!previousState && currentState) {
                // 마우스 버튼이 처음 눌렸을 때 GLFW_PRESS 호출
                mouseButtonCallback?.invoke(handle, glfwButton, GLFW_PRESS, mods)
            } else if (previousState && !currentState) {
                // 마우스 버튼을 뗐을 때 GLFW_RELEASE 호출
                mouseButtonCallback?.invoke(handle, glfwButton, GLFW_RELEASE, mods)
            }

            // 현재 상태 갱신
            this.currentState[glfwButton] = currentState
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
        val client = MinecraftClient.getInstance()
        (client?.mouse as? MouseXYAccessor)?.setX(x)
        (client?.mouse as? MouseXYAccessor)?.setY(y)
//        println("Set mouse pos to $x, $y")
    }

    fun setCursorShown(show: Boolean) {
        showCursor = show
    }

    fun moveMouseBy(
        dx: Int,
        dy: Int,
    ) {
        // dx와 dy의 절대값 계산 (정수로 변환)
        val stepsX = Math.abs(dx)
        val stepsY = Math.abs(dy)
        // dx와 dy의 이동 방향 계산
        val stepX = if (dx > 0) 1 else -1
        val stepY = if (dy > 0) 1 else -1
        // 최대 이동 횟수 계산 (더 큰 쪽을 기준으로 반복)
        val maxSteps = Math.max(stepsX, stepsY)
        // X와 Y를 번갈아 가며 이동
        var movedX = 0
        var movedY = 0
        for (i in 0 until maxSteps) {
            if (movedX < stepsX) {
                mouseX += stepX
                cursorPosCallback?.invoke(handle, mouseX, mouseY)
                movedX++
            }
            if (movedY < stepsY) {
                mouseY += stepY
                cursorPosCallback?.invoke(handle, mouseX, mouseY)
                movedY++
            }
        }
    }
}
