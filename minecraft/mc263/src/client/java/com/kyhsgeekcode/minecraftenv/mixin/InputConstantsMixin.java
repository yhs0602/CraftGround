package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.KeyboardInfo;
import com.kyhsgeekcode.minecraftenv.MouseInfo;
import com.mojang.blaze3d.platform.InputConstants;
import com.mojang.blaze3d.platform.Window;
import org.lwjgl.glfw.GLFWCharCallbackI;
import org.lwjgl.glfw.GLFWCursorPosCallbackI;
import org.lwjgl.glfw.GLFWDropCallbackI;
import org.lwjgl.glfw.GLFWIMEStatusCallbackI;
import org.lwjgl.glfw.GLFWKeyCallbackI;
import org.lwjgl.glfw.GLFWMouseButtonCallbackI;
import org.lwjgl.glfw.GLFWPreeditCallbackI;
import org.lwjgl.glfw.GLFWScrollCallbackI;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Overwrite;

// Port of mc121's InputUtilMixin onto 26.2's InputConstants (renamed from InputUtil).
// Every overwritten method now takes a Window instead of a raw glfw handle, and
// setupKeyboardCallbacks gained a GLFWCharCallbackI (was GLFWCharModsCallbackI) plus
// IME/preedit callbacks mc121 never had - accepted here to match the signature exactly,
// but not wired to anything since nothing in this mod drives IME input.
@Mixin(InputConstants.class)
public class InputConstantsMixin {
  @Overwrite
  public static boolean isKeyDown(Window window, int key) {
    return KeyboardInfo.INSTANCE.isKeyPressed(key);
  }

  @Overwrite
  public static void setupKeyboardCallbacks(
      Window window,
      GLFWKeyCallbackI keyPressCallback,
      GLFWCharCallbackI charTypedCallback,
      GLFWPreeditCallbackI preeditCallback,
      GLFWIMEStatusCallbackI imeStatusCallback) {
    KeyboardInfo.INSTANCE.setKeyCallback(keyPressCallback);
    KeyboardInfo.INSTANCE.setCharModsCallback(charTypedCallback);
    KeyboardInfo.INSTANCE.setHandle(window.handle());
  }

  @Overwrite
  public static void setupMouseCallbacks(
      Window window,
      GLFWCursorPosCallbackI onMoveCallback,
      GLFWMouseButtonCallbackI onPressCallback,
      GLFWScrollCallbackI onScrollCallback,
      GLFWDropCallbackI onDropCallback) {
    MouseInfo.INSTANCE.setCursorPosCallback(onMoveCallback);
    MouseInfo.INSTANCE.setMouseButtonCallback(onPressCallback);
    MouseInfo.INSTANCE.setHandle(window.handle());
  }

  @Overwrite
  public static void grabOrReleaseMouse(Window window, int cursorMode, double xpos, double ypos) {
    MouseInfo.INSTANCE.setCursorPos(xpos, ypos);
    MouseInfo.INSTANCE.setCursorShown(cursorMode == InputConstants.CURSOR_NORMAL);
  }
}
