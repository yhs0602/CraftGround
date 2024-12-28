package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.KeyboardInfo;
import com.kyhsgeekcode.minecraftenv.MouseInfo;
import org.lwjgl.glfw.*;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Overwrite;

@Mixin(net.minecraft.client.util.InputUtil.class)
public class InputUtilMixin {
  @Overwrite
  public static boolean isKeyPressed(long handle, int code) {
    return KeyboardInfo.INSTANCE.isKeyPressed(code);
  }

  @Overwrite
  public static void setMouseCallbacks(
      long handle,
      GLFWCursorPosCallbackI cursorPosCallback,
      GLFWMouseButtonCallbackI mouseButtonCallback,
      GLFWScrollCallbackI scrollCallback,
      GLFWDropCallbackI dropCallback) {
    MouseInfo.INSTANCE.setCursorPosCallback(cursorPosCallback);
    MouseInfo.INSTANCE.setMouseButtonCallback(mouseButtonCallback);
    MouseInfo.INSTANCE.setHandle(handle);
  }

  @Overwrite
  public static void setCursorParameters(long handler, int inputModeValue, double x, double y) {
    MouseInfo.INSTANCE.setCursorPos(x, y);
    MouseInfo.INSTANCE.setCursorShown(inputModeValue == GLFW.GLFW_CURSOR_NORMAL);
  }

  @Overwrite
  public static void setKeyboardCallbacks(
      long handle, GLFWKeyCallbackI keyCallback, GLFWCharModsCallbackI charModsCallback) {
    KeyboardInfo.INSTANCE.setKeyCallback(keyCallback);
    KeyboardInfo.INSTANCE.setCharModsCallback(charModsCallback);
    KeyboardInfo.INSTANCE.setHandle(handle);
  }
}
