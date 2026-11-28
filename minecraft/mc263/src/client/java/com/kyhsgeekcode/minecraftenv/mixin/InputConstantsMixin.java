package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.KeyboardInfo;
import com.kyhsgeekcode.minecraftenv.MouseInfo;
import com.mojang.blaze3d.platform.InputConstants;
import com.mojang.blaze3d.platform.Window;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Overwrite;

@Mixin(InputConstants.class)
public class InputConstantsMixin {
  @Overwrite
  public static boolean isKeyDown(int key) {
    return KeyboardInfo.INSTANCE.isKeyPressed(key);
  }

  @Overwrite
  public static void grabMouse(Window window, double x, double y) {
    MouseInfo.INSTANCE.setCursorPos(x, y);
    MouseInfo.INSTANCE.setCursorShown(false);
  }

  @Overwrite
  public static void releaseMouse(Window window, double x, double y) {
    MouseInfo.INSTANCE.setCursorPos(x, y);
    MouseInfo.INSTANCE.setCursorShown(true);
  }
}
