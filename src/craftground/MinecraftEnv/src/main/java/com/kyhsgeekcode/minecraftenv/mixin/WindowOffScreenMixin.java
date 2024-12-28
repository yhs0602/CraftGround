package com.kyhsgeekcode.minecraftenv.mixin;

import net.minecraft.client.WindowSettings;
import net.minecraft.client.util.Window;
import net.minecraft.client.util.WindowProvider;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(WindowProvider.class)
public class WindowOffScreenMixin {
  @Inject(method = "createWindow", at = @At(value = "TAIL"))
  public void createWindow(
      WindowSettings settings, String videoMode, String title, CallbackInfoReturnable<Window> cir) {
    //        GLFW.glfwWindowHint(GLFW.GLFW_VISIBLE, GLFW.GLFW_FALSE);
    //        GLFW.glfwIconifyWindow(cir.getReturnValue().getHandle());
    //        GLFW.glfwHideWindow(cir.getReturnValue().getHandle());
  }
}
