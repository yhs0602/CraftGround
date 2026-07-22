package com.kyhsgeekcode.minecraftenv.mixin;

import com.mojang.blaze3d.systems.RenderSystem;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gl.Framebuffer;
import net.minecraft.client.render.Tessellator;
import net.minecraft.client.util.Window;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

@Mixin(MinecraftClient.class)
public class RenderMixin {
  @Redirect(
      method = "render",
      at = @At(value = "INVOKE", target = "Lnet/minecraft/client/gl/Framebuffer;endWrite()V"))
  private void frameBufferEndWrite(Framebuffer instance) {
    // do nothing
  }

  @Redirect(
      method = "render",
      at = @At(value = "INVOKE", target = "Lnet/minecraft/client/gl/Framebuffer;draw(II)V"))
  private void frameBufferDraw(Framebuffer instance, int width, int height) {
    // do nothing
  }

  @Redirect(
      method = "render",
      at = @At(value = "INVOKE", target = "Lnet/minecraft/client/util/Window;swapBuffers()V"))
  private void windowSwapBuffers(Window instance) {
    RenderSystemPollEventsInvoker.pollEvents();
    RenderSystem.replayQueue();
    Tessellator.getInstance().clear();
    //        GLFW.glfwSwapBuffers(window);
    RenderSystemPollEventsInvoker.pollEvents();
  }
}
