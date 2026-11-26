package com.kyhsgeekcode.minecraftenv.mixin;

import com.mojang.blaze3d.platform.SDLEventHandler;
import org.lwjgl.sdl.SDL_Event;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Only the agent supplies keyboard and mouse input. Keep SDL window events flowing. */
@Mixin(SDLEventHandler.class)
public class SDLEventHandlerMixin {
  @Inject(
      method = {
        "handleKeyEvent", "handleTextInputEvent", "handleTextEditingEvent",
        "handleMouseMotionEvent", "handleMouseButtonEvent", "handleMouseWheelEvent",
        "handleDropFileEvent"
      },
      at = @At("HEAD"),
      cancellable = true)
  private void ignorePhysicalInput(SDL_Event event, CallbackInfo ci) {
    ci.cancel();
  }
}
