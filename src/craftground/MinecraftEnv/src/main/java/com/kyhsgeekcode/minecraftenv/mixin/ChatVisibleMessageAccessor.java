package com.kyhsgeekcode.minecraftenv.mixin;

import java.util.List;
import net.minecraft.client.gui.hud.ChatHudLine;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(net.minecraft.client.gui.hud.ChatHud.class)
public interface ChatVisibleMessageAccessor {
  @Accessor("visibleMessages")
  List<ChatHudLine.Visible> getVisibleMessages();
}
