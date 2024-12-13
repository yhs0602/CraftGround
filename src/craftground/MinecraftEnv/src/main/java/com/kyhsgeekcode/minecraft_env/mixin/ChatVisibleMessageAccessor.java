package com.kyhsgeekcode.minecraft_env.mixin;

import net.minecraft.client.gui.hud.ChatHudLine;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

import java.util.List;

@Mixin(net.minecraft.client.gui.hud.ChatHud.class)
public interface ChatVisibleMessageAccessor {
    @Accessor("visibleMessages")
    List<ChatHudLine.Visible> getVisibleMessages();
}
