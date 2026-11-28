package com.kyhsgeekcode.minecraftenv.mixin;

import java.util.List;
import net.minecraft.client.gui.components.ChatComponent;
import net.minecraft.client.multiplayer.chat.GuiMessage;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

// mc121 targeted ChatHud.visibleMessages (the trimmed, screen-sized list). 26.2's
// ChatComponent keeps a separate `allMessages` (untrimmed, full history) - closer to what
// we actually want for death-message collection, so this targets that field instead.
@Mixin(ChatComponent.class)
public interface ChatVisibleMessageAccessor {
  @Accessor("allMessages")
  List<GuiMessage> getAllMessages();
}
