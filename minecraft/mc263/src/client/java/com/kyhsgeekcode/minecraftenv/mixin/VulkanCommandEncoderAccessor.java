package com.kyhsgeekcode.minecraftenv.mixin;

import com.mojang.renderpearl.backend.vulkan.VulkanCommandEncoder;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

// Exposes VulkanCommandEncoder's private commandBuffer() - the transient command buffer already
// carrying the current frame's commands, the same one copyTextureToBuffer/copyTextureToTexture
// record into. Read-only access, no behavior change: this is what lets
// VulkanMetalZerocopy.recordCopy() record its blit into that same buffer so it rides on
// Blaze3dCapture's existing armFence()/awaitPendingFence() ordering instead of needing its own
// submit.
@Mixin(VulkanCommandEncoder.class)
public interface VulkanCommandEncoderAccessor {
  @Invoker("commandBuffer")
  VkCommandBuffer invokeCommandBuffer();
}
