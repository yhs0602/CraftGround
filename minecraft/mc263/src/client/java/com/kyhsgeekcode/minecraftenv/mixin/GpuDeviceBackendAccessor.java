package com.kyhsgeekcode.minecraftenv.mixin;

import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.GpuDeviceBackend;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

// RenderSystem.getDevice() always returns the concrete GpuDevice wrapper class (it composes a
// GpuDeviceBackend rather than being implemented by one - VulkanDevice's only supertype is
// Object), so `RenderSystem.getDevice() as VulkanDevice` can never succeed. This exposes the
// private backend field GpuDevice actually wraps, which - on the Vulkan backend - really is a
// VulkanDevice instance.
@Mixin(GpuDevice.class)
public interface GpuDeviceBackendAccessor {
  @Accessor("backend")
  GpuDeviceBackend getBackend();
}
