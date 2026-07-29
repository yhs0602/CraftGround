package com.kyhsgeekcode.minecraftenv.mixin;

import com.mojang.blaze3d.vulkan.VulkanBackend;
import com.mojang.blaze3d.vulkan.VulkanPhysicalDevice;
import com.mojang.blaze3d.vulkan.init.VulkanFeature;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import org.lwjgl.vulkan.VkDevice;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

// Prerequisite for Vulkan+Metal ZEROCOPY (importing an IOSurface-backed MTLTexture as a VkImage
// via VK_EXT_metal_objects) - see the "ZEROCOPY (Metal)" section of docs/26_2_vulkan_capture.md.
// MC's VulkanBackend.REQUIRED_DEVICE_EXTENSIONS never includes VK_EXT_metal_objects, and it must
// be enabled at device-creation time (there is no adding it after the fact), so this redirects the
// call from the public createDevice(window, ...) into the private
// createDevice(Collection<String>, VulkanPhysicalDevice, Set<VulkanFeature>) overload and extends
// the extension set there, when the physical device actually supports it.
//
// This only enables the extension - it does not implement ZEROCOPY. The MTLTexture import
// (VkImportMetalTextureInfoEXT) and per-frame vkCmdCopyImage are still unimplemented and require
// native Vulkan code.
//
// Off by default: this intercepts Mojang's Vulkan device-creation path, the riskiest point to
// touch in the whole capture stack, and hasn't been verified not to break normal Vulkan startup.
// Pass -Dcraftground.enableMetalObjects=true to opt in for testing.
@Mixin(VulkanBackend.class)
public abstract class VulkanBackendMetalExtensionMixin {
  @Shadow
  private static VkDevice createDevice(
      Collection<String> deviceExtensions,
      VulkanPhysicalDevice physicalDevice,
      Set<VulkanFeature> vulkanFeatures) {
    throw new AssertionError("mixed in");
  }

  @Redirect(
      method =
          "createDevice(JLcom/mojang/blaze3d/shaders/ShaderSource;Lcom/mojang/blaze3d/shaders/GpuDebugOptions;Ljava/lang/Runnable;)Lcom/mojang/blaze3d/systems/GpuDevice;",
      at =
          @At(
              value = "INVOKE",
              target =
                  "Lcom/mojang/blaze3d/vulkan/VulkanBackend;createDevice(Ljava/util/Collection;Lcom/mojang/blaze3d/vulkan/VulkanPhysicalDevice;Ljava/util/Set;)Lorg/lwjgl/vulkan/VkDevice;"))
  private static VkDevice minecraftEnv$injectMetalObjectsExtension(
      Collection<String> deviceExtensions,
      VulkanPhysicalDevice physicalDevice,
      Set<VulkanFeature> vulkanFeatures) {
    if (Boolean.getBoolean("craftground.enableMetalObjects")
        && physicalDevice.hasDeviceExtension("VK_EXT_metal_objects")) {
      Set<String> extended = new HashSet<>(deviceExtensions);
      extended.add("VK_EXT_metal_objects");
      System.out.println(
          "CraftGround: enabling VK_EXT_metal_objects (opt-in) for future ZEROCOPY support");
      return createDevice(extended, physicalDevice, vulkanFeatures);
    }
    return createDevice(deviceExtensions, physicalDevice, vulkanFeatures);
  }
}
