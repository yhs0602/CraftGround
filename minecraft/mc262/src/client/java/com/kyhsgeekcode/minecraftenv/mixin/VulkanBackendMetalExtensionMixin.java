package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.VulkanMetalObjectsState;
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

// Enables the device extensions Vulkan+Metal ZEROCOPY needs (VulkanMetalZerocopy.kt): importing
// an IOSurface directly as a VkImage via VK_EXT_metal_objects's VkImportMetalIOSurfaceInfoEXT,
// chained off a VkExternalMemoryImageCreateInfo whose handle type
// (VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT) comes from VK_EXT_external_memory_metal -
// see the "ZEROCOPY (Metal)" section of docs/26_2_vulkan_capture.md. MC's
// VulkanBackend.REQUIRED_DEVICE_EXTENSIONS never includes either, and they must be enabled at
// device-creation time (there is no adding them after the fact), so this redirects the call from
// the public createDevice(window, ...) into the private createDevice(Collection<String>,
// VulkanPhysicalDevice, Set<VulkanFeature>) overload and extends the extension set there, when the
// physical device actually supports both.
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
    // VK_EXT_metal_objects alone only gets the import/export entry points
    // (vkExportMetalObjectsEXT, VkImportMetalIOSurfaceInfoEXT, ...) - actually importing an
    // IOSurface as a VkImage backed by an MTLTexture also needs VK_EXT_external_memory_metal for
    // the VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT handle type used in
    // VkExternalMemoryImageCreateInfo. Both are gated behind the same opt-in flag and both must be
    // supported, or neither is added (a half-enabled pair isn't useful for this feature).
    if (Boolean.getBoolean("craftground.enableMetalObjects")
        && physicalDevice.hasDeviceExtension("VK_EXT_metal_objects")
        && physicalDevice.hasDeviceExtension("VK_EXT_external_memory_metal")) {
      Set<String> extended = new HashSet<>(deviceExtensions);
      extended.add("VK_EXT_metal_objects");
      extended.add("VK_EXT_external_memory_metal");
      VulkanMetalObjectsState.metalObjectsEnabled = true;
      System.out.println(
          "CraftGround: enabling VK_EXT_metal_objects + VK_EXT_external_memory_metal (opt-in) "
              + "for Vulkan+Metal ZEROCOPY");
      return createDevice(extended, physicalDevice, vulkanFeatures);
    }
    return createDevice(deviceExtensions, physicalDevice, vulkanFeatures);
  }
}
