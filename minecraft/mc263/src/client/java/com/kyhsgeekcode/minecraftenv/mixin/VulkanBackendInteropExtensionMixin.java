package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.VulkanCudaObjectsState;
import com.kyhsgeekcode.minecraftenv.VulkanMetalObjectsState;
import com.mojang.renderpearl.backend.vulkan.VulkanBackend;
import com.mojang.renderpearl.backend.vulkan.VulkanPhysicalDevice;
import com.mojang.renderpearl.backend.vulkan.init.FeatureSet;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import org.lwjgl.vulkan.VkDevice;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

@Mixin(VulkanBackend.class)
public abstract class VulkanBackendInteropExtensionMixin {
  @Shadow
  private static VkDevice createDevice(FeatureSet features, VulkanPhysicalDevice physicalDevice) {
    throw new AssertionError("mixed in");
  }

  @Redirect(
      method =
          "createDevice(Lcom/mojang/renderpearl/api/device/GpuDebugOptions;)Lcom/mojang/renderpearl/api/device/GpuDevice;",
      at =
          @At(
              value = "INVOKE",
              target =
                  "Lcom/mojang/renderpearl/backend/vulkan/VulkanBackend;createDevice(Lcom/mojang/renderpearl/backend/vulkan/init/FeatureSet;Lcom/mojang/renderpearl/backend/vulkan/VulkanPhysicalDevice;)Lorg/lwjgl/vulkan/VkDevice;"))
  private VkDevice minecraftEnv$injectInteropExtensions(
      FeatureSet features, VulkanPhysicalDevice physicalDevice) {
    Set<String> extended = new HashSet<>(features.extensions());
    boolean changed = false;

    // VK_EXT_metal_objects alone only gets the import/export entry points
    // (vkExportMetalObjectsEXT, VkImportMetalIOSurfaceInfoEXT, ...) - actually importing an
    // IOSurface as a VkImage backed by an MTLTexture also needs VK_EXT_external_memory_metal for
    // the VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT handle type used in
    // VkExternalMemoryImageCreateInfo. Both are gated behind the same opt-in flag and both must be
    // supported, or neither is added (a half-enabled pair isn't useful for this feature).
    if (Boolean.getBoolean("craftground.enableMetalObjects")
        && physicalDevice.hasDeviceExtension("VK_EXT_metal_objects")
        && physicalDevice.hasDeviceExtension("VK_EXT_external_memory_metal")) {
      extended.add("VK_EXT_metal_objects");
      extended.add("VK_EXT_external_memory_metal");
      VulkanMetalObjectsState.metalObjectsEnabled = true;
      changed = true;
      System.out.println(
          "CraftGround: enabling VK_EXT_metal_objects + VK_EXT_external_memory_metal (opt-in) "
              + "for Vulkan+Metal ZEROCOPY");
    }

    if (Boolean.getBoolean("craftground.enableCudaInterop")) {
      boolean isWindows =
          System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win");
      String osHandleExtension =
          isWindows ? "VK_KHR_external_memory_win32" : "VK_KHR_external_memory_fd";
      if (physicalDevice.hasDeviceExtension(osHandleExtension)) {
        extended.add(osHandleExtension);
        VulkanCudaObjectsState.cudaInteropEnabled = true;
        changed = true;
        System.out.println(
            "CraftGround: enabling " + osHandleExtension + " (opt-in) for Vulkan+CUDA ZEROCOPY");
      }
    }

    FeatureSet requested =
        changed
            ? new FeatureSet(features.name(), extended, features.features(), features.condition())
            : features;
    return createDevice(requested, physicalDevice);
  }
}
