package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.VulkanCudaObjectsState;
import com.kyhsgeekcode.minecraftenv.VulkanMetalObjectsState;
import com.mojang.blaze3d.vulkan.VulkanBackend;
import com.mojang.blaze3d.vulkan.VulkanPhysicalDevice;
import com.mojang.blaze3d.vulkan.init.VulkanFeature;
import java.util.Collection;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import org.lwjgl.vulkan.VkDevice;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

// Enables the device extensions the Vulkan cross-API ZEROCOPY paths need - both gated behind
// their own opt-in flag, both off by default, and both handled by the SAME @Redirect (rather than
// one mixin each) because Sponge Mixin can't cleanly stack two independent @Redirects onto the
// same call site: the redirect replaces the instruction outright, so a second one targeting it
// would either conflict or silently shadow the first depending on mixin priority. See
// docs/26_2_vulkan_capture.md's ZEROCOPY sections for both designs. MC's
// VulkanBackend.REQUIRED_DEVICE_EXTENSIONS never includes any of these, and they must be enabled
// at device-creation time (there is no adding them after the fact), so this redirects the call
// from the public createDevice(window, ...) into the private createDevice(Collection<String>,
// VulkanPhysicalDevice, Set<VulkanFeature>) overload and extends the extension set there, when the
// physical device actually supports what was asked for.
//
//  * Metal (VulkanMetalZerocopy.kt): VK_EXT_metal_objects + VK_EXT_external_memory_metal, opt-in
//    via -Dcraftground.enableMetalObjects=true. Verified end to end on Apple Silicon + MoltenVK.
//  * CUDA (VulkanCudaZerocopy.kt): VK_KHR_external_memory_fd (Linux) or
//    VK_KHR_external_memory_win32 (Windows) - VK_KHR_external_memory itself is core since Vulkan
//    1.1, only the OS-handle-specific extension needs requesting - opt-in via
//    -Dcraftground.enableCudaInterop=true. NOT verified on real hardware: this development
//    environment has neither Linux nor an NVIDIA GPU, so only the Kotlin/Java/native code paths
//    have been written and built for syntax, not run.
//
// Off by default: this intercepts Mojang's Vulkan device-creation path, the riskiest point to
// touch in the whole capture stack, and hasn't been verified not to break normal Vulkan startup.
@Mixin(VulkanBackend.class)
public abstract class VulkanBackendInteropExtensionMixin {
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
  private static VkDevice minecraftEnv$injectInteropExtensions(
      Collection<String> deviceExtensions,
      VulkanPhysicalDevice physicalDevice,
      Set<VulkanFeature> vulkanFeatures) {
    Set<String> extended = new HashSet<>(deviceExtensions);
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

    return changed
        ? createDevice(extended, physicalDevice, vulkanFeatures)
        : createDevice(deviceExtensions, physicalDevice, vulkanFeatures);
  }
}
