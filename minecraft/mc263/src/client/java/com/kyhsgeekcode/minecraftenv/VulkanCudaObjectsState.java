package com.kyhsgeekcode.minecraftenv;

// Holder for whether VulkanBackendInteropExtensionMixin actually enabled the OS-specific external
// memory extension (VK_KHR_external_memory_fd on Linux, VK_KHR_external_memory_win32 on Windows)
// that Vulkan+CUDA ZEROCOPY needs. See VulkanMetalObjectsState for why this lives outside the
// mixin package instead of on the mixin itself.
public final class VulkanCudaObjectsState {
  public static boolean cudaInteropEnabled = false;

  private VulkanCudaObjectsState() {}
}
