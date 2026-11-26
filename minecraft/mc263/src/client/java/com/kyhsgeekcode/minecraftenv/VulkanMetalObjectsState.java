package com.kyhsgeekcode.minecraftenv;

// Holder for whether VulkanBackendInteropExtensionMixin actually enabled VK_EXT_metal_objects +
// VK_EXT_external_memory_metal. Deliberately outside the mixin package: Sponge Mixin reserves
// com.kyhsgeekcode.minecraftenv.mixin.* for mixin classes only and forbids referencing anything in
// it directly from transformed code (IllegalClassLoadError). A mixin's own fields also get merged
// into its target (VulkanBackend here) and must be private, so this state can't live on the mixin
// itself either way - it needs a separate, ordinary class outside that package.
public final class VulkanMetalObjectsState {
  public static boolean metalObjectsEnabled = false;

  private VulkanMetalObjectsState() {}
}
