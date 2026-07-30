package com.kyhsgeekcode.minecraftenv

import com.google.protobuf.ByteString
import com.mojang.blaze3d.vulkan.VulkanDevice
import com.mojang.blaze3d.vulkan.VulkanGpuTexture

/**
 * Dispatches to whichever Vulkan cross-API ZEROCOPY backend [VulkanBackendInteropExtensionMixin]
 * actually enabled - [VulkanMetalZerocopy] on Apple (verified), [VulkanCudaZerocopy] on
 * Linux/Windows+NVIDIA (unverified, no hardware in this dev environment). The two flags are set by
 * that single mixin from mutually exclusive device-extension support (a device won't report both
 * VK_EXT_metal_objects and VK_KHR_external_memory_fd/win32 as supported), so at most one backend
 * is ever active; callers ([MinecraftEnv]) don't need to know which.
 */
object VulkanZerocopy {
    private enum class Backend { METAL, CUDA }

    private fun activeBackend(): Backend =
        when {
            VulkanMetalObjectsState.metalObjectsEnabled -> {
                Backend.METAL
            }

            VulkanCudaObjectsState.cudaInteropEnabled -> {
                Backend.CUDA
            }

            else -> {
                throw IllegalStateException(
                    "VulkanZerocopy used without an enabled interop backend - " +
                        "EnvironmentInitializer.checkRenderBackend should have already fail-fast.",
                )
            }
        }

    fun initialize(
        device: VulkanDevice,
        width: Int,
        height: Int,
        pythonPid: Int,
    ) {
        when (activeBackend()) {
            Backend.METAL -> VulkanMetalZerocopy.initialize(device, width, height, pythonPid)
            Backend.CUDA -> VulkanCudaZerocopy.initialize(device, width, height, pythonPid)
        }
    }

    fun recordCopy(
        device: VulkanDevice,
        colorTexture: VulkanGpuTexture,
        width: Int,
        height: Int,
    ) {
        when (activeBackend()) {
            Backend.METAL -> VulkanMetalZerocopy.recordCopy(device, colorTexture, width, height)
            Backend.CUDA -> VulkanCudaZerocopy.recordCopy(device, colorTexture, width, height)
        }
    }

    /**
     * No-op on Metal (Python reads the IOSurface directly once the fence completes). On CUDA,
     * must be called after [Blaze3dCapture.awaitPendingFence] - see
     * [VulkanCudaZerocopy.syncAfterFence] for why that path needs an extra device-to-device copy.
     */
    fun syncAfterFence() {
        if (VulkanCudaObjectsState.cudaInteropEnabled) VulkanCudaZerocopy.syncAfterFence()
    }

    val ipcHandle: ByteString
        get() =
            when (activeBackend()) {
                Backend.METAL -> VulkanMetalZerocopy.ipcHandle
                Backend.CUDA -> VulkanCudaZerocopy.ipcHandle
            }

    fun close() {
        VulkanMetalZerocopy.close()
        VulkanCudaZerocopy.close()
    }
}
