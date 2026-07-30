package com.kyhsgeekcode.minecraftenv

import com.google.protobuf.ByteString
import com.kyhsgeekcode.minecraftenv.mixin.VulkanCommandEncoderAccessor
import com.mojang.blaze3d.vulkan.VulkanDevice
import com.mojang.blaze3d.vulkan.VulkanGpuTexture
import org.lwjgl.system.MemoryStack
import org.lwjgl.vulkan.KHRExternalMemoryFd
import org.lwjgl.vulkan.KHRExternalMemoryWin32
import org.lwjgl.vulkan.VK10
import org.lwjgl.vulkan.VK11
import org.lwjgl.vulkan.VK12
import org.lwjgl.vulkan.VkExportMemoryAllocateInfo
import org.lwjgl.vulkan.VkExternalMemoryImageCreateInfo
import org.lwjgl.vulkan.VkImageCopy
import org.lwjgl.vulkan.VkImageCreateInfo
import org.lwjgl.vulkan.VkMemoryAllocateInfo
import org.lwjgl.vulkan.VkMemoryGetFdInfoKHR
import org.lwjgl.vulkan.VkMemoryGetWin32HandleInfoKHR
import org.lwjgl.vulkan.VkMemoryRequirements
import org.lwjgl.vulkan.VkPhysicalDeviceIDProperties
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties
import org.lwjgl.vulkan.VkPhysicalDeviceProperties2
import java.util.Locale

/**
 * Vulkan+CUDA ZEROCOPY: the Linux/Windows+NVIDIA counterpart of [VulkanMetalZerocopy], reusing the
 * same `cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle` wire format the GL+CUDA path already uses (see
 * `shared-native/gl-capture/framebuffer_capturer_cuda.cpp`, `shared/native-ipc/ipc_cuda.cpp`) so
 * `observation_converter.py` needs no changes.
 *
 * Direction is the mirror image of Metal's: there, Vulkan *imports* a natively-created IOSurface.
 * Here, Vulkan *exports* its own image memory as an OS handle (a POSIX fd on Linux via
 * `VK_KHR_external_memory_fd`, a `HANDLE` on Windows via `VK_KHR_external_memory_win32` -
 * `VK_KHR_external_memory` itself, i.e. the `VkExternalMemoryImageCreateInfo`/
 * `VkExportMemoryAllocateInfo` structs, is core since Vulkan 1.1 and needs no extension string),
 * and CUDA imports *that* via `cudaImportExternalMemory` + `cudaExternalMemoryGetMappedMipmappedArray`.
 * That imported view can't be hit with `cudaIpcGetMemHandle` directly (that call only works on
 * `cudaMalloc`-backed pointers), so there is still a separate `cudaMalloc`'d buffer - exactly like
 * the GL+CUDA path's `sharedCudaColorMem` - that [syncAfterFence] `cudaMemcpy2DFromArray`s the
 * imported view into, once per frame, after the `vkCmdCopyImage` submission has completed. Python
 * opens that buffer's IPC handle exactly as it already does for GL+CUDA zerocopy.
 *
 * Picking the right CUDA device matters here in a way it doesn't for Metal: multi-GPU Linux
 * training boxes are the primary audience for this path, and CUDA's default device need not be the
 * one Vulkan is actually rendering on. [initialize] resolves this the same way NVIDIA's own
 * Vulkan/CUDA interop samples do - matching `VkPhysicalDeviceIDProperties.deviceUUID` (queried via
 * `vkGetPhysicalDeviceProperties2`) against each CUDA device's `cudaDeviceProp.uuid` - so the
 * native side never guesses device 0.
 *
 * **NOT verified on real hardware.** This was written and built for syntax on a Mac with no CUDA
 * toolkit and no Vulkan `VK_KHR_external_memory_fd`/`_win32` support to test against - unlike
 * [VulkanMetalZerocopy], none of the `VkExternalMemoryImageCreateInfo`/`VkExportMemoryAllocateInfo`
 * struct chaining, the exported-handle-to-CUDA-import path, or the per-frame
 * `cudaMemcpy2DFromArray` has been exercised end to end. Treat every assumption here (tiling,
 * struct chaining, device matching, synchronization) as a first guess to be checked against a real
 * Linux+NVIDIA+Vulkan run.
 */
object VulkanCudaZerocopy {
    init {
        System.loadLibrary("native-lib")
    }

    var ipcHandle: ByteString = ByteString.EMPTY
        private set

    private var dstImage: Long = 0L
    private var dstMemory: Long = 0L
    private var cudaImportDone = false

    private val isWindows: Boolean =
        System.getProperty("os.name", "").lowercase(Locale.ROOT).contains("win")

    /**
     * Call once per resolution, guarded like [VulkanMetalZerocopy.initialize]. `device` must be
     * the live Vulkan device (see `MinecraftEnv.vulkanDeviceFromRenderSystem`). `pythonPid` is
     * kept for call-site parity with [VulkanMetalZerocopy.initialize] / [VulkanZerocopy.initialize]
     * but unused: unlike a mach port, a `cudaIpcMemHandle_t` isn't addressed to a specific
     * process, matching the existing GL+CUDA path (`jni_cuda_zerocopy.cpp`'s `python_pid` param).
     */
    fun initialize(
        device: VulkanDevice,
        width: Int,
        height: Int,
        @Suppress("UNUSED_PARAMETER") pythonPid: Int,
    ) {
        if (ipcHandle != ByteString.EMPTY) return

        val vkDevice = device.vkDevice()
        val vkPhysicalDevice = vkDevice.physicalDevice
        val handleTypeBit =
            if (isWindows) {
                VK11.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
            } else {
                VK11.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
            }

        var osHandle = 0L
        var allocationSize = 0L
        val deviceUUID = ByteArray(16)

        MemoryStack.stackPush().use { stack ->
            val idProps = VkPhysicalDeviceIDProperties.calloc(stack).`sType$Default`()
            val props2 =
                VkPhysicalDeviceProperties2
                    .calloc(stack)
                    .`sType$Default`()
                    .pNext(idProps.address())
            VK11.vkGetPhysicalDeviceProperties2(vkPhysicalDevice, props2)
            idProps.deviceUUID().get(deviceUUID)

            val externalImageInfo =
                VkExternalMemoryImageCreateInfo.calloc(stack).apply {
                    `sType$Default`()
                    handleTypes(handleTypeBit)
                }
            val imageCreateInfo =
                VkImageCreateInfo
                    .calloc(stack)
                    .`sType$Default`()
                    .pNext(externalImageInfo.address())
                    .imageType(VK10.VK_IMAGE_TYPE_2D)
                    .format(VK10.VK_FORMAT_R8G8B8A8_UNORM)
                    .mipLevels(1)
                    .arrayLayers(1)
                    .samples(VK10.VK_SAMPLE_COUNT_1_BIT)
                    // Unlike the IOSurface-backed Metal destination, CUDA reads this through an
                    // opaque cudaMipmappedArray (cudaExternalMemoryGetMappedMipmappedArray), not a
                    // CPU-visible pointer - OPTIMAL tiling is the intended, GPU-native choice here.
                    .tiling(VK10.VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK10.VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED)
            imageCreateInfo.extent().set(width, height, 1)

            val pImage = stack.mallocLong(1)
            VulkanCudaUtilsCheck.check(
                VK10.vkCreateImage(vkDevice, imageCreateInfo, null, pImage),
                "vkCreateImage (CUDA zerocopy destination)",
            )
            dstImage = pImage[0]

            val requirements = VkMemoryRequirements.calloc(stack)
            VK10.vkGetImageMemoryRequirements(vkDevice, dstImage, requirements)
            allocationSize = requirements.size()

            val memProperties = VkPhysicalDeviceMemoryProperties.calloc(stack)
            VK10.vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, memProperties)
            val memoryTypeIndex = findMemoryTypeIndex(memProperties, requirements.memoryTypeBits())
            check(memoryTypeIndex >= 0) {
                "VulkanCudaZerocopy: no memory type matches requirements (bits=${requirements.memoryTypeBits()})"
            }

            val exportInfo =
                VkExportMemoryAllocateInfo.calloc(stack).apply {
                    `sType$Default`()
                    handleTypes(handleTypeBit)
                }
            val allocateInfo =
                VkMemoryAllocateInfo
                    .calloc(stack)
                    .`sType$Default`()
                    .pNext(exportInfo.address())
                    .allocationSize(requirements.size())
                    .memoryTypeIndex(memoryTypeIndex)

            val pMemory = stack.mallocLong(1)
            VulkanCudaUtilsCheck.check(
                VK10.vkAllocateMemory(vkDevice, allocateInfo, null, pMemory),
                "vkAllocateMemory (CUDA zerocopy export)",
            )
            dstMemory = pMemory[0]

            VulkanCudaUtilsCheck.check(
                VK10.vkBindImageMemory(vkDevice, dstImage, dstMemory, 0L),
                "vkBindImageMemory (CUDA zerocopy destination)",
            )

            transitionToTransferDst(device, stack)

            osHandle =
                if (isWindows) {
                    val getInfo =
                        VkMemoryGetWin32HandleInfoKHR.calloc(stack).apply {
                            `sType$Default`()
                            memory(dstMemory)
                            handleType(handleTypeBit)
                        }
                    val pHandle = stack.mallocPointer(1)
                    VulkanCudaUtilsCheck.check(
                        KHRExternalMemoryWin32.vkGetMemoryWin32HandleKHR(vkDevice, getInfo, pHandle),
                        "vkGetMemoryWin32HandleKHR",
                    )
                    pHandle[0]
                } else {
                    val getInfo =
                        VkMemoryGetFdInfoKHR.calloc(stack).apply {
                            `sType$Default`()
                            memory(dstMemory)
                            handleType(handleTypeBit)
                        }
                    val pFd = stack.mallocInt(1)
                    VulkanCudaUtilsCheck.check(
                        KHRExternalMemoryFd.vkGetMemoryFdKHR(vkDevice, getInfo, pFd),
                        "vkGetMemoryFdKHR",
                    )
                    pFd[0].toLong()
                }
        }

        // Handle lifetime differs by platform (see vulkan_cuda_zerocopy.cpp's import function for
        // the actual close/no-close logic): on success, a POSIX fd is consumed by CUDA; a Win32
        // HANDLE is never taken over by CUDA and is always closed there once imported (or on
        // failure). Either way, the Kotlin/Vulkan side does nothing further with osHandle after
        // this call.
        val result =
            importVulkanMemoryAndInitCudaIpcImpl(
                osHandle,
                allocationSize,
                deviceUUID,
                width,
                height,
                isWindows,
            )
        check(result != null && result != ByteString.EMPTY) {
            "VulkanCudaZerocopy: failed to import Vulkan memory into CUDA / initialize CUDA IPC"
        }
        cudaImportDone = true
        ipcHandle = result
    }

    // One-time UNDEFINED -> TRANSFER_DST_OPTIMAL transition, off the frame-critical path: a
    // transient command buffer submitted and waited on immediately, not the frame's own buffer.
    private fun transitionToTransferDst(
        device: VulkanDevice,
        stack: MemoryStack,
    ) {
        val transientCmd = device.createCommandEncoder().allocateAndBeginTransientCommandBuffer()

        val barrier =
            org.lwjgl.vulkan.VkImageMemoryBarrier
                .calloc(1, stack)
                .`sType$Default`()
                .srcAccessMask(0)
                .dstAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                .oldLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED)
                .newLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                .image(dstImage)
        barrier[0].subresourceRange().apply {
            aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
            baseMipLevel(0)
            levelCount(1)
            baseArrayLayer(0)
            layerCount(1)
        }
        VK10.vkCmdPipelineBarrier(
            transientCmd,
            VK10.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            null,
            null,
            barrier,
        )

        device.graphicsQueue().beginSubmit().use { submission ->
            submission.executeCommands(transientCmd)
        }
        device.graphicsQueue().waitIdle()
    }

    // Unlike VulkanMetalZerocopy's IOSurface-backed destination (inherently host-visible, linear
    // memory), this image is TILING_OPTIMAL and read by CUDA as a GPU-resident cudaMipmappedArray
    // - it must land in device-local memory, or it silently falls back to slower host-visible
    // memory (or fails the CUDA import outright) on GPUs that expose a matching but non-local
    // type first. Prefer a DEVICE_LOCAL type; only fall back to any matching type if none is.
    private fun findMemoryTypeIndex(
        memProperties: VkPhysicalDeviceMemoryProperties,
        typeBits: Int,
    ): Int {
        var fallback = -1
        for (i in 0 until memProperties.memoryTypeCount()) {
            if ((typeBits and (1 shl i)) == 0) continue
            if (fallback < 0) fallback = i
            val flags = memProperties.memoryTypes(i).propertyFlags()
            if ((flags and VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
                return i
            }
        }
        return fallback
    }

    /**
     * Records this frame's `vkCmdCopyImage` into the same command buffer MC's frame is building.
     * Must be called from the same before-submit hook as [Blaze3dCapture.recordColorReadback],
     * before [Blaze3dCapture.armFence] - same discipline as [VulkanMetalZerocopy.recordCopy].
     */
    fun recordCopy(
        device: VulkanDevice,
        colorTexture: VulkanGpuTexture,
        width: Int,
        height: Int,
    ) {
        val cmd = (device.createCommandEncoder() as VulkanCommandEncoderAccessor).invokeCommandBuffer()

        MemoryStack.stackPush().use { stack ->
            val region = VkImageCopy.calloc(1, stack)
            region[0].srcSubresource().set(VK10.VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1)
            region[0].dstSubresource().set(VK10.VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1)
            region[0].srcOffset().set(0, 0, 0)
            region[0].dstOffset().set(0, 0, 0)
            region[0].extent().set(width, height, 1)

            VK12.vkCmdCopyImage(
                cmd,
                colorTexture.vkImage(),
                VK10.VK_IMAGE_LAYOUT_GENERAL,
                dstImage,
                VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                region,
            )
            com.mojang.blaze3d.vulkan.VulkanCommandEncoder
                .memoryBarrier(cmd, stack)
        }
    }

    /**
     * Unlike Metal (where Python reads the IOSurface directly), the CUDA path needs one more
     * device-to-device copy: `vkCmdCopyImage` writes into the *Vulkan-owned* exported image, but
     * Python's `cudaIpcMemHandle` refers to a separate `cudaMalloc`'d buffer (see class kdoc for
     * why `cudaIpcGetMemHandle` can't target the imported view directly). This must be called
     * after [Blaze3dCapture.awaitPendingFence] has confirmed the `vkCmdCopyImage` submission
     * completed - only then is it safe for CUDA to read the memory Vulkan just wrote.
     */
    fun syncAfterFence() {
        if (!cudaImportDone) return
        copyImportedArrayToCudaSharedMemoryImpl()
    }

    fun close() {
        if (dstImage == 0L && dstMemory == 0L && !cudaImportDone) return
        val device =
            (
                com.mojang.blaze3d.systems.RenderSystem
                    .getDevice() as?
                    com.kyhsgeekcode.minecraftenv.mixin.GpuDeviceBackendAccessor
            )?.backend as? VulkanDevice
        // vkDestroyImage/vkFreeMemory on a resource with an in-flight vkCmdCopyImage referencing it
        // is undefined behavior - recordCopy's submission may still be executing (its fence is only
        // guaranteed awaited by the *next* frame's awaitPendingFence, not by the time close() runs
        // on a resolution change or shutdown). waitIdle() guarantees nothing is still using dstImage.
        device?.graphicsQueue()?.waitIdle()
        if (cudaImportDone) destroyCudaImportImpl()
        if (device != null) {
            if (dstImage != 0L) VK10.vkDestroyImage(device.vkDevice(), dstImage, null)
            if (dstMemory != 0L) VK10.vkFreeMemory(device.vkDevice(), dstMemory, null)
        }
        dstImage = 0L
        dstMemory = 0L
        cudaImportDone = false
        ipcHandle = ByteString.EMPTY
    }

    // Imports the exported Vulkan memory (osHandle: POSIX fd on Linux, HANDLE value on Windows)
    // into CUDA via cudaImportExternalMemory, maps it as a cudaMipmappedArray, selects the CUDA
    // device whose cudaDeviceProp.uuid matches deviceUUID, cudaMalloc's the separate IPC-shared
    // destination buffer, and returns its cudaIpcMemHandle + device id - same byte layout
    // shared/native-ipc/ipc_cuda.cpp's mtl_tensor_from_cuda_ipc_handle already expects
    // (sizeof(cudaIpcMemHandle_t) bytes, then a little-endian int device id).
    private external fun importVulkanMemoryAndInitCudaIpcImpl(
        osHandle: Long,
        allocationSize: Long,
        deviceUUID: ByteArray,
        width: Int,
        height: Int,
        isWin32: Boolean,
    ): ByteString?

    private external fun copyImportedArrayToCudaSharedMemoryImpl()

    private external fun destroyCudaImportImpl()
}

private object VulkanCudaUtilsCheck {
    fun check(
        result: Int,
        what: String,
    ) {
        check(result == VK10.VK_SUCCESS) { "VulkanCudaZerocopy: $what failed with VkResult $result" }
    }
}
