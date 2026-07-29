package com.kyhsgeekcode.minecraftenv

import com.google.protobuf.ByteString
import com.kyhsgeekcode.minecraftenv.mixin.VulkanCommandEncoderAccessor
import com.mojang.blaze3d.vulkan.VulkanDevice
import com.mojang.blaze3d.vulkan.VulkanGpuTexture
import org.lwjgl.system.MemoryStack
import org.lwjgl.vulkan.EXTExternalMemoryMetal
import org.lwjgl.vulkan.EXTMetalObjects
import org.lwjgl.vulkan.VK10
import org.lwjgl.vulkan.VK12
import org.lwjgl.vulkan.VkExternalMemoryImageCreateInfo
import org.lwjgl.vulkan.VkImageCopy
import org.lwjgl.vulkan.VkImageCreateInfo
import org.lwjgl.vulkan.VkImportMetalIOSurfaceInfoEXT
import org.lwjgl.vulkan.VkMemoryAllocateInfo
import org.lwjgl.vulkan.VkMemoryRequirements
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties

/**
 * Vulkan+Metal ZEROCOPY: imports an IOSurface directly as a `VkImage` via
 * `VK_EXT_metal_objects`/`VK_EXT_external_memory_metal` (both enabled by
 * [com.kyhsgeekcode.minecraftenv.mixin.VulkanBackendMetalExtensionMixin], opt-in behind
 * `-Dcraftground.enableMetalObjects=true`), and `vkCmdCopyImage`s MC's color target into it every
 * frame - the Vulkan equivalent of what [FramebufferCapturer]'s GL zerocopy path does with
 * `glBlitFramebuffer` into an IOSurface-backed GL texture. The same IOSurface's mach port is
 * handed to Python exactly the same way, so `observation_converter.py`'s
 * `initialize_from_mach_port` needs no changes.
 *
 * Kept separate from [Blaze3dCapture] (which owns the CPU-readback buffers/fences/RGBA->RGB
 * conversion for RAW/PNG) because this owns a wholly different resource - a foreign
 * `VkImage`/`VkDeviceMemory` plus a native IOSurface - with its own lifetime. It plugs into
 * `Blaze3dCapture`'s existing record-before-submit / `armFence()` / `awaitPendingFence()`
 * discipline (see that class' kdoc) rather than duplicating it: [recordCopy] must be called from
 * the same before-submit hook, before `Blaze3dCapture.armFence()`, and needs no CPU-side read
 * afterwards - Python reads the frame straight off the IOSurface once the fence completes.
 *
 * **Caveat**: the exact struct-chaining used in [initialize] (`VkExternalMemoryImageCreateInfo`
 * with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT`, `VkImportMetalIOSurfaceInfoEXT`
 * chained off `VkMemoryAllocateInfo`) is inferred from the `VK_EXT_metal_objects` spec, not
 * verified against a known-working sample - cross-check against MoltenVK's own use of the
 * extension if this doesn't produce a valid image on first try.
 */
object VulkanMetalZerocopy {
    init {
        System.loadLibrary("native-lib")
    }

    var ipcHandle: ByteString = ByteString.EMPTY
        private set

    private var dstImage: Long = 0L
    private var dstMemory: Long = 0L
    private var ioSurfacePtr: Long = 0L

    /**
     * Call once per resolution, guarded like [FramebufferCapturer.initializeZeroCopy]. `device`
     * must be the live Vulkan device (see `MinecraftEnv.vulkanDeviceFromRenderSystem`).
     */
    fun initialize(
        device: VulkanDevice,
        width: Int,
        height: Int,
        pythonPid: Int,
    ) {
        if (ipcHandle != ByteString.EMPTY) return

        ioSurfacePtr = createSharedIOSurfaceImpl(width, height)
        check(ioSurfacePtr != 0L) { "VulkanMetalZerocopy: failed to create IOSurface" }

        val vkDevice = device.vkDevice()
        val vkPhysicalDevice = vkDevice.physicalDevice

        MemoryStack.stackPush().use { stack ->
            val externalImageInfo =
                VkExternalMemoryImageCreateInfo.calloc(stack).apply {
                    `sType$Default`()
                    handleTypes(EXTExternalMemoryMetal.VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT)
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
                    // IOSurfaces are inherently linear (row-major host-visible) memory, unlike an
                    // opaque GPU-optimal tiling - OPTIMAL here produced a SIGSEGV inside
                    // libMoltenVK.dylib on vkCmdCopyImage (empirically confirmed on this hardware).
                    .tiling(VK10.VK_IMAGE_TILING_LINEAR)
                    .usage(VK10.VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED)
            imageCreateInfo.extent().set(width, height, 1)

            val pImage = stack.mallocLong(1)
            VulkanUtilsCheck.check(
                VK10.vkCreateImage(vkDevice, imageCreateInfo, null, pImage),
                "vkCreateImage (Metal zerocopy destination)",
            )
            dstImage = pImage[0]

            val requirements = VkMemoryRequirements.calloc(stack)
            VK10.vkGetImageMemoryRequirements(vkDevice, dstImage, requirements)

            val memProperties = VkPhysicalDeviceMemoryProperties.calloc(stack)
            VK10.vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, memProperties)
            val memoryTypeIndex = findMemoryTypeIndex(memProperties, requirements.memoryTypeBits())
            check(memoryTypeIndex >= 0) {
                "VulkanMetalZerocopy: no memory type matches requirements (bits=${requirements.memoryTypeBits()})"
            }

            val importInfo =
                VkImportMetalIOSurfaceInfoEXT.calloc(stack).apply {
                    sType(EXTMetalObjects.VK_STRUCTURE_TYPE_IMPORT_METAL_IO_SURFACE_INFO_EXT)
                    ioSurface(ioSurfacePtr)
                }
            val allocateInfo =
                VkMemoryAllocateInfo
                    .calloc(stack)
                    .`sType$Default`()
                    .pNext(importInfo.address())
                    .allocationSize(requirements.size())
                    .memoryTypeIndex(memoryTypeIndex)

            val pMemory = stack.mallocLong(1)
            VulkanUtilsCheck.check(
                VK10.vkAllocateMemory(vkDevice, allocateInfo, null, pMemory),
                "vkAllocateMemory (Metal zerocopy IOSurface import)",
            )
            dstMemory = pMemory[0]

            VulkanUtilsCheck.check(
                VK10.vkBindImageMemory(vkDevice, dstImage, dstMemory, 0L),
                "vkBindImageMemory (Metal zerocopy destination)",
            )

            transitionToTransferDst(device, stack)
        }

        val machPortHandle = createMachPortForIOSurfaceImpl(ioSurfacePtr, pythonPid)
        check(machPortHandle != null && machPortHandle != ByteString.EMPTY) {
            "VulkanMetalZerocopy: failed to create mach port"
        }
        ipcHandle = machPortHandle
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

    private fun findMemoryTypeIndex(
        memProperties: VkPhysicalDeviceMemoryProperties,
        typeBits: Int,
    ): Int {
        for (i in 0 until memProperties.memoryTypeCount()) {
            if ((typeBits and (1 shl i)) != 0) {
                return i
            }
        }
        return -1
    }

    /**
     * Records this frame's `vkCmdCopyImage` into the same command buffer MC's frame is building.
     * Must be called from the same before-submit hook as [Blaze3dCapture.recordColorReadback],
     * before [Blaze3dCapture.armFence]. No CPU-side read follows - Python reads via the mach port
     * once the fence completes, same as it does for GL zerocopy.
     */
    fun recordCopy(
        device: VulkanDevice,
        colorTexture: VulkanGpuTexture,
        width: Int,
        height: Int,
    ) {
        val cmd = (device.createCommandEncoder() as VulkanCommandEncoderAccessor).invokeCommandBuffer()

        MemoryStack.stackPush().use { stack ->
            val region =
                VkImageCopy
                    .calloc(1, stack)
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
            // Full pipeline barrier, same coverage as VulkanCommandEncoder.memoryBarrier's private
            // instance overload - makes the write visible before this submission's fence signals,
            // which is what Blaze3dCapture.armFence()/awaitPendingFence() then waits on.
            com.mojang.blaze3d.vulkan.VulkanCommandEncoder.memoryBarrier(cmd, stack)
        }
    }

    fun close() {
        if (dstImage == 0L && dstMemory == 0L && ioSurfacePtr == 0L) return
        // See MinecraftEnv.vulkanDeviceFromRenderSystem's kdoc: RenderSystem.getDevice() always
        // returns the GpuDevice wrapper, never VulkanDevice directly, so the actual backend must
        // be unwrapped via GpuDeviceBackendAccessor.
        val device =
            (
                com.mojang.blaze3d.systems.RenderSystem.getDevice() as?
                    com.kyhsgeekcode.minecraftenv.mixin.GpuDeviceBackendAccessor
            )?.backend as? VulkanDevice
        if (device != null) {
            if (dstImage != 0L) VK10.vkDestroyImage(device.vkDevice(), dstImage, null)
            if (dstMemory != 0L) VK10.vkFreeMemory(device.vkDevice(), dstMemory, null)
        }
        if (ioSurfacePtr != 0L) destroyIOSurfaceImpl(ioSurfacePtr)
        dstImage = 0L
        dstMemory = 0L
        ioSurfacePtr = 0L
        ipcHandle = ByteString.EMPTY
    }

    private external fun createSharedIOSurfaceImpl(
        width: Int,
        height: Int,
    ): Long

    private external fun createMachPortForIOSurfaceImpl(
        ioSurfacePtr: Long,
        pythonPid: Int,
    ): ByteString?

    private external fun destroyIOSurfaceImpl(ioSurfacePtr: Long)
}

// vkCreateImage/vkAllocateMemory/vkBindImageMemory return VkResult ints; Blaze3D's own Vulkan
// backend has no public "throw if not VK_SUCCESS" helper, so this is the minimal local one.
private object VulkanUtilsCheck {
    fun check(
        result: Int,
        what: String,
    ) {
        check(result == VK10.VK_SUCCESS) { "VulkanMetalZerocopy: $what failed with VkResult $result" }
    }
}
