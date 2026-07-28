package com.kyhsgeekcode.minecraftenv

import com.google.protobuf.ByteString
import com.mojang.blaze3d.buffers.GpuBuffer
import com.mojang.blaze3d.buffers.GpuFence
import com.mojang.blaze3d.systems.RenderSystem
import com.mojang.blaze3d.textures.GpuTexture
import java.nio.ByteOrder

/**
 * Backend-neutral frame/depth readback - the path that makes capture work on 26.2's Vulkan
 * rendering backend. See docs/26_2_vulkan_capture.md.
 *
 * It needs no native Vulkan code: Blaze3D already exposes a GPU->CPU readback that both of its
 * backends implement - `CommandEncoder.copyTextureToBuffer` into a `USAGE_MAP_READ` buffer,
 * synchronized with a `GpuFence`, then `GpuBuffer.map()`. The only native call left is the
 * RGBA->RGB conversion (`FramebufferCapturer.convertCapturedFrameImpl`), which reuses the exact
 * resize/cursor/PNG code the OpenGL path uses, so both paths produce the same wire format.
 *
 * This lives in the client source set rather than next to FramebufferCapturer because Blaze3D's
 * GpuDevice/GpuBuffer/GpuTexture types are client-only.
 *
 * **The record/read split is load-bearing.** A `GpuFence` captures the command encoder's *current*
 * submit index at creation time, and `awaitCompletion()` blocks until that index has been both
 * submitted and completed. A fence created after `CommandEncoder.submit()` therefore names a
 * submission that has not been issued yet, and waiting on it hangs. So the copy command must be
 * recorded and the fence armed *before* the frame's submit ([recordColorReadback] +
 * [recordDepthReadback] + [armFence], from RenderMixin's before-submit hook and the depth mixin),
 * and only the wait and the map happen after it ([awaitPendingFence], [readColor],
 * [readPendingDepth]).
 */
object Blaze3dCapture {
    /**
     * Which capture path to use. BLAZE3D covers Vulkan (and anything else that isn't OpenGL), but
     * is also selectable *on* OpenGL - that is what makes the two paths comparable byte-for-byte on
     * one machine, which is how [blaze3dFlipVertically] below is meant to be validated.
     */
    enum class CaptureBackend { OPENGL, BLAZE3D }

    // Set via CRAFTGROUND_CAPTURE_BACKEND=opengl|blaze3d or -Dcraftground.captureBackend=...
    private val forcedBackend: CaptureBackend? =
        (System.getenv("CRAFTGROUND_CAPTURE_BACKEND") ?: System.getProperty("craftground.captureBackend"))
            ?.trim()
            ?.uppercase()
            ?.let { name ->
                CaptureBackend.entries.find { it.name == name }
                    ?: throw IllegalArgumentException(
                        "Unknown capture backend '$name'; expected one of ${CaptureBackend.entries}",
                    )
            }

    private var detectedBackend: CaptureBackend? = null

    fun backendFor(texture: GpuTexture): CaptureBackend {
        detectedBackend?.let { return it }
        val detected =
            forcedBackend
                ?: if (texture is com.mojang.blaze3d.opengl.GlTexture) {
                    CaptureBackend.OPENGL
                } else {
                    CaptureBackend.BLAZE3D
                }
        detectedBackend = detected
        return detected
    }

    // Whether this readback yields rows in the opposite order to glReadPixels, which is what the
    // rest of the pipeline (and the Python side) is calibrated against.
    //
    // OpenGL: never. GlCommandEncoder.copyTextureToBuffer is literally glReadPixels into a PBO, so
    // it is byte-identical to the native path by construction.
    //
    // Vulkan: yes. 26.2's Vulkan backend sets a plain, non-negative-height viewport
    // (VulkanRenderPass) and compensates by declaring VK_FRONT_FACE_CLOCKWISE in every pipeline
    // (VulkanRenderPipeline) - the winding fix you only need when the Y axis is flipped relative to
    // GL. So it renders with y-down NDC, image row 0 is the top of the screen, and
    // vkCmdCopyImageToBuffer hands back rows in the reverse order to glReadPixels.
    //
    // Keyed off the actual device rather than off CaptureBackend, because this path can be forced
    // on an OpenGL device and there it must stay byte-identical to the native GL path. Override
    // with -Dcraftground.blaze3dFlipVertically=<bool> if the inference above ever stops holding.
    private val flipOverride: Boolean? =
        System.getProperty("craftground.blaze3dFlipVertically")?.toBooleanStrictOrNull()

    private fun blaze3dFlipVertically(): Boolean =
        flipOverride ?: !RenderSystem
            .getDevice()
            .deviceInfo
            .backendName()
            .contains("OpenGL", ignoreCase = true)

    private const val READBACK_USAGE = GpuBuffer.USAGE_MAP_READ or GpuBuffer.USAGE_COPY_DST
    private const val FENCE_TIMEOUT_NANOS = 5_000_000_000L

    private var colorReadbackBuffer: GpuBuffer? = null
    private var colorReadbackBytes = 0L
    private var depthReadbackBuffer: GpuBuffer? = null
    private var depthReadbackBytes = 0L

    // Armed before submit(), consumed after it.
    private var pendingFence: GpuFence? = null
    private var pendingColorWidth = 0
    private var pendingColorHeight = 0
    private var pendingDepthWidth = 0
    private var pendingDepthHeight = 0
    private var pendingDepthNear = 0.0f
    private var pendingDepthFar = 0.0f
    private var pendingDepthZZeroToOne = false
    private var hasPendingDepthReadback = false

    private fun ensureReadback(
        current: GpuBuffer?,
        currentBytes: Long,
        requiredBytes: Long,
        label: String,
    ): GpuBuffer =
        if (current != null && currentBytes == requiredBytes && !current.isClosed) {
            current
        } else {
            current?.close()
            RenderSystem.getDevice().createBuffer({ label }, READBACK_USAGE, requiredBytes)
        }

    /** Records a color copy into the readback buffer. Must run before the frame's submit(). */
    fun recordColorReadback(colorTexture: GpuTexture) {
        val width = colorTexture.getWidth(0)
        val height = colorTexture.getHeight(0)
        val required = width.toLong() * height.toLong() * colorTexture.format.blockSize()
        val buffer =
            ensureReadback(colorReadbackBuffer, colorReadbackBytes, required, "CraftGround color readback")
        colorReadbackBuffer = buffer
        colorReadbackBytes = required
        pendingColorWidth = width
        pendingColorHeight = height
        RenderSystem.getDevice().createCommandEncoder().copyTextureToBuffer(colorTexture, buffer, 0L, {}, 0)
    }

    /**
     * Records a depth copy. Called from the depth mixin, i.e. right after renderLevel() and well
     * before submit() - which is the other half of why this is a record-now/read-later split: 26.2
     * clears the depth texture between there and the end of the frame, so the copy cannot wait.
     */
    fun recordDepthReadback(
        depthTexture: GpuTexture,
        near: Float,
        far: Float,
        zZeroToOne: Boolean,
    ) {
        val width = depthTexture.getWidth(0)
        val height = depthTexture.getHeight(0)
        val required = width.toLong() * height.toLong() * depthTexture.format.blockSize()
        val buffer =
            ensureReadback(depthReadbackBuffer, depthReadbackBytes, required, "CraftGround depth readback")
        depthReadbackBuffer = buffer
        depthReadbackBytes = required
        pendingDepthWidth = width
        pendingDepthHeight = height
        pendingDepthNear = near
        pendingDepthFar = far
        pendingDepthZZeroToOne = zZeroToOne
        hasPendingDepthReadback = true
        RenderSystem.getDevice().createCommandEncoder().copyTextureToBuffer(depthTexture, buffer, 0L, {}, 0)
    }

    /** Arms the fence for the submission that is about to happen. Must run before submit(). */
    fun armFence() {
        pendingFence?.close()
        pendingFence = RenderSystem.getDevice().createCommandEncoder().createFence()
    }

    /** Waits for the armed submission to finish. Must run after submit(); a no-op if none is armed. */
    fun awaitPendingFence() {
        val fence = pendingFence ?: return
        pendingFence = null
        fence.use {
            if (!it.awaitCompletion(FENCE_TIMEOUT_NANOS)) {
                throw IllegalStateException(
                    "Timed out after ${FENCE_TIMEOUT_NANOS / 1_000_000}ms waiting for the capture readback to complete",
                )
            }
        }
    }

    /** Reads back the color copy recorded earlier. Only valid after [awaitPendingFence]. */
    fun readColor(
        targetSizeX: Int,
        targetSizeY: Int,
        encodingMode: Int,
        drawCursor: Boolean,
        xPos: Int,
        yPos: Int,
    ): ByteString {
        val buffer =
            colorReadbackBuffer
                ?: throw IllegalStateException("readColor() called without a recorded color readback")
        return buffer.map(true, false).use { view ->
            FramebufferCapturer.convertCapturedFrameImpl(
                view.data(),
                pendingColorWidth,
                pendingColorHeight,
                targetSizeX,
                targetSizeY,
                encodingMode,
                blaze3dFlipVertically(),
                drawCursor,
                xPos,
                yPos,
            )
        }
    }

    /**
     * Reads back the depth copy recorded by [recordDepthReadback] and linearizes it, matching what
     * the native GL path (src/main/cpp/depth_capture.cpp) produces. Returns null when nothing was
     * recorded, i.e. on the OpenGL path, where the depth mixin already holds the finished array.
     *
     * Only valid after [awaitPendingFence]; consumes the pending readback.
     */
    fun readPendingDepth(requiresDepthConversion: Boolean): FloatArray? {
        if (!hasPendingDepthReadback) return null
        hasPendingDepthReadback = false
        val buffer =
            depthReadbackBuffer
                ?: throw IllegalStateException("readPendingDepth() called without a recorded depth readback")
        val width = pendingDepthWidth
        val height = pendingDepthHeight
        val near = pendingDepthNear
        val far = pendingDepthFar
        val zZeroToOne = pendingDepthZZeroToOne
        val flip = blaze3dFlipVertically()
        return buffer.map(true, false).use { view ->
            val floats = view.data().order(ByteOrder.nativeOrder()).asFloatBuffer()
            val out = FloatArray(width * height)
            for (y in 0 until height) {
                val srcRow = if (flip) height - 1 - y else y
                for (x in 0 until width) {
                    val raw = floats.get(srcRow * width + x)
                    out[y * width + x] =
                        if (requiresDepthConversion) {
                            linearizeReverseZ(raw, near, far, zZeroToOne) / far
                        } else {
                            raw
                        }
                }
            }
            out
        }
    }

    /**
     * Ported verbatim from src/main/cpp/depth_capture.cpp's linearizeReverseZ. 26.2 renders the
     * level with a reversed depth range, so raw depth 1.0 is the near plane and 0.0 the far plane.
     */
    private fun linearizeReverseZ(
        d: Float,
        n: Float,
        f: Float,
        zZeroToOne: Boolean,
    ): Float =
        if (zZeroToOne) {
            (n * f) / (n + d * (f - n))
        } else {
            (2.0f * n * f) / (n + f + (2.0f * d - 1.0f) * (f - n))
        }

    /**
     * One-shot variant for the stereo path, which re-renders mid-frame and needs the pixels
     * immediately rather than at the frame's own submit(). Costs one extra submit per eye, which is
     * the price of reading a frame vanilla was never going to submit on its own.
     */
    fun captureNow(
        colorTexture: GpuTexture,
        targetSizeX: Int,
        targetSizeY: Int,
        encodingMode: Int,
        drawCursor: Boolean,
        xPos: Int,
        yPos: Int,
    ): ByteString {
        recordColorReadback(colorTexture)
        armFence()
        RenderSystem.getDevice().createCommandEncoder().submit()
        awaitPendingFence()
        return readColor(targetSizeX, targetSizeY, encodingMode, drawCursor, xPos, yPos)
    }
}
