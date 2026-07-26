package com.kyhsgeekcode.minecraftenv

import com.google.protobuf.ByteString
import com.kyhsgeekcode.minecraftenv.proto.ActionSpace
import com.kyhsgeekcode.minecraftenv.proto.InitialEnvironment
import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace
import org.lwjgl.opengl.GL11
import org.lwjgl.opengl.GL30

// mc262 counterpart of mc121's FramebufferCapturer. Most of the native capture path is shared
// with mc121 (see shared-native/gl-capture/), compiled into this project's own native-lib so it
// stays independent per Minecraft version - except RAW/PNG capture (captureFramebufferImpl),
// which is mc262-local (src/main/cpp/framebuffer_capturer.cpp + rgb_capture.cpp): 26.2 has no
// FBO integer to hand over anymore (GpuTexture/RenderTarget replaced Framebuffer), so it's
// texture-based instead, with the native side owning and caching its own capture FBO (see
// docs/26_2_phase2_plan.md W3). VULKAN is scaffolding only for now: mc262's Vulkan renderer
// readback isn't implemented yet, so it fails loudly instead of silently falling back to a
// slower path in this hot-loop call. ZEROCOPY still uses the old mc121-era frameBufferId-based
// native path and isn't wired up for 26.2 yet either (see initializeZeroCopy below).
object FramebufferCapturer {
    init {
        System.loadLibrary("native-lib")
    }

    fun captureFramebuffer(
        textureId: Int,
        frameBufferId: Int,
        textureWidth: Int,
        textureHeight: Int,
        targetSizeX: Int,
        targetSizeY: Int,
        encodingMode: Int,
        isExtensionAvailable: Boolean,
        drawCursor: Boolean,
        xPos: Int,
        yPos: Int,
    ): ByteString {
        if (encodingMode == VULKAN) {
            throw UnsupportedOperationException(
                "Vulkan frame capture is not implemented yet for mc262 (encodingMode=VULKAN)",
            )
        }
        if (encodingMode == ZEROCOPY) {
            assert(textureWidth == targetSizeX && textureHeight == targetSizeY)
            return captureFramebufferZerocopyImpl(
                frameBufferId,
                targetSizeX,
                targetSizeY,
                drawCursor,
                xPos,
                yPos,
            ) ?: ByteString.EMPTY
        } else {
            return captureFramebufferImpl(
                textureId,
                textureWidth,
                textureHeight,
                targetSizeX,
                targetSizeY,
                encodingMode,
                isExtensionAvailable,
                drawCursor,
                xPos,
                yPos,
            )
        }
    }

    // Texture-based (no frameBufferId - see the header comment above).
    external fun captureFramebufferImpl(
        textureId: Int,
        textureWidth: Int,
        textureHeight: Int,
        targetSizeX: Int,
        targetSizeY: Int,
        encodingMode: Int,
        isExtensionAvailable: Boolean,
        drawCursor: Boolean,
        xPos: Int,
        yPos: Int,
    ): ByteString

    external fun initializeGLEW(): Boolean

    fun checkGLEW(): Boolean {
        if (hasInitializedGLEW) return true
        val result = initializeGLEW()
        hasInitializedGLEW = result
        println("FramebufferCapturer: GLEW initialized: $result")
        return result
    }

    fun checkExtensionJVM() {
        if (hasCheckedExtension) return
        val vendor = GL11.glGetString(GL11.GL_VENDOR)
        if (vendor == null) {
            println("FramebufferCapturer: Vendor is null")
        } else {
            println("FramebufferCapturer: Vendor: $vendor")
        }
        val numExtensions = GL30.glGetInteger(GL30.GL_NUM_EXTENSIONS)
        for (i in 0 until numExtensions) {
            val extension = GL30.glGetStringi(GL30.GL_EXTENSIONS, i)
            println("FramebufferCapturer: Extension $i: $extension")
            if (extension == null) {
                println("FramebufferCapturer: Extension is null")
            } else if (extension.contains("GL_ANGLE_pack_reverse_row_order")) {
                println("FramebufferCapturer: Extension available")
                isExtensionAvailable = true
            }
        }
        if (!isExtensionAvailable) {
            println("FramebufferCapturer: Extension not available")
        }
        hasCheckedExtension = true
    }

    fun initializeZeroCopy(
        width: Int,
        height: Int,
        colorAttachment: Int,
        depthAttachment: Int,
        pythonPid: Int,
    ) {
        if (ipcHandle != ByteString.EMPTY) {
            return
        }
        val result = initializeZerocopyImpl(width, height, colorAttachment, depthAttachment, pythonPid)
        if (result == null || result == ByteString.EMPTY) {
            println("FramebufferCapturer: ZeroCopy initialization failed")
            throw RuntimeException("ZeroCopy initialization failed")
        }
        ipcHandle = result
    }

    external fun initializeZerocopyImpl(
        width: Int,
        height: Int,
        colorAttachment: Int,
        depthAttachment: Int,
        pythonPid: Int,
    ): ByteString?

    external fun captureFramebufferZerocopyImpl(
        frameBufferId: Int,
        targetSizeX: Int,
        targetSizeY: Int,
        drawCursor: Boolean,
        mouseX: Int,
        mouseY: Int,
    ): ByteString?

    external fun captureDepthImpl(
        depthFrameBufferId: Int,
        textureWidth: Int,
        textureHeight: Int,
        requiresDepthConversion: Boolean,
        near: Float,
        far: Float,
    ): FloatArray

    const val RAW = 0
    const val PNG = 1
    const val ZEROCOPY = 2

    // mc262-only: Vulkan readback. Not implemented yet (see captureFramebuffer above) -
    // reserved so the wire format/proto stays stable once it lands.
    const val VULKAN = 3

    var isExtensionAvailable: Boolean = false
    private var hasCheckedExtension: Boolean = false
    private var hasInitializedGLEW: Boolean = false
    var ipcHandle: ByteString = ByteString.EMPTY
        private set

    private var actionBuffer: ByteArray? = null
    var shouldCaptureDepth: Boolean = false
    var requiresDepthConversion: Boolean = false

    external fun readInitialEnvironmentImpl(
        p2jMemoryName: String,
        port: Int,
    ): ByteArray

    external fun readActionImpl(
        p2jMemoryName: String,
        actionData: ByteArray?,
    ): ByteArray

    external fun writeObservationImpl(
        p2jMemoryName: String,
        j2pMemoryName: String,
        observationData: ByteArray,
    )

    fun readInitialEnvironment(
        p2jMemoryName: String,
        port: Int,
    ): InitialEnvironment.InitialEnvironmentMessage =
        InitialEnvironment.InitialEnvironmentMessage.parseFrom(readInitialEnvironmentImpl(p2jMemoryName, port))

    fun readAction(p2jMemoryName: String): ActionSpace.ActionSpaceMessageV2 {
        actionBuffer = readActionImpl(p2jMemoryName, actionBuffer)
        return ActionSpace.ActionSpaceMessageV2.parseFrom(actionBuffer)
    }

    fun writeObservation(
        p2jMemoryName: String,
        j2pMemoryName: String,
        observationData: ObservationSpace.ObservationSpaceMessage,
    ) {
        writeObservationImpl(p2jMemoryName, j2pMemoryName, observationData.toByteArray())
    }
}
