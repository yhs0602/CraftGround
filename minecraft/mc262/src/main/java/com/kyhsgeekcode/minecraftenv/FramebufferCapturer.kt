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
// docs/26_2_phase2_plan.md W3).
//
// There are now two capture paths, picked by which rendering backend 26.2 actually came up with
// (see Blaze3dCapture.CaptureBackend / docs/26_2_vulkan_capture.md):
//
//  * OPENGL  - the original one, and everything in this file. GlTexture.glId() -> native
//              glReadPixels. Untouched, so the GL backend keeps exactly its previous performance.
//  * BLAZE3D - backend-neutral, and the one that makes Vulkan work. Lives in Blaze3dCapture (in
//              the client source set, because Blaze3D's GpuDevice/GpuBuffer types are client-only).
//              It needs no native Vulkan code at all; the only part of it that crosses into JNI is
//              the RGBA->RGB conversion, convertCapturedFrameImpl below, which reuses the same
//              resize/cursor/PNG code as the GL path.
//
// ZEROCOPY_TORCH still uses the old mc121-era frameBufferId-based native path and isn't wired up
// for 26.2 on either backend yet (see initializeZeroCopy below).
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
        if (encodingMode == ZEROCOPY_TORCH) {
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

    // Texture-based like captureFramebufferImpl, and reverse-Z aware: 26.2 renders the level
    // with a reversed depth range, so raw depth 1.0 is the near plane and 0.0 the far plane.
    // `zZeroToOne` is RenderSystem.getDevice().deviceInfo.isZZeroToOne (true when
    // GL_ARB_clip_control is available, which selects a [0,1] rather than [-1,1] clip range);
    // the two cases need different linearization. See src/main/cpp/depth_capture.cpp.
    external fun captureDepthImpl(
        depthTextureId: Int,
        textureWidth: Int,
        textureHeight: Int,
        requiresDepthConversion: Boolean,
        near: Float,
        far: Float,
        zZeroToOne: Boolean,
    ): FloatArray

    // The one native entry point of the backend-neutral capture path (Blaze3dCapture, client source
    // set). It stays declared here because the JNI symbol is bound to this class' name and because
    // the C++ side of it shares the resize/cursor/PNG code with captureFramebufferImpl above.
    //
    // Takes the direct ByteBuffer from GpuBuffer.map(): tightly packed RGBA8 (CommandEncoder sizes
    // the destination as width * height * format.blockSize(), so there is no row padding).
    external fun convertCapturedFrameImpl(
        src: java.nio.ByteBuffer,
        srcWidth: Int,
        srcHeight: Int,
        targetSizeX: Int,
        targetSizeY: Int,
        encodingMode: Int,
        flipVertically: Boolean,
        drawCursor: Boolean,
        xPos: Int,
        yPos: Int,
    ): ByteString

    // Encoding modes are the wire format (see src/craftground/screen_encoding_modes.py) and are
    // orthogonal to CaptureBackend: a Vulkan run still produces RAW or PNG bytes in exactly the
    // same layout. There is deliberately no VULKAN encoding mode - an earlier scaffolding constant
    // used ordinal 3, which collides with Python's ZEROCOPY_JAX.
    const val RAW = 0
    const val PNG = 1
    const val ZEROCOPY_TORCH = 2

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
