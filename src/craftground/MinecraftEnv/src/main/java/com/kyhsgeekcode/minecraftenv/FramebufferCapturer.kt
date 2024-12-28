package com.kyhsgeekcode.minecraftenv

import com.google.protobuf.ByteString
import org.lwjgl.opengl.GL11
import org.lwjgl.opengl.GL30

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
        if (encodingMode == ZEROCOPY) {
            captureFramebufferZerocopyImpl(
                frameBufferId,
                targetSizeX,
                targetSizeY,
                drawCursor,
                xPos,
                yPos,
            )
            return ByteString.EMPTY
        } else {
            return captureFramebufferImpl(
                textureId,
                frameBufferId,
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

    external fun captureFramebufferImpl(
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
    ): ByteString

    external fun initializeGLEW(): Boolean

    fun checkGLEW(): Boolean {
        if (hasInitializedGLEW) return true
        val result = initializeGLEW()
        hasInitializedGLEW = result
        println("FramebufferCapturer: GLEW initialized: $result")
        return result
    }

    //    private external fun checkExtension(): Boolean
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
    ) {
        val result = initializeZerocopyImpl(width, height, colorAttachment, depthAttachment)
        if (result == null) {
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
    ): ByteString?

    external fun captureFramebufferZerocopyImpl(
        frameBufferId: Int,
        targetSizeX: Int,
        targetSizeY: Int,
        drawCursor: Boolean,
        mouseX: Int,
        mouseY: Int,
    )

    const val RAW = 0
    const val PNG = 1
    const val ZEROCOPY = 2

    var isExtensionAvailable: Boolean = false
    private var hasCheckedExtension: Boolean = false
    private var hasInitializedGLEW: Boolean = false
    lateinit var ipcHandle: ByteString
        private set
}
