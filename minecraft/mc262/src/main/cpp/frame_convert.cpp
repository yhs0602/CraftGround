#include <cstddef>
#include <jni.h>
#include <vector>

#include "cross_gl.h"
#include "cursor.h"
#include "frame_convert.h"
#include "framebuffer_capturer.h"
#include "png_util.h"

// See include/frame_convert.h and docs/26_2_vulkan_capture.md.
//
// Nothing in this file calls OpenGL. It only includes cross_gl.h because
// GLubyte is the pixel type the rest of the capture code (cursor.cpp,
// rgb_capture.cpp) already speaks - which is what lets the Vulkan/Blaze3D
// readback reuse the existing resize/cursor/PNG code verbatim instead of
// growing a second copy of it.

// **Note**: Flipping should be done in python side (matching rgb_capture.cpp).
extern "C" GLubyte *resize_pixels(
    jint &textureWidth,
    jint &textureHeight,
    jint &targetSizeX,
    jint &targetSizeY,
    GLubyte *pixels
) {
    auto *resizedPixels = new GLubyte[targetSizeX * targetSizeY * 3];
    for (int y = 0; y < targetSizeY; y++) {
        for (int x = 0; x < targetSizeX; x++) {
            int srcX = x * textureWidth / targetSizeX;
            int srcY = y * textureHeight / targetSizeY;
            int dstIndex = (y * targetSizeX + x) * 3;
            int srcIndex = (srcY * textureWidth + srcX) * 3;
            resizedPixels[dstIndex] = pixels[srcIndex];
            resizedPixels[dstIndex + 1] = pixels[srcIndex + 1];
            resizedPixels[dstIndex + 2] = pixels[srcIndex + 2];
        }
    }
    return resizedPixels;
}

jobject makeByteStringFromRgb(
    JNIEnv *env,
    GLubyte *pixels,
    jint srcWidth,
    jint srcHeight,
    jint targetSizeX,
    jint targetSizeY,
    jint encodingMode,
    jboolean drawCursor,
    jint xPos,
    jint yPos
) {
    if (byteStringClass == nullptr || copyFromMethod == nullptr ||
        env->ExceptionCheck()) {
        return nullptr;
    }

    jbyteArray byteArray = nullptr;
    if (encodingMode == RAW) {
        byteArray = env->NewByteArray(targetSizeX * targetSizeY * 3);
        if (byteArray == nullptr || env->ExceptionCheck()) {
            return nullptr;
        }
    }

    bool resized = false;
    if (srcWidth != targetSizeX || srcHeight != targetSizeY) {
        pixels = resize_pixels(
            srcWidth, srcHeight, targetSizeX, targetSizeY, pixels
        );
        resized = true;
    }

    if (drawCursor && xPos >= 0 && xPos < targetSizeX && yPos >= 0 &&
        yPos < targetSizeY) {
        drawCursorCPU(xPos, yPos, targetSizeX, targetSizeY, pixels);
    }

    if (encodingMode == PNG) {
#ifdef HAS_PNG
        std::vector<ui8> imageBytes;
        WritePngToMemory(
            (size_t)targetSizeX, (size_t)targetSizeY, pixels, imageBytes
        );
        byteArray = env->NewByteArray(imageBytes.size());
        env->SetByteArrayRegion(
            byteArray,
            0,
            imageBytes.size(),
            reinterpret_cast<jbyte *>(imageBytes.data())
        );
#else
        if (resized) {
            delete[] pixels;
        }
        env->ThrowNew(
            env->FindClass("java/lang/RuntimeException"),
            "PNG encoding is not supported on this platform: Could not find "
            "libpng"
        );
        return nullptr;
#endif
    } else if (encodingMode == RAW) {
        env->SetByteArrayRegion(
            byteArray,
            0,
            targetSizeX * targetSizeY * 3,
            reinterpret_cast<jbyte *>(pixels)
        );
    }

    jobject byteStringObject =
        env->CallStaticObjectMethod(byteStringClass, copyFromMethod, byteArray);
    if (byteArray != nullptr) {
        env->DeleteLocalRef(byteArray);
    }
    if (resized) {
        delete[] pixels;
    }
    if (byteStringObject == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    return byteStringObject;
}

// Scratch buffer for the RGBA -> RGB conversion below. Reused across frames
// like rgb_capture.cpp's, since the capture runs once per environment step.
static GLubyte *rgbScratch = nullptr;
static size_t rgbScratchSize = 0;

static GLubyte *ensureRgbScratch(size_t size) {
    if (size != rgbScratchSize) {
        delete[] rgbScratch;
        rgbScratch = new GLubyte[size];
        rgbScratchSize = size;
    }
    return rgbScratch;
}

// Backend-neutral entry point: takes the direct ByteBuffer that
// GpuBuffer.map() handed back after CommandEncoder.copyTextureToBuffer(), which
// is tightly packed RGBA8 (CommandEncoder sizes the destination as
// width * height * format.blockSize(), so there is no row padding to skip).
//
// `flipVertically` exists because the two backends disagree - or may disagree -
// on row order: the GL path reads through glReadPixels, whose first row is the
// framebuffer's bottom row, whereas vkCmdCopyImageToBuffer emits image row 0
// first. The flag lets the Kotlin side pin the convention down per backend
// without a rebuild (see docs/26_2_vulkan_capture.md, verification step 3).
extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_convertCapturedFrameImpl(
    JNIEnv *env,
    jclass clazz,
    jobject srcBuffer,
    jint srcWidth,
    jint srcHeight,
    jint targetSizeX,
    jint targetSizeY,
    jint encodingMode,
    jboolean flipVertically,
    jboolean drawCursor,
    jint xPos,
    jint yPos
) {
    auto *src =
        static_cast<const GLubyte *>(env->GetDirectBufferAddress(srcBuffer));
    if (src == nullptr) {
        env->ThrowNew(
            env->FindClass("java/lang/IllegalArgumentException"),
            "convertCapturedFrameImpl expects a direct ByteBuffer"
        );
        return nullptr;
    }
    const jlong capacity = env->GetDirectBufferCapacity(srcBuffer);
    const jlong required = (jlong)srcWidth * (jlong)srcHeight * 4;
    if (capacity < required) {
        env->ThrowNew(
            env->FindClass("java/lang/IllegalArgumentException"),
            "convertCapturedFrameImpl: buffer smaller than srcWidth*srcHeight*4"
        );
        return nullptr;
    }

    GLubyte *rgb = ensureRgbScratch((size_t)srcWidth * (size_t)srcHeight * 3);
    for (int y = 0; y < srcHeight; y++) {
        const int srcRow = flipVertically ? (srcHeight - 1 - y) : y;
        const GLubyte *srcPixel = src + (size_t)srcRow * srcWidth * 4;
        GLubyte *dstPixel = rgb + (size_t)y * srcWidth * 3;
        for (int x = 0; x < srcWidth; x++) {
            dstPixel[0] = srcPixel[0];
            dstPixel[1] = srcPixel[1];
            dstPixel[2] = srcPixel[2];
            srcPixel += 4;
            dstPixel += 3;
        }
    }

    return makeByteStringFromRgb(
        env,
        rgb,
        srcWidth,
        srcHeight,
        targetSizeX,
        targetSizeY,
        encodingMode,
        drawCursor,
        xPos,
        yPos
    );
}
