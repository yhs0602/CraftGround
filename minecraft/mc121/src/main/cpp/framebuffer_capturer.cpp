
#include <cstddef>
#include <jni.h>
#include "cross_gl.h"
#include "png_util.h"
#include "cursor.h"
#include "depth_capture.h"
#include "rgb_capture.h"
#include "framebuffer_capturer.h"

#include <cstring> // For strcmp
#include <iostream>
#include <stdlib.h>

// JNI_Onload to cache classes and methods
jclass byteStringClass = nullptr;
jmethodID copyFromMethod = nullptr;

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *env;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) !=
        JNI_OK) {
        return JNI_ERR;
    }

    // Cache classes and methods
    auto tmpByteStringClass = env->FindClass("com/google/protobuf/ByteString");
    if (tmpByteStringClass == nullptr || env->ExceptionCheck()) {
        return JNI_ERR;
    }
    byteStringClass =
        static_cast<jclass>(env->NewGlobalRef(tmpByteStringClass));
    env->DeleteLocalRef(tmpByteStringClass);
    copyFromMethod = env->GetStaticMethodID(
        byteStringClass, "copyFrom", "([B)Lcom/google/protobuf/ByteString;"
    );
    if (copyFromMethod == nullptr || env->ExceptionCheck()) {
        return JNI_ERR;
    }

    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv *env;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) !=
        JNI_OK) {
        return;
    }

    // Clean up
    if (byteStringClass != nullptr) {
        env->DeleteGlobalRef(byteStringClass);
    }
}

// FIXME: Use glGetIntegerv(GL_NUM_EXTENSIONS) then use glGetStringi for
// OpenGL 3.0+
bool isExtensionSupported(const char *extName) {
    // Get the list of supported extensions
    const char *extensions =
        reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS));
    // FIXME: It returns nullptr after OpenGL 3.0+, even if there are extensions
    // Check for NULL pointer (just in case no OpenGL context is active)
    if (extensions == nullptr) {
        std::cerr
            << "Could not get OpenGL extensions list. Make sure an OpenGL "
               "context is active."
            << std::endl;
        return false;
    }

    // Search for the extension in the list
    const char *start = extensions;
    const char *where;
    const char *terminator;

    // Extension names should not have spaces
    while ((where = strchr(start, ' ')) || (where = strchr(start, '\0'))) {
        terminator = where;
        if ((terminator - start) == strlen(extName) &&
            strncmp(start, extName, terminator - start) == 0) {
            // Found the extension
            return true;
        }
        if (*where == '\0') {
            break; // Reached the end of the list
        }
        start = where + 1; // Move past the space
    }

    // Extension not found
    return false;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_checkExtension(
    JNIEnv *env, jclass clazz
) {
    // Check for the GL_ARB_pixel_buffer_object extension
    return (jboolean)isExtensionSupported("GL_ANGLE_pack_reverse_row_order");
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_initializeGLEW(
    JNIEnv *env, jclass clazz
) {
#ifdef __APPLE__
    return true;
#else
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW initialization failed: " << glewGetErrorString(err)
                  << std::endl;
    }
    return err == GLEW_OK;
#endif
}

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

extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_captureFramebufferImpl(
    JNIEnv *env,
    jclass clazz,
    jint textureId,
    jint frameBufferId,
    jint textureWidth,
    jint textureHeight,
    jint targetSizeX,
    jint targetSizeY,
    jint encodingMode,
    jboolean isExtensionAvailable,
    jboolean drawCursor,
    jint xPos,
    jint yPos
) {
    //    glBindTexture(GL_TEXTURE_2D, textureId);
    //    glPixelStorei(GL_PACK_ALIGNMENT, 1); // Set pixel data alignment
    //    auto* pixels = new GLubyte[textureWidth * textureHeight * 3]; // RGB
    //    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    // jclass byteStringClass =
    // env->FindClass("com/google/protobuf/ByteString");
    if (byteStringClass == nullptr || env->ExceptionCheck()) {
        // Handle error
        return nullptr;
    }
    // jmethodID copyFromMethod = env->GetStaticMethodID(
    //     byteStringClass, "copyFrom", "([B)Lcom/google/protobuf/ByteString;"
    // );
    if (copyFromMethod == nullptr || env->ExceptionCheck()) {
        // Handle error
        return nullptr;
    }
    jbyteArray byteArray = nullptr;
    if (encodingMode == RAW) {
        byteArray = env->NewByteArray(targetSizeX * targetSizeY * 3);
        if (byteArray == nullptr || env->ExceptionCheck()) {
            // Handle error
            return nullptr;
        }
    }

    GLubyte *pixels = caputreRGB(frameBufferId, textureWidth, textureHeight);
    bool resized = false;
    // resize if needed
    if (textureWidth != targetSizeX || textureHeight != targetSizeY) {
        pixels = resize_pixels(
            textureWidth, textureHeight, targetSizeX, targetSizeY, pixels
        );
        resized = true;
    }

    // Draw cursor
    if (drawCursor && xPos >= 0 && xPos < targetSizeX && yPos >= 0 &&
        yPos < targetSizeY) {
        drawCursorCPU(xPos, yPos, targetSizeX, targetSizeY, pixels);
    }

    // make png bytes from the pixels
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
        // Handle error
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
    if (byteStringObject == nullptr || env->ExceptionCheck()) {
        // Handle error
        if (byteArray != nullptr) {
            env->DeleteLocalRef(byteArray);
        }
        if (pixels != nullptr && resized) {
            delete[] pixels;
        }
        return nullptr;
    }
    // Clean up
    env->DeleteLocalRef(byteArray);
    if (pixels != nullptr && resized) {
        delete[] pixels;
    }
    return byteStringObject;
}

static bool initializedDepth = false;
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_captureDepthImpl(
    JNIEnv *env,
    jclass clazz,
    jint depthFrameBufferId,
    jint textureWidth,
    jint textureHeight,
    jboolean requiresDepthConversion,
    jfloat near,
    jfloat far
) {
    // 1. Check if GL context is initialized
    // const GLubyte *ver = glGetString(GL_VERSION);
    // if (!ver) {
    //     fprintf(stderr, "No current GL context in captureDepthImpl!\n");
    //     return nullptr;
    // }

    // // 2. Check if GL functions are initialized
    // if (!glCreateShader || !glShaderSource) {
    //     fprintf(stderr, "GL loader not initialized!\n");
    //     return nullptr;
    // }

    if (requiresDepthConversion) {
        if (!initializedDepth) {
            initDepthResources(textureWidth, textureHeight);
            initializedDepth = true;
        }
    }

    float *depthBuffer = captureDepth(
        depthFrameBufferId,
        textureWidth,
        textureHeight,
        requiresDepthConversion,
        near,
        far
    );
    jfloatArray depthArray = env->NewFloatArray(textureWidth * textureHeight);
    env->SetFloatArrayRegion(
        depthArray,
        0,
        textureWidth * textureHeight,
        reinterpret_cast<jfloat *>(depthBuffer)
    );
    // delete[] depthBuffer;
    // env->DeleteLocalRef(depthArray);
    return depthArray;
}