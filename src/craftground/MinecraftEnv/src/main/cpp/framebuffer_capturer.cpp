#include <jni.h>
#include "cross_gl.h"
#include "png_util.h"
#include "cursor.h"

#include <cstring> // For strcmp
#include <iostream>
#include <stdlib.h>

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

enum EncodingMode { RAW = 0, PNG = 1, ZEROCOPY = 2 };

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
    jclass byteStringClass = env->FindClass("com/google/protobuf/ByteString");
    if (byteStringClass == nullptr || env->ExceptionCheck()) {
        // Handle error
        return nullptr;
    }
    jmethodID copyFromMethod = env->GetStaticMethodID(
        byteStringClass, "copyFrom", "([B)Lcom/google/protobuf/ByteString;"
    );
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
    // **Note**: Flipping should be done in python side.
    glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferId);
    auto *pixels = new GLubyte[textureWidth * textureHeight * 3];
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(
        0, 0, textureWidth, textureHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels
    );

    // resize if needed
    if (textureWidth != targetSizeX || textureHeight != targetSizeY) {
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
        delete[] pixels;
        pixels = resizedPixels;
    }

    int cursorHeight = 16;
    int cursorWidth = 16;

    // Draw cursor
    if (drawCursor && xPos >= 0 && xPos < targetSizeX && yPos >= 0 &&
        yPos < targetSizeY) {
        drawCursorCPU(
            xPos,
            yPos,
            targetSizeX,
            targetSizeY,
            cursorWidth,
            cursorHeight,
            pixels
        );
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
        if (pixels != nullptr) {
            delete[] pixels;
        }
        return nullptr;
    }
    // Clean up
    env->DeleteLocalRef(byteArray);
    delete[] pixels;
    return byteStringObject;
}

#ifdef __APPLE__
#include "framebuffer_capturer_apple.h"

extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_initializeZerocopyImpl(
    JNIEnv *env,
    jclass clazz,
    jint width,
    jint height,
    jint colorAttachment,
    jint depthAttachment,
    jint python_pid
) {
    if (!initCursorTexture()) {
        fflush(stderr);
        fflush(stdout);
        return nullptr;
    }
    jclass byteStringClass = env->FindClass("com/google/protobuf/ByteString");
    if (byteStringClass == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    jmethodID copyFromMethod = env->GetStaticMethodID(
        byteStringClass, "copyFrom", "([B)Lcom/google/protobuf/ByteString;"
    );
    if (copyFromMethod == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    void *mach_port = nullptr;
    int size = initializeIoSurface(width, height, &mach_port, python_pid);
    if (size < 0 || mach_port == nullptr) {
        return nullptr;
    }

    jbyteArray byteArray = env->NewByteArray(size);
    if (byteArray == nullptr || env->ExceptionCheck()) {
        // Handle error
        free(mach_port);
        return nullptr;
    }

    env->SetByteArrayRegion(
        byteArray, 0, size, reinterpret_cast<jbyte *>(mach_port)
    );
    jobject byteStringObject =
        env->CallStaticObjectMethod(byteStringClass, copyFromMethod, byteArray);
    if (byteStringObject == nullptr || env->ExceptionCheck()) {
        // Handle error
        if (byteArray != nullptr) {
            env->DeleteLocalRef(byteArray);
        }
        free(mach_port);
        return nullptr;
    }
    env->DeleteLocalRef(byteArray);
    free(mach_port);
    return byteStringObject;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_captureFramebufferZerocopyImpl(
    JNIEnv *env,
    jclass clazz,
    jint frameBufferId,
    jint targetSizeX,
    jint targetSizeY,
    jboolean drawCursor,
    jint mouseX,
    jint mouseY
) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferId);
    if (drawCursor) {
        renderCursor(mouseX, mouseY);
    }

    // It could have been that the rendered image is already being shared,
    // but the original texture is TEXTURE_2D, so we need to convert to
    // TEXTURE_2D_RECTANGLE_ARB
    copyFramebufferToIOSurface(targetSizeX, targetSizeY);
    return nullptr;
}

#elif defined(HAS_CUDA)
#include "framebuffer_capturer_cuda.h"

extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_initializeZerocopyImpl(
    JNIEnv *env,
    jclass clazz,
    jint width,
    jint height,
    jint colorAttachment,
    jint depthAttachment,
    jint python_pid
) {
    if (!initCursorTexture()) {
        fflush(stderr);
        fflush(stdout);
        return nullptr;
    }
    jclass runtimeExceptionClass = env->FindClass("java/lang/RuntimeException");
    if (runtimeExceptionClass == nullptr) {
        fprintf(stderr, "Failed to find RuntimeException class\n");
        fflush(stderr);
        return nullptr; // JVM automatically throws NoClassDefFoundError
    }

    jclass byteStringClass = env->FindClass("com/google/protobuf/ByteString");
    if (byteStringClass == nullptr || env->ExceptionCheck()) {
        fprintf(stderr, "Failed to find ByteString class\n");
        fflush(stderr);
        return nullptr; // JVM automatically throws NoClassDefFoundError
    }
    jmethodID copyFromMethod = env->GetStaticMethodID(
        byteStringClass, "copyFrom", "([B)Lcom/google/protobuf/ByteString;"
    );
    if (copyFromMethod == nullptr || env->ExceptionCheck()) {
        fprintf(stderr, "Failed to get copyFrom method\n");
        fflush(stderr);
        return nullptr; // JVM automatically throws NoSuchMethodError
    }

    cudaIpcMemHandle_t memHandle;
    int deviceId = -1;
    int size = initialize_cuda_ipc(
        width, height, colorAttachment, depthAttachment, &memHandle, &deviceId
    );

    if (size < 0) {
        fflush(stderr);
        env->ThrowNew(
            runtimeExceptionClass,
            "Failed to initialize CUDA IPC for framebuffer capture"
        );
        return nullptr;
    }

    jbyteArray byteArray = env->NewByteArray(size + sizeof(int));
    if (byteArray == nullptr || env->ExceptionCheck()) {
        // Handle error
        fprintf(stderr, "Failed to create byte array\n");
        fflush(stderr);
        return nullptr;
    }

    env->SetByteArrayRegion(
        byteArray, 0, size, reinterpret_cast<jbyte *>(&memHandle)
    );
    env->SetByteArrayRegion(
        byteArray, size, sizeof(int), reinterpret_cast<jbyte *>(&deviceId)
    );
    jobject byteStringObject =
        env->CallStaticObjectMethod(byteStringClass, copyFromMethod, byteArray);
    if (byteStringObject == nullptr || env->ExceptionCheck()) {
        // Handle error
        fprintf(stderr, "Failed to create ByteString object\n");
        fflush(stderr);
        if (byteArray != nullptr) {
            env->DeleteLocalRef(byteArray);
        }
        return nullptr;
    }
    env->DeleteLocalRef(byteArray);
    return byteStringObject;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_captureFramebufferZerocopyImpl(
    JNIEnv *env,
    jclass clazz,
    jint frameBufferId,
    jint targetSizeX,
    jint targetSizeY,
    jboolean drawCursor,
    jint mouseX,
    jint mouseY
) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferId);

    if (drawCursor) {
        renderCursor(mouseX, mouseY);
    }

    // CUDA IPC handles are used to share the framebuffer with the Python side
    // However copy is required anyway
    copyFramebufferToCudaSharedMemory(targetSizeX, targetSizeY);
    return nullptr;
}

#else
// Returns an empty ByteString object.
// TODO: Implement this function for normal mmap IPC based one copy. (GPU ->
// CPU)
extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_initializeZerocopyImpl(
    JNIEnv *env,
    jclass clazz,
    jint width,
    jint height,
    jint colorAttachment,
    jint depthAttachment,
    jint python_pid
) {
    jclass byteStringClass = env->FindClass("com/google/protobuf/ByteString");
    if (byteStringClass == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    jfieldID emptyField = env->GetStaticFieldID(
        byteStringClass, "EMPTY", "Lcom/google/protobuf/ByteString;"
    );
    if (emptyField == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    jobject emptyByteString =
        env->GetStaticObjectField(byteStringClass, emptyField);
    return emptyByteString;
}

// TODO: Implement this function for normal mmap IPC based one copy. (GPU ->
// CPU)
extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_captureFramebufferZerocopyImpl(
    JNIEnv *env,
    jclass clazz,
    jint frameBufferId,
    jint targetSizeX,
    jint targetSizeY,
    jboolean drawCursor,
    jint mouseX,
    jint mouseY
) {
    return Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_captureFramebufferImpl(
        env,
        clazz,
        0,
        frameBufferId,
        targetSizeX,
        targetSizeY,
        targetSizeX,
        targetSizeY,
        RAW,
        false,
        drawCursor,
        mouseX,
        mouseY
    );
}

#endif