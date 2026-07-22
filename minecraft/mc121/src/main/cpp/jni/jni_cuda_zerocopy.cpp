#if defined(HAS_CUDA)
#include <jni.h>
#include "cross_gl.h"
#include "png_util.h"
#include "cursor.h"

#include <cstring> // For strcmp
#include <iostream>
#include <stdlib.h>
#include "framebuffer_capturer.h"
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

    // jclass byteStringClass =
    // env->FindClass("com/google/protobuf/ByteString");
    if (byteStringClass == nullptr || env->ExceptionCheck()) {
        fprintf(stderr, "Failed to find ByteString class\n");
        fflush(stderr);
        return nullptr; // JVM automatically throws NoClassDefFoundError
    }
    // jmethodID copyFromMethod = env->GetStaticMethodID(
    //     byteStringClass, "copyFrom", "([B)Lcom/google/protobuf/ByteString;"
    // );
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
#endif