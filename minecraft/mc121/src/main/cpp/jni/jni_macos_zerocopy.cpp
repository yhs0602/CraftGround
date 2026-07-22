#ifdef __APPLE__
#include <jni.h>
#include "cross_gl.h"
#include "cursor.h"
#include <stdlib.h>
#include "framebuffer_capturer.h"
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
    // jclass byteStringClass =
    // env->FindClass("com/google/protobuf/ByteString");
    if (byteStringClass == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    // jmethodID copyFromMethod = env->GetStaticMethodID(
    //     byteStringClass, "copyFrom", "([B)Lcom/google/protobuf/ByteString;"
    // );
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
#endif