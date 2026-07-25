#if !defined(HAS_CUDA) && !defined(__APPLE__)
#include "framebuffer_capturer.h"

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
);

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
    // jclass byteStringClass =
    // env->FindClass("com/google/protobuf/ByteString");
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