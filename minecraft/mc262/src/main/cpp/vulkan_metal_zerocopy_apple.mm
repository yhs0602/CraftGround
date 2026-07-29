#import <Foundation/Foundation.h>
#include <IOSurface/IOSurface.h>
#include <jni.h>
#include <mach/mach.h>
#include <stdlib.h>
#include <cstring>

#include "framebuffer_capturer.h"
#include "framebuffer_capturer_apple.h"

// JNI glue for VulkanMetalZerocopy.kt's native calls. Unlike the GL zerocopy path
// (jni_macos_zerocopy.cpp), no GL/GLEW is involved here: the IOSurface created below is imported
// directly into Vulkan as a VkImage via VK_EXT_metal_objects/VK_EXT_external_memory_metal, and
// vkCmdCopyImage runs entirely on the Kotlin side (VulkanMetalZerocopy.recordCopy). This file only
// owns the two Apple-native primitives that side needs: creating the shared IOSurface and
// exporting its mach port to Python, both reused verbatim from
// shared-native/gl-capture/framebuffer_capturer_apple.mm.

extern "C" JNIEXPORT jlong JNICALL
Java_com_kyhsgeekcode_minecraftenv_VulkanMetalZerocopy_createSharedIOSurfaceImpl(
    JNIEnv *env, jobject thiz, jint width, jint height
) {
    IOSurfaceRef surface = createSharedIOSurface(width, height);
    if (surface == nullptr) {
        return 0;
    }
    // +1 ref owned by the Kotlin-side lifetime of VulkanMetalZerocopy; released in
    // destroyIOSurfaceImpl (close()).
    CFRetain(surface);
    return reinterpret_cast<jlong>(surface);
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_VulkanMetalZerocopy_createMachPortForIOSurfaceImpl(
    JNIEnv *env, jobject thiz, jlong ioSurfacePtr, jint pythonPid
) {
    if (byteStringClass == nullptr || copyFromMethod == nullptr ||
        env->ExceptionCheck()) {
        return nullptr;
    }
    IOSurfaceRef surface = reinterpret_cast<IOSurfaceRef>(ioSurfacePtr);
    mach_port_t machPort = createMachPortForIOSurface(surface, pythonPid);
    if (machPort == MACH_PORT_NULL || machPort == (mach_port_t)-1) {
        return nullptr;
    }

    const int size = sizeof(machPort);
    jbyteArray byteArray = env->NewByteArray(size);
    if (byteArray == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    env->SetByteArrayRegion(
        byteArray, 0, size, reinterpret_cast<jbyte *>(&machPort)
    );
    jobject byteStringObject =
        env->CallStaticObjectMethod(byteStringClass, copyFromMethod, byteArray);
    env->DeleteLocalRef(byteArray);
    if (byteStringObject == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    return byteStringObject;
}

extern "C" JNIEXPORT void JNICALL
Java_com_kyhsgeekcode_minecraftenv_VulkanMetalZerocopy_destroyIOSurfaceImpl(
    JNIEnv *env, jobject thiz, jlong ioSurfacePtr
) {
    if (ioSurfacePtr == 0) {
        return;
    }
    CFRelease(reinterpret_cast<IOSurfaceRef>(ioSurfacePtr));
}
