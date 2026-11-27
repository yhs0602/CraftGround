#import <Foundation/Foundation.h>
#include <IOSurface/IOSurface.h>
#include <jni.h>
#include <mach/mach.h>
#include <stdlib.h>
#include <cstring>

#include "framebuffer_capturer.h"
#include "framebuffer_capturer_apple.h"

extern "C" JNIEXPORT jlong JNICALL
Java_com_kyhsgeekcode_minecraftenv_VulkanMetalZerocopy_createSharedIOSurfaceImpl(
    JNIEnv *env, jobject thiz, jint width, jint height
) {
    NSDictionary *attributes = @{
        (id)kIOSurfaceWidth : @(width),
        (id)kIOSurfaceHeight : @(height),
        (id)kIOSurfaceBytesPerElement : @4,
        (id)kIOSurfacePixelFormat : @('BGRA')
    };
    IOSurfaceRef surface = IOSurfaceCreate((CFDictionaryRef)attributes);
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
