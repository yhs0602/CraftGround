#if defined(HAS_CUDA)
#include <jni.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "framebuffer_capturer.h"

// JNI glue for VulkanCudaZerocopy.kt's native calls. Mirrors vulkan_metal_zerocopy_apple.mm's
// role for the Metal path: all the Vulkan-side work (creating/exporting the destination VkImage,
// vkCmdCopyImage) happens in Kotlin; this file only owns the CUDA-side half - importing the
// exported Vulkan memory, mapping it as a cudaMipmappedArray, and the per-frame device-to-device
// copy into the separate cudaMalloc'd buffer Python's cudaIpcMemHandle refers to (see
// VulkanCudaZerocopy.kt's kdoc for why that extra buffer/copy is needed, unlike the Metal path).
//
// NOT verified on real hardware: written and built for syntax only, no CUDA toolkit or NVIDIA GPU
// available in this development environment.

namespace {

bool g_initialized = false;
int g_deviceId = -1;
int g_width = 0;
int g_height = 0;
cudaExternalMemory_t g_extMem = nullptr;
cudaMipmappedArray_t g_mipmap = nullptr;
cudaArray_t g_level0Array = nullptr;
void *g_sharedBuffer = nullptr;

// Picks the CUDA device whose cudaDeviceProp.uuid matches the Vulkan physical device's
// VkPhysicalDeviceIDProperties.deviceUUID - both are 16-byte identifiers for the same physical
// GPU. Fails closed (-1) if nothing matches, rather than falling back to device 0: on a
// multi-GPU machine, device 0 need not be the one Vulkan is rendering on nor own the exported
// memory, and importing a mismatched device's memory can invalid-access or silently produce
// garbage rather than a clean cudaImportExternalMemory failure. Callers must treat -1 as
// initialization failure and not attempt the import.
int findMatchingCudaDevice(const jbyte *deviceUUID) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "VulkanCudaZerocopy: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            continue;
        }
        if (std::memcmp(prop.uuid.bytes, deviceUUID, 16) == 0) {
            return i;
        }
    }
    fprintf(
        stderr,
        "VulkanCudaZerocopy: no CUDA device UUID matched the Vulkan physical device; "
        "failing closed rather than guessing device 0\n"
    );
    return -1;
}

} // namespace

extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_VulkanCudaZerocopy_importVulkanMemoryAndInitCudaIpcImpl(
    JNIEnv *env,
    jobject thiz,
    jlong osHandle,
    jlong allocationSize,
    jbyteArray deviceUUID,
    jint width,
    jint height,
    jboolean isWin32
) {
    if (byteStringClass == nullptr || copyFromMethod == nullptr || env->ExceptionCheck()) {
        fprintf(stderr, "VulkanCudaZerocopy: ByteString class/method not initialized\n");
        return nullptr;
    }
    if (g_initialized) {
        fprintf(stderr, "VulkanCudaZerocopy: already initialized\n");
        return nullptr;
    }

    jbyte uuidBytes[16];
    env->GetByteArrayRegion(deviceUUID, 0, 16, uuidBytes);

    int deviceId = findMatchingCudaDevice(uuidBytes);
    if (deviceId < 0) {
        fprintf(stderr, "VulkanCudaZerocopy: no CUDA device available\n");
        return nullptr;
    }

    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "VulkanCudaZerocopy: cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    cudaExternalMemoryHandleDesc memHandleDesc{};
    memHandleDesc.size = static_cast<unsigned long long>(allocationSize);
    if (isWin32) {
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        memHandleDesc.handle.win32.handle = reinterpret_cast<void *>(osHandle);
    } else {
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memHandleDesc.handle.fd = static_cast<int>(osHandle);
    }

    cudaExternalMemory_t extMem;
    err = cudaImportExternalMemory(&extMem, &memHandleDesc);
    // Handle ownership per the CUDA runtime docs: an opaque POSIX fd is consumed by CUDA only on
    // a *successful* import (the driver closes it internally) - on failure it's still ours to
    // close. An opaque Win32 HANDLE is never taken over by CUDA (it duplicates internally), so we
    // must always close our copy, success or failure.
    if (isWin32) {
        CloseHandle(reinterpret_cast<HANDLE>(osHandle));
    } else if (err != cudaSuccess) {
        close(static_cast<int>(osHandle));
    }
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "VulkanCudaZerocopy: cudaImportExternalMemory failed: %s\n",
            cudaGetErrorString(err)
        );
        return nullptr;
    }

    cudaExternalMemoryMipmappedArrayDesc mipmapDesc{};
    mipmapDesc.offset = 0;
    mipmapDesc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    mipmapDesc.extent = make_cudaExtent(static_cast<size_t>(width), static_cast<size_t>(height), 0);
    mipmapDesc.flags = 0;
    mipmapDesc.numLevels = 1;

    cudaMipmappedArray_t mipmap;
    err = cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &mipmapDesc);
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "VulkanCudaZerocopy: cudaExternalMemoryGetMappedMipmappedArray failed: %s\n",
            cudaGetErrorString(err)
        );
        cudaDestroyExternalMemory(extMem);
        return nullptr;
    }

    cudaArray_t level0Array;
    err = cudaGetMipmappedArrayLevel(&level0Array, mipmap, 0);
    if (err != cudaSuccess) {
        fprintf(
            stderr, "VulkanCudaZerocopy: cudaGetMipmappedArrayLevel failed: %s\n", cudaGetErrorString(err)
        );
        cudaFreeMipmappedArray(mipmap);
        cudaDestroyExternalMemory(extMem);
        return nullptr;
    }

    // Separate cudaMalloc'd buffer that cudaIpcGetMemHandle can actually target - it cannot be
    // called on the memory imported above (see VulkanCudaZerocopy.kt's kdoc).
    void *sharedBuffer = nullptr;
    err = cudaMalloc(&sharedBuffer, static_cast<size_t>(width) * height * 4);
    if (err != cudaSuccess) {
        fprintf(stderr, "VulkanCudaZerocopy: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFreeMipmappedArray(mipmap);
        cudaDestroyExternalMemory(extMem);
        return nullptr;
    }

    cudaIpcMemHandle_t ipcHandle;
    err = cudaIpcGetMemHandle(&ipcHandle, sharedBuffer);
    if (err != cudaSuccess) {
        fprintf(stderr, "VulkanCudaZerocopy: cudaIpcGetMemHandle failed: %s\n", cudaGetErrorString(err));
        cudaFree(sharedBuffer);
        cudaFreeMipmappedArray(mipmap);
        cudaDestroyExternalMemory(extMem);
        return nullptr;
    }

    g_extMem = extMem;
    g_mipmap = mipmap;
    g_level0Array = level0Array;
    g_sharedBuffer = sharedBuffer;
    g_deviceId = deviceId;
    g_width = width;
    g_height = height;
    g_initialized = true;

    // Same wire format shared/native-ipc/ipc_cuda.cpp's mtl_tensor_from_cuda_ipc_handle already
    // expects (reused verbatim by observation_converter.py's mtl_tensor_from_cuda_mem_handle for
    // the existing GL+CUDA zerocopy path): sizeof(cudaIpcMemHandle_t) bytes, then the device id.
    const int handleSize = sizeof(cudaIpcMemHandle_t);
    jbyteArray byteArray = env->NewByteArray(handleSize + sizeof(int));
    if (byteArray == nullptr || env->ExceptionCheck()) {
        fprintf(stderr, "VulkanCudaZerocopy: failed to create byte array\n");
        return nullptr;
    }
    env->SetByteArrayRegion(byteArray, 0, handleSize, reinterpret_cast<jbyte *>(&ipcHandle));
    env->SetByteArrayRegion(
        byteArray, handleSize, sizeof(int), reinterpret_cast<jbyte *>(&deviceId)
    );
    jobject byteStringObject =
        env->CallStaticObjectMethod(byteStringClass, copyFromMethod, byteArray);
    env->DeleteLocalRef(byteArray);
    if (byteStringObject == nullptr || env->ExceptionCheck()) {
        fprintf(stderr, "VulkanCudaZerocopy: failed to create ByteString object\n");
        return nullptr;
    }
    return byteStringObject;
}

extern "C" JNIEXPORT void JNICALL
Java_com_kyhsgeekcode_minecraftenv_VulkanCudaZerocopy_copyImportedArrayToCudaSharedMemoryImpl(
    JNIEnv *env, jobject thiz
) {
    if (!g_initialized) {
        return;
    }
    cudaSetDevice(g_deviceId);
    // No stream argument -> the default (legacy) stream, which is synchronous with the host: this
    // call blocks until the device-to-device copy completes, so by the time
    // VulkanCudaZerocopy.syncAfterFence() (its only caller) returns, g_sharedBuffer is fully
    // written and safe for Python to read via the cudaIpcMemHandle. The caller is required to only
    // invoke this after Blaze3dCapture.awaitPendingFence() confirms the source vkCmdCopyImage
    // finished (see syncAfterFence's kdoc) - Vulkan and CUDA have no shared queue/stream to order
    // against otherwise.
    cudaError_t err = cudaMemcpy2DFromArray(
        g_sharedBuffer,
        static_cast<size_t>(g_width) * 4,
        g_level0Array,
        0,
        0,
        static_cast<size_t>(g_width) * 4,
        static_cast<size_t>(g_height),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "VulkanCudaZerocopy: cudaMemcpy2DFromArray failed: %s\n",
            cudaGetErrorString(err)
        );
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_kyhsgeekcode_minecraftenv_VulkanCudaZerocopy_destroyCudaImportImpl(
    JNIEnv *env, jobject thiz
) {
    if (!g_initialized) {
        return;
    }
    cudaSetDevice(g_deviceId);
    if (g_sharedBuffer != nullptr) {
        cudaFree(g_sharedBuffer);
        g_sharedBuffer = nullptr;
    }
    if (g_mipmap != nullptr) {
        cudaFreeMipmappedArray(g_mipmap);
        g_mipmap = nullptr;
    }
    if (g_extMem != nullptr) {
        cudaDestroyExternalMemory(g_extMem);
        g_extMem = nullptr;
    }
    g_level0Array = nullptr;
    g_deviceId = -1;
    g_width = 0;
    g_height = 0;
    g_initialized = false;
}
#endif
