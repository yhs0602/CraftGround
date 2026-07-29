#include "framebuffer_capturer_cuda.h"
#include "driver_types.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <stdio.h>

static void *sharedCudaColorMem = nullptr;
static bool initialized = false;
static int rendering_gpu = -1;

int initialize_cuda_ipc(
    int width, int height, cudaIpcMemHandle_t *memHandlePtr, int *deviceId
) {
    if (initialized) {
        fprintf(stderr, "CUDA IPC already initialized\n");
        return -1;
    }
    sharedCudaColorMem = nullptr;
    cudaError_t err;

    // Get the device of the current context
    unsigned int deviceCount = 0;
    int devices[1];

    // We should select the device that is currently rendering, to avoid gpu-gpu
    // copy
    // TODO: Support SLI?
    err = cudaGLGetDevices(
        &deviceCount, devices, 1, cudaGLDeviceListCurrentFrame
    );

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get devices: %s\n", cudaGetErrorString(err));
        return -1;
    }

    rendering_gpu = devices[0];
    fprintf(
        stdout,
        "Device count: %d, current Device: %d\n",
        deviceCount,
        rendering_gpu
    );

    err = cudaSetDevice(rendering_gpu);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc(&sharedCudaColorMem, width * height * 4);
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Failed to allocate CUDA memory: %s\n",
            cudaGetErrorString(err)
        );
        return -1;
    }
    // Get the ipc handle
    err = cudaIpcGetMemHandle(memHandlePtr, sharedCudaColorMem);

    if (err != cudaSuccess) {
        fprintf(
            stderr, "Failed to get IPC handle: %s\n", cudaGetErrorString(err)
        );
        cudaFree(sharedCudaColorMem);
        return -1;
    }
    initialized = true;
    *deviceId = rendering_gpu;
    fprintf(stdout, "\n\nInitialized CUDA IPC: %p\n\n", sharedCudaColorMem);
    fflush(stdout);
    return sizeof(cudaIpcMemHandle_t);
}

void copyFramebufferToCudaSharedMemory(
    int sourceTextureId, int width, int height
) {
    // mc262's main color target is a texture handed directly from Java (see
    // docs/26_2_vulkan_capture.md - the RAW/PNG path is texture-based, not
    // FBO-based), so it's registered with CUDA directly instead of being
    // looked up from a bound read framebuffer's attachment.
    assert(sharedCudaColorMem != nullptr);
    assert(initialized);

    cudaError_t err;
    cudaGraphicsResource_t cudaResource;

    cudaSetDevice(rendering_gpu);
    // register the texture with CUDA
    err = cudaGraphicsGLRegisterImage(
        &cudaResource,
        sourceTextureId,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsReadOnly
    );

    if (err != cudaSuccess) {
        fprintf(
            stderr, "Failed to register GL image: %s\n", cudaGetErrorString(err)
        );
        assert(false);
    }
    // This function provides the synchronization guarantee that any
    // graphics
    // calls issued before cudaGraphicsMapResources() will complete before
    // any subsequent CUDA work issued in stream begins. Map the resource
    // for access by CUDA glFinish();
    err = cudaGraphicsMapResources(1, &cudaResource);
    if (err != cudaSuccess) {
        fprintf(
            stderr, "Failed to map resources: %s\n", cudaGetErrorString(err)
        );
        cudaGraphicsUnregisterResource(cudaResource);
        assert(false);
    }

    // Note: cudaGraphicsResourceGetMappedPointer cannot be used to map texture
    // memory void *devicePtr; size_t size; err =
    // cudaGraphicsResourceGetMappedPointer(&devicePtr, &size, cudaResource);
    cudaArray_t cudaArray;
    err = cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);

    if (err != cudaSuccess) {
        fprintf(
            stderr, "Failed to get mapped array: %s\n", cudaGetErrorString(err)
        );
        cudaGraphicsUnmapResources(1, &cudaResource);
        cudaGraphicsUnregisterResource(cudaResource);
        assert(false);
    }

    // Copy the texture to the shared memory
    err = cudaMemcpy2DFromArray(
        sharedCudaColorMem,
        width * 4,
        cudaArray,
        0,
        0,
        width * 4,
        height,
        cudaMemcpyDeviceToDevice
    );

    if (err != cudaSuccess) {
        fprintf(
            stderr, "Failed to copy from array: %s\n", cudaGetErrorString(err)
        );
        cudaGraphicsUnmapResources(1, &cudaResource);
        cudaGraphicsUnregisterResource(cudaResource);
        assert(false);
    }
    // err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to synchronize: %s\n", cudaGetErrorString(err));
        cudaGraphicsUnmapResources(1, &cudaResource);
        cudaGraphicsUnregisterResource(cudaResource);
        assert(false);
    }

    if (cudaResource != nullptr) {
        err = cudaGraphicsUnmapResources(1, &cudaResource);
        if (err != cudaSuccess) {
            fprintf(
                stderr,
                "Failed to unmap resources: %s\n",
                cudaGetErrorString(err)
            );
            cudaGraphicsUnregisterResource(cudaResource);
            assert(false);
        }
        cudaGraphicsUnregisterResource(cudaResource);
        cudaResource = nullptr;
    }
}