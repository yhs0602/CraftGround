#include "framebuffer_capturer_cuda.h"
#include "driver_types.h"

int initialize_cuda_ipc(
    int width,
    int height,
    int colorAttachment,
    int depthAttachment,
    cudaIpcMemHandle_t *memHandlePtr
) {
    cudaError_t err;
    cudaGraphicsResource *cudaResource;

    // register the texture with CUDA
    err = cudaGraphicsGLRegisterImage(
        &cudaResource,
        colorAttachment,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsReadOnly
    );

    if (err != cudaSuccess) {
       fprintf(stderr, "Failed to register GL image: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // This function provides the synchronization guarantee that any graphics
    // calls issued before cudaGraphicsMapResources() will complete before any
    // subsequent CUDA work issued in stream begins. Map the resource for access
    // by CUDA
    // glFinish();
    err = cudaGraphicsMapResources(1, &cudaResource);

    if (err != cudaSuccess) {
       fprintf(stderr, "Failed to map resources: %s\n", cudaGetErrorString(err));
        cudaGraphicsUnregisterResource(cudaResource);

        return -1;
    }

    void *devicePtr;
    size_t size;
    err = cudaGraphicsResourceGetMappedPointer(&devicePtr, &size, cudaResource);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get mapped pointer: %s\n", cudaGetErrorString(err));
        cudaGraphicsUnmapResources(1, &cudaResource);
        cudaGraphicsUnregisterResource(cudaResource);
        return -1;
    }


    // Get the ipc handle
    err = cudaIpcGetMemHandle(memHandlePtr, devicePtr);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get IPC handle: %s\n", cudaGetErrorString(err));
        cudaGraphicsUnmapResources(1, &cudaResource);
        cudaGraphicsUnregisterResource(cudaResource);
        return -1;
    }

    // cudaGraphicsUnmapResources(1, &cudaResource, 0);
    return sizeof(cudaIpcMemHandle_t);
}
