#include "framebuffer_capturer_cuda.h"

void initialize_cuda_ipc(
    int width,
    int height,
    int colorAttachment,
    int depthAttachment,
    cudaIpcMemHandle_t *memHandlePtr
) {
    cudaGraphicsResource *cudaResource;

    // register the texture with CUDA
    cudaGraphicsGLRegisterImage(
        &cudaResource,
        textureID,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsReadOnly
    );

    // This function provides the synchronization guarantee that any graphics
    // calls issued before cudaGraphicsMapResources() will complete before any
    // subsequent CUDA work issued in stream begins. Map the resource for access
    // by CUDA
    cudaGraphicsMapResources(1, &cudaResource);

    void *devicePtr;
    size_t size;
    cudaGraphicsResourceGetMappedPointer(&devicePtr, &size, cudaResource);

    // Get the ipc handle
    cudaIpcGetMemHandle(memHandlePtr, devicePtr);

    // cudaGraphicsUnmapResources(1, &cudaResource, 0);
}
