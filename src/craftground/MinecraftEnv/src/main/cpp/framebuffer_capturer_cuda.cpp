#include "framebuffer_capturer_cuda.h"
#include "driver_types.h"
#include <cassert>
#include <stdio.h>

static void *sharedCudaColorMem = nullptr;
static cudaGraphicsResource *cudaResource = nullptr;
static bool initialized = false;
static int oldColorAttachment = -1;

// TODO: depth attachment
int initialize_cuda_ipc(
    int width,
    int height,
    int colorAttachment,
    int depthAttachment,
    cudaIpcMemHandle_t *memHandlePtr
) {
    if (initialized) {
        fprintf(stderr, "CUDA IPC already initialized\n");
        return -1;
    }
    sharedCudaColorMem = nullptr;
    cudaError_t err;

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
    return sizeof(cudaIpcMemHandle_t);
}

void copyFramebufferToCudaSharedMemory(int width, int height) {
    GLuint renderedTextureId;
    glGetFramebufferAttachmentParameteriv(
        GL_READ_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
        (GLint *)&renderedTextureId
    );
    glBindTexture(GL_TEXTURE_2D, renderedTextureId);
    int textureWidth, textureHeight;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &textureWidth);
    glGetTexLevelParameteriv(
        GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &textureHeight
    );
    // printf("width: %d, height: %d\n", textureWidth, textureHeight);
    glViewport(0, 0, width, height);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    GLenum status = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        printf("Framebuffer is not complete! Status: 0x%x\n", status);
        fflush(stdout);
        assert(status == GL_FRAMEBUFFER_COMPLETE);
    }
    assert(glGetError() == GL_NO_ERROR);
    assert(width == textureWidth);
    assert(height == textureHeight);

    assert(sharedCudaColorMem != nullptr);
    assert(initialized);

    cudaError_t err;

    if (oldColorAttachment != renderedTextureId && cudaResource != nullptr) {
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
        }
        // register the texture with CUDA
        err = cudaGraphicsGLRegisterImage(
            &cudaResource,
            renderedTextureId,
            GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsReadOnly
        );

        if (err != cudaSuccess) {
            fprintf(
                stderr,
                "Failed to register GL image: %s\n",
                cudaGetErrorString(err)
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
        oldColorAttachment = renderedTextureId;
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
}