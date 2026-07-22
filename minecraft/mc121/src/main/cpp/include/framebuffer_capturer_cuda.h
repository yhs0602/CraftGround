#ifndef __FRAMEBUFFER_CAPTURER_CUDA_H__

#define __FRAMEBUFFER_CAPTURER_CUDA_H__

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <driver_types.h>

int initialize_cuda_ipc(
    int width,
    int height,
    int colorAttachment,
    int depthAttachment,
    cudaIpcMemHandle_t *memHandlePtr,
    int *deviceId
);

void copyFramebufferToCudaSharedMemory(int width, int height);

#endif