#ifndef __FRAMEBUFFER_CAPTURER_CUDA_H__

#define __FRAMEBUFFER_CAPTURER_CUDA_H__

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

void initialize_cuda_ipc(int width, int height, int colorAttachment, int depthAttachment, cudaIpcMemHandle_t* memHandlePtr);

#endif