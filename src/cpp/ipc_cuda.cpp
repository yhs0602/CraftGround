#include "ipc_cuda.h"
#include "dlpack.h"
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <stdio.h>

static void deleteDLManagedTensor(DLManagedTensor *self) {
    return;
    /*
    Allocate once, deallocate once. (On exit)
    cudaError_t err = cudaIpcCloseMemHandle(self->dl_tensor.data);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to close CUDA IPC handle: %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "Closed CUDA IPC handle\n");
    }
    fflush(stderr);
    free(self->dl_tensor.shape);
    free(self->dl_tensor.strides);
    free(self);
    */
}

DLManagedTensor *
mtl_tensor_from_cuda_ipc_handle(void *cuda_ipc_handle, int width, int height) {
    cudaError_t err;
    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "Failed to get CUDA device count: " +
            std::string(cudaGetErrorString(err))
        );
    }

    void *device_ptr = nullptr;
    int success_device_id = 0;
    for (; success_device_id < deviceCount; ++success_device_id) {
        cudaSetDevice(success_device_id);
        err= cudaIpcOpenMemHandle(
            &device_ptr,
            *reinterpret_cast<cudaIpcMemHandle_t *>(cuda_ipc_handle),
            cudaIpcMemLazyEnablePeerAccess
        );
        if (err == cudaSuccess) {
            break;
        }
    }
    if (err != cudaSuccess || device_ptr == nullptr) {
        throw std::runtime_error(
            "Failed to open CUDA IPC handle: " +
            std::string(cudaGetErrorString(err))
        );
    }
    printf("Opened CUDA IPC handle: %p\n", device_ptr);

    DLManagedTensor *tensor =
        (DLManagedTensor *)malloc(sizeof(DLManagedTensor));
    
    if (!tensor) {
       throw std::runtime_error("Failed to allocate memory for DLManagedTensor");
    }
    tensor->dl_tensor.data = device_ptr;
    tensor->dl_tensor.ndim = 3; // H x W x C
    tensor->dl_tensor.shape = (int64_t *)malloc(3 * sizeof(int64_t));
    if (!tensor->dl_tensor.shape) {
        free(tensor);
        throw std::runtime_error("Failed to allocate memory for tensor shape");
    }
    tensor->dl_tensor.shape[0] = height;
    tensor->dl_tensor.shape[1] = width;
    tensor->dl_tensor.shape[2] = 4; // RGBA
    tensor->dl_tensor.strides = (int64_t *)malloc(3 * sizeof(int64_t));
    if (!tensor->dl_tensor.strides) {
        free(tensor->dl_tensor.shape);
        free(tensor);
        throw std::runtime_error("Failed to allocate memory for tensor strides");
    }
    tensor->dl_tensor.strides[0] = width * 4;
    tensor->dl_tensor.strides[1] = 4;
    tensor->dl_tensor.strides[2] = 1;
    tensor->dl_tensor.byte_offset = 0;

    cudaPointerAttributes attributes;
    err = cudaPointerGetAttributes(&attributes, device_ptr);

    if (err != cudaSuccess) {
        free(tensor->dl_tensor.shape);
        free(tensor);
        throw std::runtime_error(
            "Failed to get CUDA pointer attributes: " +
            std::string(cudaGetErrorString(err))
        );
    }

    int device_id;
    if (attributes.devicePointer != nullptr) {
        device_id = attributes.device;
        printf("\nOpen tensor from ipc handle: Device ID: %d\n", device_id);
        fflush(stdout);
    } else {
        free(tensor->dl_tensor.shape);
        free(tensor);
        cudaIpcCloseMemHandle(device_ptr);
        throw std::runtime_error("Failed to get CUDA device ID");
    }

    tensor->dl_tensor.dtype =
        (DLDataType){kDLUInt, 8, 1}; // Unsigned 8-bit integer
    tensor->dl_tensor.device = (DLDevice){kDLCUDA, device_id}; // cuda gpu  + 1

    tensor->deleter = deleteDLManagedTensor;
    return tensor;
}
