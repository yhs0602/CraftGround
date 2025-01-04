#include "ipc_cuda.h"
#include "dlpack.h"
#include <cassert>
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
        fprintf(stderr, "Failed to close CUDA IPC handle: %s\n",
    cudaGetErrorString(err)); } else { fprintf(stderr, "Closed CUDA IPC
    handle\n");
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
    // cuda_ipc_handle : sizeof(cudaIpcMemHandle_t)
    // after the data (sizeof(int)): device_id concatenated
    cudaIpcMemHandle_t *handle = (cudaIpcMemHandle_t *)cuda_ipc_handle;
    int device_id = *(int *)(handle + 1);

    void *device_ptr = nullptr;
    cudaSetDevice(device_id);
    err = cudaIpcOpenMemHandle(
        &device_ptr,
        *reinterpret_cast<cudaIpcMemHandle_t *>(cuda_ipc_handle),
        cudaIpcMemLazyEnablePeerAccess
    );
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
        throw std::runtime_error("Failed to allocate memory for DLManagedTensor"
        );
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
        throw std::runtime_error("Failed to allocate memory for tensor strides"
        );
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

    if (attributes.devicePointer != nullptr) {
        if (attributes.type == cudaMemoryTypeDevice) {
            printf("Memory type: Device\n");
        } else if (attributes.type == cudaMemoryTypeHost) {
            printf("Memory type: Host\n");
        } else if (attributes.type == cudaMemoryTypeManaged) {
            printf("Memory type: Managed\n");
        } else {
            printf("Memory type: Unknown\n");
        }
        if (attributes.device != device_id) {
            free(tensor->dl_tensor.shape);
            free(tensor);
            throw std::runtime_error(
                "Device ID mismatch: Attribute=" +
                std::to_string(attributes.device) +
                "!= actual=" + std::to_string(device_id)
            );
        }
        printf("\nOpen tensor from ipc handle: Device ID: %d\n", device_id);
        fflush(stdout);
    } else {
        free(tensor->dl_tensor.shape);
        free(tensor);
        cudaIpcCloseMemHandle(device_ptr);
        throw std::runtime_error("Failed to get CUDA device ID");
    }

    tensor->dl_tensor.dtype = DLDataType{kDLUInt, 8, 1};
    tensor->dl_tensor.device = DLDevice{kDLCUDA, device_id};

    tensor->deleter = deleteDLManagedTensor;
    return tensor;
}
