#include "ipc_cuda.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
struct ImportedFrame {
    DLManagedTensor tensor{};
    int64_t shape[3];
    int64_t strides[3];
};

void checkCuda(cudaError_t err, const char *operation) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(err)
        );
    }
}

void deleteDLManagedTensor(DLManagedTensor *tensor) {
    auto *frame = static_cast<ImportedFrame *>(tensor->manager_ctx);
    const int device = tensor->dl_tensor.device.device_id;
    int previous_device = -1;
    cudaGetDevice(&previous_device);
    cudaError_t err = cudaSetDevice(device);
    if (err == cudaSuccess) {
        // A consumer may still be copying the final frame on another stream.
        cudaDeviceSynchronize();
        err = cudaIpcCloseMemHandle(tensor->dl_tensor.data);
    }
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Failed to close CUDA IPC handle: %s\n",
            cudaGetErrorString(err)
        );
    }
    if (previous_device >= 0 && previous_device != device) {
        cudaSetDevice(previous_device);
    }
    delete frame;
}
} // namespace

DLManagedTensor *mtl_tensor_from_cuda_ipc_handle(
    const std::string &cuda_ipc_handle, int width, int height
) {
    cudaIpcMemHandle_t handle;
    int device;
    if (cuda_ipc_handle.size() != sizeof(handle) + sizeof(device)) {
        throw std::invalid_argument("Invalid CUDA IPC handle size");
    }
    memcpy(&handle, cuda_ipc_handle.data(), sizeof(handle));
    memcpy(&device, cuda_ipc_handle.data() + sizeof(handle), sizeof(device));

    auto frame = std::make_unique<ImportedFrame>();
    frame->shape[0] = height;
    frame->shape[1] = width;
    frame->shape[2] = 4;
    frame->strides[0] = static_cast<int64_t>(width) * 4;
    frame->strides[1] = 4;
    frame->strides[2] = 1;
    DLManagedTensor *tensor = &frame->tensor;
    tensor->dl_tensor.ndim = 3;
    tensor->dl_tensor.shape = frame->shape;
    tensor->dl_tensor.strides = frame->strides;
    tensor->dl_tensor.dtype = DLDataType{kDLUInt, 8, 1};
    tensor->dl_tensor.device = DLDevice{kDLCUDA, device};
    tensor->manager_ctx = frame.get();
    tensor->deleter = deleteDLManagedTensor;

    checkCuda(cudaSetDevice(device), "Failed to select CUDA device");
    checkCuda(
        cudaIpcOpenMemHandle(
            &tensor->dl_tensor.data, handle, cudaIpcMemLazyEnablePeerAccess
        ),
        "Failed to open CUDA IPC handle"
    );
    cudaPointerAttributes attributes{};
    cudaError_t err =
        cudaPointerGetAttributes(&attributes, tensor->dl_tensor.data);
    if (err != cudaSuccess || !attributes.devicePointer ||
        attributes.device != device) {
        cudaIpcCloseMemHandle(tensor->dl_tensor.data);
        checkCuda(err, "Failed to get CUDA pointer attributes");
        throw std::runtime_error(
            "CUDA IPC pointer does not belong to the selected device"
        );
    }
    frame.release();
    return tensor;
}
