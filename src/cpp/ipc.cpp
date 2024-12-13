#include "ipc.h"

#ifdef __APPLE__
#include "ipc_apple.h"
py::object initialize_from_mach_port(int machPort, int width, int height) {
    DLManagedTensor* tensor = mtl_tensor_from_mach_port(machPort, width, height);
    return py::reinterpret_steal<py::object>(
        PyCapsule_New(tensor, "dltensor", [](PyObject* capsule) {
            DLManagedTensor* tensor = (DLManagedTensor*)PyCapsule_GetPointer(capsule, "dltensor");
            tensor->deleter(tensor);
        })
    );
}
py::object mtl_tensor_from_cuda_mem_handle(void *cuda_ipc_handle, int width, int height) {
    return py::none();
}

#elif __CUDA__
#include "ipc_cuda.h"
py::object initialize_from_mach_port(int machPort, int width, int height) {
    return py::none();
}

py::object mtl_tensor_from_cuda_mem_handle(void *cuda_ipc_handle, int width, int height) {
    DLManagedTensor* tensor = mtl_tensor_from_cuda_ipc_handle(cuda_ipc_handle, width, height);
    return py::reinterpret_steal<py::object>(
        PyCapsule_New(tensor, "dltensor", [](PyObject* capsule) {
            DLManagedTensor* tensor = (DLManagedTensor*)PyCapsule_GetPointer(capsule, "dltensor");
            tensor->deleter(tensor);
        })
    );
}

#else
py::object initialize_from_mach_port(int machPort, int width, int height) {
    return py::none();
}

py::object mtl_tensor_from_cuda_mem_handle(void *cuda_ipc_handle, int width, int height) {
    return py::none();
}
#endif