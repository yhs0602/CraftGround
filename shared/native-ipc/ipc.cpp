#include <iostream>
#include <pybind11/pybind11.h>
#include "ipc.h"
#define MACRO_STRINGIFY(x) #x

#ifdef __APPLE__

#include "ipc_apple.h"
py::object
initialize_from_mach_port(unsigned int machPort, int width, int height) {
    return mtl_tensor_from_mach_port(machPort, width, height);
}
py::capsule mtl_tensor_from_cuda_mem_handle(
    const char *cuda_ipc_handle, int width, int height
) {
    return py::none();
}

#elif HAS_CUDA
#include "ipc_cuda.h"
py::object
initialize_from_mach_port(unsigned int machPort, int width, int height) {
    return py::none();
}

py::capsule mtl_tensor_from_cuda_mem_handle(
    const char *cuda_ipc_handle, int width, int height
) {
    DLManagedTensor *tensor = mtl_tensor_from_cuda_ipc_handle(
        reinterpret_cast<void *>(const_cast<char *>(cuda_ipc_handle)),
        width,
        height
    );

    if (!tensor) {
        throw std::runtime_error(
            "Failed to create DLManagedTensor from CUDA IPC handle"
        );
    }

    return py::capsule(tensor, "dltensor", [](void *ptr) {
        DLManagedTensor *managed_tensor = static_cast<DLManagedTensor *>(ptr);
        if (managed_tensor && managed_tensor->deleter) {
            managed_tensor->deleter(managed_tensor);
        }
    });
}

#else
py::object
initialize_from_mach_port(unsigned int machPort, int width, int height) {
    return py::none();
}

py::capsule mtl_tensor_from_cuda_mem_handle(
    const char *cuda_ipc_handle, int width, int height
) {
    return py::none();
}
#endif

// #include "ipc_boost.hpp"
#include "ipc_noboost.hpp"

int initialize_shared_memory(
    int port,
    const char *initial_data,
    size_t data_size,
    size_t action_size,
    bool find_free_port
) {
    try {
        return create_shared_memory_impl(
            port, initial_data, data_size, action_size, find_free_port
        );
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Failed to initialize shared memory: errno=" << errno
                  << std::endl;
        throw std::runtime_error(e.what());
    }
}

void write_to_shared_memory(
    const char *p2j_memory_name, const char *data, size_t action_size
) {
    write_to_shared_memory_impl(p2j_memory_name, data, action_size);
}

py::bytes read_from_shared_memory(
    const char *p2j_memory_name, const char *j2p_memory_name
) {
    return read_from_shared_memory_impl(p2j_memory_name, j2p_memory_name);
}

void destroy_shared_memory(const char *memory_name, bool release_semaphores) {
    destroy_shared_memory_impl(memory_name, release_semaphores);
}

PYBIND11_MODULE(craftground_native, m) {
    m.doc() = "Craftground Native Module";
    m.def("initialize_from_mach_port", &initialize_from_mach_port);
    m.def("mtl_tensor_from_cuda_mem_handle", &mtl_tensor_from_cuda_mem_handle);
    m.def("initialize_shared_memory", &initialize_shared_memory);
    m.def("write_to_shared_memory", &write_to_shared_memory);
    m.def("read_from_shared_memory", &read_from_shared_memory);
    m.def("destroy_shared_memory", &destroy_shared_memory);
    m.def("shared_memory_exists", &shared_memory_exists);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
