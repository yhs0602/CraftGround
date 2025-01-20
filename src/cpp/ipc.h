#ifndef __IPC_H__
#define __IPC_H__

#include <stdlib.h>
namespace py = pybind11;
py::object
initialize_from_mach_port(unsigned int machPort, int width, int height);
py::capsule mtl_tensor_from_cuda_mem_handle(
    const char *cuda_ipc_handle, int width, int height
);
void initialize_shared_memory(
    const char *memory_name,
    const char *synchronization_memory_name,
    const char *action_memory_name,
    const char *initial_data,
    size_t data_size,
    size_t action_size
);

void write_to_shared_memory(
    const char *memory_name, const char *data, const size_t data_size
);

py::bytes read_from_shared_memory(
    const char *memory_name, const char *management_memory_name
);

void destroy_shared_memory(const char *memory_name);

#endif