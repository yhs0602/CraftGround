#ifndef __IPC_H__
#define __IPC_H__

#include <stdlib.h>
#include <string>

namespace py = pybind11;
py::object
initialize_from_mach_port(unsigned int machPort, int width, int height);
py::capsule mtl_tensor_from_cuda_mem_handle(
    const char *cuda_ipc_handle, int width, int height
);

int initialize_shared_memory(
    int port,
    const char *initial_data,
    size_t data_size,
    size_t action_size,
    bool find_free_port
);

void write_to_shared_memory(
    const char *p2j_memory_name, const char *data, size_t action_size
);

py::bytes read_from_shared_memory(
    const char *p2j_memory_name, const char *j2p_memory_name
);

void destroy_shared_memory(const char *memory_name, bool release_semaphores);

bool shared_memory_exists(const std::string &name);

#endif