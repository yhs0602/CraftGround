#ifndef IPC_NOBOOST_HPP
#define IPC_NOBOOST_HPP

#include <cstddef>
#include <cstring>
#include <string>
#include <pybind11/pybind11.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define IS_WINDOWS 1
#define SHMEM_PREFIX "Global\\"
#else
#define SHMEM_PREFIX "/"
#define IS_WINDOWS 0
#endif
#include "cross_semaphore.h"

namespace py = pybind11;

struct SharedMemoryLayout {
    size_t layout_size;                // to be set on initialization
    size_t action_offset;              // Always sizeof(SharedMemoryLayout)
    size_t action_size;                // to be set on initialization
    size_t initial_environment_offset; // Always action_size +
                                       // sizeof(SharedDataHeader)
    size_t initial_environment_size;   // to be set on initialization
    rk_sema sem_obs_ready;
    rk_sema sem_action_ready;
};

struct J2PSharedMemoryLayout {
    size_t layout_size; // to be set on initialization
    size_t data_offset; // Always sizeof(J2PSharedMemoryLayout)
    size_t data_size;   // to be set on initialization
};

int create_shared_memory_impl(
    int port,
    const char *initial_data,
    size_t data_size,
    size_t action_size,
    bool find_free_port
);

void write_to_shared_memory_impl(
    const std::string &p2j_memory_name, const char *data,  size_t action_size
);

py::bytes read_from_shared_memory_impl(
    const std::string &p2j_memory_name, const std::string &j2p_memory_name
);

// remove shared memory노
void destroy_shared_memory_impl(const std::string &memory_name, bool release_semaphores);

#endif // SHARED_MEMORY_UTILS_HPP
