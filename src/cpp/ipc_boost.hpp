#ifndef SHARED_MEMORY_UTILS_HPP
#define SHARED_MEMORY_UTILS_HPP

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <cstddef>
#include <cstring>
#include <string>
#include <pybind11/pybind11.h>

using namespace boost::interprocess;
namespace py = pybind11;

struct SharedMemoryLayout {
    size_t layout_size;                // to be set on initialization
    size_t action_offset;              // Always sizeof(SharedMemoryLayout)
    size_t action_size;                // to be set on initialization
    size_t initial_environment_offset; // Always action_size +
                                       // sizeof(SharedDataHeader)
    size_t initial_environment_size;   // to be set on initialization
    interprocess_mutex mutex;
    interprocess_condition condition;
    bool p2j_ready;
    bool j2p_ready;
    bool p2j_recv_ready;
    bool j2p_recv_ready;
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
    const std::string &p2j_memory_name, const char *data
);

py::bytes read_from_shared_memory_impl(
    const std::string &p2j_memory_name, const std::string &j2p_memory_name
);

// remove shared memory
void destroy_shared_memory_impl(const std::string &memory_name, bool release_semaphores);

#endif // SHARED_MEMORY_UTILS_HPP
