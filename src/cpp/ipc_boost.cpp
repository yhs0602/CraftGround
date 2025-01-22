#include <pybind11/pybind11.h>
#include "ipc_boost.hpp"
#include "boost/interprocess/interprocess_fwd.hpp"
#include <cstddef>
#include <mutex>
#include <string>
#include <iostream>

bool shared_memory_exists(const std::string &name) {
    try {
        // Try to open the shared memory object
        shared_memory_object shm(open_only, name.c_str(), read_only);
        return true; // The shared memory exists
    } catch (const interprocess_exception &e) {
        // The shared memory does not exist
        return false;
    }
}

// Create shared memory and write initial environment data
int create_shared_memory_impl(
    int port,
    const char *initial_data,
    size_t data_size,
    size_t action_size,
    bool find_free_port
) {
    std::string p2j_memory_name, j2p_memory_name;
    bool found_free_port = false;

    try {
        do {
            p2j_memory_name = "craftground_" + std::to_string(port) + "_p2j";
            j2p_memory_name = "craftground_" + std::to_string(port) + "_j2p";
            if (shared_memory_exists(p2j_memory_name) ||
                shared_memory_exists(j2p_memory_name)) {
                if (find_free_port) {
                    port++;
                    continue;
                } else {
                    throw std::runtime_error(
                        "Shared memory " + p2j_memory_name + " or " +
                        j2p_memory_name + " already exists"
                    );
                }
            }
            found_free_port = true;
        } while (!found_free_port);
    } catch (const interprocess_exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr
            << "Failed to initialize shared memory during finding port: errno="
            << errno << std::endl;
        throw std::runtime_error(e.what());
    }
    errno = 0;

    if (!shared_memory_object::remove(p2j_memory_name.c_str())) {
        std::cerr << "Failed to remove shared memory: errno=" << errno
                  << std::endl;
    }

    managed_shared_memory p2jSharedMemory, j2pSharedMemory;
    try {
        p2jSharedMemory = managed_shared_memory(
            create_only,
            p2j_memory_name.c_str(),
            1024 // Too small size fails to allocate
        );
    } catch (const interprocess_exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Failed to initialize shared memory creating"
                  << p2j_memory_name << " : errno=" << errno << std::endl;
        throw std::runtime_error(e.what());
    }

    try {
        j2pSharedMemory = managed_shared_memory(
            create_only,
            j2p_memory_name.c_str(),
            1024 // Too small size fails to allocate
        );
    } catch (const interprocess_exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Failed to initialize shared memory creating"
                  << j2p_memory_name << " : errno=" << errno << std::endl;
        throw std::runtime_error(e.what());
    }
    J2PSharedMemoryLayout *j2pLayout = static_cast<J2PSharedMemoryLayout *>(
        j2pSharedMemory.allocate(sizeof(J2PSharedMemoryLayout))
    );
    j2pLayout->layout_size = sizeof(J2PSharedMemoryLayout);
    j2pLayout->data_offset = sizeof(J2PSharedMemoryLayout);
    j2pLayout->data_size = 0;

    SharedMemoryLayout *layout =
        static_cast<SharedMemoryLayout *>(p2jSharedMemory.allocate(
            sizeof(SharedMemoryLayout) + action_size + data_size
        ));
    layout->layout_size = sizeof(SharedMemoryLayout);
    layout->action_offset = sizeof(SharedMemoryLayout);
    layout->action_size = action_size;
    layout->initial_environment_offset =
        sizeof(SharedMemoryLayout) + action_size;
    layout->initial_environment_size = data_size;
    void *action_start =
        reinterpret_cast<char *>(layout) + layout->action_offset;
    void *data_start =
        reinterpret_cast<char *>(layout) + layout->initial_environment_offset;

    if (data_size > layout->initial_environment_size) {
        throw std::runtime_error(
            "Data size exceeds allocated shared memory size"
        );
    }
    std::memcpy(data_start, initial_data, data_size);
    layout->p2j_ready = true;
    layout->j2p_ready = false;
    return port;
}

// Write action to shared memory
void write_to_shared_memory_impl(
    const std::string &p2j_memory_name, const char *data
) {
    managed_shared_memory p2jMemory(open_only, p2j_memory_name.c_str());
    SharedMemoryLayout *layout =
        static_cast<SharedMemoryLayout *>(p2jMemory.get_address());
    char *action_addr =
        static_cast<char *>(p2jMemory.get_address()) + layout->action_offset;

    std::unique_lock<interprocess_mutex> actionLock(layout->mutex);
    std::memcpy(action_addr, data, layout->action_size);
    layout->p2j_ready = true;
    layout->j2p_ready = false;
    layout->condition.notify_one();
    actionLock.unlock();
}

// Read observation from shared memory
py::bytes read_from_shared_memory_impl(
    const std::string &p2j_memory_name, const std::string &j2p_memory_name
) {
    managed_shared_memory p2jMemory;
    try {
        p2jMemory = managed_shared_memory(open_only, p2j_memory_name.c_str());
    } catch (const interprocess_exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Failed to open shared memory to read observation: " << p2j_memory_name << " errno=" << errno
                  << std::endl;
        throw std::runtime_error(e.what());
    }
    SharedMemoryLayout *p2jLayout =
        static_cast<SharedMemoryLayout *>(p2jMemory.get_address());

    std::unique_lock<interprocess_mutex> lockSynchronization(p2jLayout->mutex);
    // wait for java to write the observation
    p2jLayout->condition.wait(lockSynchronization, [&] {
        return p2jLayout->j2p_ready;
    });

    // Read the observation from shared memory
    managed_shared_memory j2pMemory(open_only, j2p_memory_name.c_str());
    J2PSharedMemoryLayout *j2pLayout =
        static_cast<J2PSharedMemoryLayout *>(j2pMemory.get_address());

    const char *data_start =
        reinterpret_cast<char *>(j2pLayout) + j2pLayout->data_offset;
    py::bytes data(data_start, j2pLayout->data_size);

    p2jLayout->j2p_ready = false;
    p2jLayout->p2j_ready = false;
    lockSynchronization.unlock();

    return data;
}

// Destroy shared memory
void destroy_shared_memory_impl(const std::string &memory_name) {
    shared_memory_object::remove(memory_name.c_str());
}
