#include <pybind11/pybind11.h>
#include "ipc_boost.hpp"
#include "boost/interprocess/detail/os_file_functions.hpp"
#include "boost/interprocess/interprocess_fwd.hpp"
#include "boost/interprocess/mapped_region.hpp"
#include "boost/interprocess/shared_memory_object.hpp"
#include <cstddef>
#include <mutex>
#include <string>
#include <iostream>
#include "print_hex.h"

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

    shared_memory_object p2jSharedMemory, j2pSharedMemory;
    try {
        p2jSharedMemory = shared_memory_object(
            create_only, p2j_memory_name.c_str(), read_write
        );
        p2jSharedMemory.truncate(1024); // Too small size fails to allocate
    } catch (const interprocess_exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Failed to initialize shared memory creating"
                  << p2j_memory_name << " : errno=" << errno << std::endl;
        throw std::runtime_error(e.what());
    }

    try {
        j2pSharedMemory = shared_memory_object(
            create_only, j2p_memory_name.c_str(), read_write
        );
        j2pSharedMemory.truncate(1024); // Too small size fails to allocate
    } catch (const interprocess_exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Failed to initialize shared memory creating"
                  << j2p_memory_name << " : errno=" << errno << std::endl;
        throw std::runtime_error(e.what());
    }
    mapped_region j2pRegion(j2pSharedMemory, read_write);
    mapped_region p2jRegion(p2jSharedMemory, read_write);
    J2PSharedMemoryLayout *j2pLayout =
        static_cast<J2PSharedMemoryLayout *>(j2pRegion.get_address());
    j2pLayout->layout_size = sizeof(J2PSharedMemoryLayout);
    j2pLayout->data_offset = sizeof(J2PSharedMemoryLayout);
    j2pLayout->data_size = 0;

    SharedMemoryLayout *p2jLayout =
        static_cast<SharedMemoryLayout *>(p2jRegion.get_address());
    p2jLayout->layout_size = sizeof(SharedMemoryLayout);
    p2jLayout->action_offset = sizeof(SharedMemoryLayout);
    p2jLayout->action_size = action_size;
    p2jLayout->initial_environment_offset =
        sizeof(SharedMemoryLayout) + action_size;
    p2jLayout->initial_environment_size = data_size;
    new (&p2jLayout->mutex) interprocess_mutex();
    new (&p2jLayout->condition) interprocess_condition();

    void *action_start =
        reinterpret_cast<char *>(p2jLayout) + p2jLayout->action_offset;
    void *data_start = reinterpret_cast<char *>(p2jLayout) +
                       p2jLayout->initial_environment_offset;

    if (data_size > p2jLayout->initial_environment_size) {
        throw std::runtime_error(
            "Data size exceeds allocated shared memory size"
        );
    }
    std::memcpy(data_start, initial_data, data_size);

    std::cout << "Wrote initial data to shared memory:" << std::endl;
    printHex(initial_data, data_size);
    std::cout << "Data size: " << data_size << std::endl;
    std::cout << "Action size: " << action_size << std::endl;
    std::cout << "Initial environment offset: "
              << p2jLayout->initial_environment_offset << std::endl;
    std::cout << "Initial environment size: "
              << p2jLayout->initial_environment_size << std::endl;
    std::cout << "Action offset: " << p2jLayout->action_offset << std::endl;
    std::cout << "Action size: " << p2jLayout->action_size << std::endl;
    std::cout << "p2j ready: " << p2jLayout->p2j_ready << std::endl;
    std::cout << "j2p ready: " << p2jLayout->j2p_ready << std::endl;

    p2jLayout->p2j_ready = false;
    p2jLayout->j2p_ready = false;
    p2jLayout->p2j_recv_ready = false;
    p2jLayout->j2p_recv_ready = false;
    return port;
}

// Write action to shared memory
void write_to_shared_memory_impl(
    const std::string &p2j_memory_name, const char *data
) {
    shared_memory_object p2jMemory(
        open_only, p2j_memory_name.c_str(), read_write
    );
    mapped_region p2jRegion(p2jMemory, read_write);
    SharedMemoryLayout *layout =
        static_cast<SharedMemoryLayout *>(p2jRegion.get_address());
    char *action_addr =
        reinterpret_cast<char *>(layout) + layout->action_offset;

    std::cout << "Writing action to shared memory" << std::endl;
    std::unique_lock<interprocess_mutex> actionLock(layout->mutex);
    layout->condition.wait(actionLock, [&] { return !layout->p2j_recv_ready; });
    std::memcpy(action_addr, data, layout->action_size);
    layout->p2j_ready = true;
    layout->j2p_ready = false;
    layout->p2j_recv_ready = false;
    layout->j2p_recv_ready = false;
    layout->condition.notify_one();
    actionLock.unlock();
    std::cout << "Wrote action to shared memory" << std::endl;
}

// Read observation from shared memory
py::bytes read_from_shared_memory_impl(
    const std::string &p2j_memory_name, const std::string &j2p_memory_name
) {
    shared_memory_object p2jMemory;
    try {
        p2jMemory = shared_memory_object(
            open_only, p2j_memory_name.c_str(), read_write
        );
    } catch (const interprocess_exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Failed to open shared memory to read observation: "
                  << p2j_memory_name << " errno=" << errno << std::endl;
        throw std::runtime_error(e.what());
    }
    mapped_region p2jRegion(p2jMemory, read_write);
    SharedMemoryLayout *p2jLayout =
        static_cast<SharedMemoryLayout *>(p2jRegion.get_address());

    std::unique_lock<interprocess_mutex> lockSynchronization(p2jLayout->mutex);
    p2jLayout->j2p_recv_ready = true;
    p2jLayout->condition.notify_all();
    lockSynchronization.unlock();
    lockSynchronization.lock();
    // wait for java to write the observation
    std::cout << "Waiting for Java to write observation" << std::endl;
    p2jLayout->condition.wait(lockSynchronization, [&] {
        return p2jLayout->j2p_ready;
    });

    // Read the observation from shared memory
    shared_memory_object j2pMemory(
        open_only, j2p_memory_name.c_str(), read_write
    );
    mapped_region j2pMemoryRegion(j2pMemory, read_write);
    J2PSharedMemoryLayout *j2pLayout =
        static_cast<J2PSharedMemoryLayout *>(j2pMemoryRegion.get_address());

    const char *data_start =
        reinterpret_cast<char *>(j2pLayout) + j2pLayout->data_offset;
    py::bytes data(data_start, j2pLayout->data_size);

    p2jLayout->j2p_ready = false;
    p2jLayout->p2j_ready = false;
    p2jLayout->j2p_recv_ready = false;
    p2jLayout->p2j_recv_ready = false;
    lockSynchronization.unlock();
    std::cout << "Read observation from shared memory" << std::endl;

    return data;
}

// Destroy shared memory
void destroy_shared_memory_impl(
    const std::string &memory_name, bool destroy_semaphores
) {
    shared_memory_object::remove(memory_name.c_str());
}
