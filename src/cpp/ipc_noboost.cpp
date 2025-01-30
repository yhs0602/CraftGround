#include <cstddef>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/semaphore.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "ipc_noboost.hpp"
#include "cross_semaphore.h"
#include "print_hex.h"

bool shared_memory_exists(const std::string &name) {
    int fd = shm_open(name.c_str(), O_RDONLY, 0666);
    if (fd == -1) {
        return false; // Shared memory does not exist
    }
    close(fd);   // Close the file descriptor
    return true; // Shared memory exists
}

std::string make_shared_memory_name(int port, const std::string &suffix) {
    return SHMEM_PREFIX "craftground_" + std::to_string(port) + "_" + suffix;
}

int create_shared_memory_impl(
    int port,
    const char *initial_data,
    size_t data_size,
    size_t action_size,
    bool find_free_port
) {
    std::string p2j_memory_name, j2p_memory_name;
    bool found_free_port = false;
    const int shared_memory_size =
        sizeof(SharedMemoryLayout) + action_size + data_size;

    do {
        p2j_memory_name = make_shared_memory_name(port, "p2j");
        j2p_memory_name = make_shared_memory_name(port, "j2p");
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

    int p2jFd = shm_open(p2j_memory_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (p2jFd == -1) {
        perror("shm_open failed while creating shared memory p2j");
        return -1;
    }

    if (ftruncate(p2jFd, shared_memory_size) == -1) {
        perror("ftruncate failed for p2jFd");
        close(p2jFd);
        shm_unlink(p2j_memory_name.c_str());
        return -1;
    }

    int j2pFd = shm_open(j2p_memory_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (j2pFd == -1) {
        close(p2jFd);
        shm_unlink(p2j_memory_name.c_str());
        perror("shm_open failed while creating shared memory j2p");
        return -1;
    }

    if (ftruncate(j2pFd, sizeof(J2PSharedMemoryLayout) + data_size) == -1) {
        perror("ftruncate failed for j2pFd");
        close(j2pFd);
        close(p2jFd);
        shm_unlink(j2p_memory_name.c_str());
        shm_unlink(p2j_memory_name.c_str());
        return -1;
    }

    void *ptr = mmap(
        0, shared_memory_size, PROT_READ | PROT_WRITE, MAP_SHARED, p2jFd, 0
    );
    if (ptr == MAP_FAILED) {
        perror("mmap failed while creating shared memory");
        close(j2pFd);
        close(p2jFd);
        shm_unlink(j2p_memory_name.c_str());
        shm_unlink(j2p_memory_name.c_str());
        return -1;
    }

    SharedMemoryLayout *p2jLayout = static_cast<SharedMemoryLayout *>(ptr);
    new (p2jLayout) SharedMemoryLayout();

    std::string sema_obs_ready_name = make_shared_memory_name(
        port, "cg_sem_obs" + std::to_string(port)
    );
    std::string sema_action_ready_name = make_shared_memory_name(
        port, "cg_sem_act" + std::to_string(port)
    );

    rk_sema_init(&p2jLayout->sem_obs_ready, sema_obs_ready_name.c_str(), 0, 1);
    rk_sema_init(
        &p2jLayout->sem_action_ready, sema_action_ready_name.c_str(), 0, 1
    );

    p2jLayout->action_offset = sizeof(SharedMemoryLayout);
    p2jLayout->action_size = action_size;
    p2jLayout->initial_environment_offset =
        sizeof(SharedMemoryLayout) + action_size;
    p2jLayout->initial_environment_size = data_size;

    void *action_start = (char *)ptr + p2jLayout->action_offset;
    void *data_start = (char *)ptr + p2jLayout->initial_environment_offset;
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

    munmap(ptr, shared_memory_size);
    close(p2jFd);
    close(j2pFd);
    return port;
}

void write_to_shared_memory_impl(
    const std::string &p2j_memory_name, const char *data, size_t action_size
) {
    int p2jFd = shm_open(p2j_memory_name.c_str(), O_RDWR, 0666);
    void *ptr = mmap(
        0,
        sizeof(SharedMemoryLayout),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        p2jFd,
        0
    );
    if (ptr == MAP_FAILED) {
        perror("mmap failed while writing to shared memory");
        close(p2jFd);
        return;
    }

    SharedMemoryLayout *layout = static_cast<SharedMemoryLayout *>(ptr);
    layout->action_size = action_size;
    const size_t action_offset = layout->action_offset;
    munmap(ptr, sizeof(SharedMemoryLayout));
    ptr = mmap(
        0,
        sizeof(SharedMemoryLayout) + action_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        p2jFd,
        0
    );
    layout = static_cast<SharedMemoryLayout *>(ptr);
    std::cout << "Writing action to shared memory" << std::endl;
    rk_sema_wait(&layout->sem_action_ready);
    std::memcpy((char *)ptr + layout->action_offset, data, layout->action_size);
    rk_sema_post(&layout->sem_action_ready);
    std::cout << "Wrote action to shared memory" << std::endl;
    munmap(ptr, sizeof(SharedMemoryLayout) + action_size);
    close(p2jFd);
}

py::bytes read_from_shared_memory_impl(
    const std::string &p2j_memory_name, const std::string &j2p_memory_name
) {
    std::cout << "Reading observation from shared memory 1" << std::endl;
    int p2jFd = shm_open(p2j_memory_name.c_str(), O_RDWR, 0666);
    if (p2jFd == -1) {
        perror("shm_open p2j failed while reading from shared memory");
        return py::bytes(); // return empty bytes
    }

    void *p2jPtr = mmap(
        0,
        sizeof(SharedMemoryLayout),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        p2jFd,
        0
    );
    if (p2jPtr == MAP_FAILED) {
        perror("mmap p2j failed while reading from shared memory");
        close(p2jFd);
        return py::bytes();
    }
    SharedMemoryLayout *layout = static_cast<SharedMemoryLayout *>(p2jPtr);
    size_t action_size = layout->action_size;
    size_t data_size = layout->initial_environment_size;

    std::cout << "Reading observation from shared memory 2" << std::endl;

    // Wait for the observation to be ready
    rk_sema_wait(&layout->sem_obs_ready);

    int j2pFd = shm_open(j2p_memory_name.c_str(), O_RDWR, 0666);
    if (j2pFd == -1) {
        perror("shm_open j2p failed while reading from shared memory");
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
        return py::bytes(); // return empty bytes
    }

    void *j2pPtr = mmap(
        0,
        sizeof(J2PSharedMemoryLayout),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        j2pFd,
        0
    );

    if (j2pPtr == MAP_FAILED) {
        perror("mmap j2p failed while reading from shared memory");
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
        close(p2jFd);
        close(j2pFd);
        return py::bytes();
    }

    J2PSharedMemoryLayout *j2pLayout =
        static_cast<J2PSharedMemoryLayout *>(j2pPtr);

    const size_t obs_length = j2pLayout->data_size;
    munmap(j2pPtr, sizeof(J2PSharedMemoryLayout));

    j2pPtr = mmap(
        0,
        sizeof(J2PSharedMemoryLayout) + obs_length,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        j2pFd,
        0
    );

    if (j2pPtr == MAP_FAILED) {
        perror("mmap j2p failed while reading from shared memory");
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
        close(p2jFd);
        close(j2pFd);
        return py::bytes();
    }

    j2pLayout = static_cast<J2PSharedMemoryLayout *>(j2pPtr);
    const char *data_start = (char *)j2pPtr + j2pLayout->data_offset;

    py::bytes data(data_start, j2pLayout->data_size);
    rk_sema_post(&layout->sem_obs_ready); // Notify that the observation is read

    std::cout << "Read observation from shared memory" << std::endl;
    munmap(j2pPtr, sizeof(J2PSharedMemoryLayout) + obs_length);
    munmap(p2jPtr, sizeof(SharedMemoryLayout));
    close(p2jFd);
    close(j2pFd);
    return data;
}

// remove shared memory
void destroy_shared_memory_impl(
    const std::string &memory_name, bool release_semaphores
) {
    if (release_semaphores) {
        int p2jFd = shm_open(memory_name.c_str(), O_RDWR, 0666);
        if (p2jFd == -1) {
            perror("shm_open failed while destroying shared memory");
        } else {
            void *ptr = mmap(
                0,
                sizeof(SharedMemoryLayout),
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                p2jFd,
                0
            );
            if (ptr == MAP_FAILED) {
                perror("mmap failed while destroying shared memory");
            } else {
                SharedMemoryLayout *layout =
                    static_cast<SharedMemoryLayout *>(ptr);
                rk_sema_destroy(&layout->sem_obs_ready);
                rk_sema_destroy(&layout->sem_action_ready);
                munmap(ptr, sizeof(SharedMemoryLayout));
            }
            close(p2jFd);
        }
    }
    shm_unlink(memory_name.c_str());
}
