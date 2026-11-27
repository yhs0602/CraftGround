#include <cstddef>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include "ipc_noboost.hpp"
#include "cross_semaphore.h"
#include "print_hex.h"

#if defined(WIN32) || defined(_WIN32) ||                                       \
    defined(__WIN32) && !defined(__CYGWIN__)
#define _WIN32 1
#include <windows.h>
#include <cstdint>
#include <cstdio>

// Define some POSIX-like constants for use with our mmap_win
#ifndef PROT_READ
#define PROT_READ 0x1
#endif
#ifndef PROT_WRITE
#define PROT_WRITE 0x2
#endif
#ifndef MAP_SHARED
#define MAP_SHARED 0x01
#endif
#ifndef MAP_FAILED
#define MAP_FAILED ((void *)-1)
#endif

// Windows does not need an unlink; when all handles are closed the mapping is
// gone.
int shm_unlink_win(const char *name) { return 0; }

// ftruncate is not needed on Windows because the size is set at creation.
int ftruncate_win(int /*fd*/, size_t /*size*/) { return 0; }

// Our wrapper for opening/creating a shared memory “file”
// (Note: when creating, we pass in the desired size.)
int shm_open_wrapper(const char *name, int oflag, int mode, size_t size) {
    HANDLE hMap;
    DWORD dwDesiredAccess = FILE_MAP_READ | FILE_MAP_WRITE;
    if (oflag & O_CREAT) {
        hMap = CreateFileMapping(
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            (DWORD)((size >> 32) & 0xFFFFFFFF),
            (DWORD)(size & 0xFFFFFFFF),
            name
        );
        if (hMap == NULL) {
            fprintf(
                stderr,
                "CreateFileMapping failed for %s with error %lu\n",
                name,
                GetLastError()
            );
            return -1;
        }
    } else {
        hMap = OpenFileMapping(dwDesiredAccess, FALSE, name);
        if (hMap == NULL) {
            fprintf(
                stderr,
                "OpenFileMapping failed for %s with error %lu\n",
                name,
                GetLastError()
            );
            return -1;
        }
    }
    return (int)(intptr_t)hMap;
}

// Our wrapper for mapping a view of the shared memory
void *mmap_win(
    void * /*addr*/,
    size_t length,
    int prot,
    int /*flags*/,
    int fd,
    size_t offset
) {
    HANDLE hMap = (HANDLE)(intptr_t)fd;
    DWORD dwDesiredAccess = 0;
    if (prot & PROT_READ)
        dwDesiredAccess |= FILE_MAP_READ;
    if (prot & PROT_WRITE)
        dwDesiredAccess |= FILE_MAP_WRITE;
    void *map = MapViewOfFile(
        hMap,
        dwDesiredAccess,
        (DWORD)((offset >> 32) & 0xFFFFFFFF),
        (DWORD)(offset & 0xFFFFFFFF),
        length
    );
    if (map == NULL) {
        fprintf(
            stderr, "MapViewOfFile failed with error %lu\n", GetLastError()
        );
        return MAP_FAILED;
    }
    return map;
}

int munmap_win(void *addr, size_t /*length*/) {
    if (!UnmapViewOfFile(addr)) {
        fprintf(
            stderr, "UnmapViewOfFile failed with error %lu\n", GetLastError()
        );
        return -1;
    }
    return 0;
}

int close_win(int fd) {
    HANDLE hMap = (HANDLE)(intptr_t)fd;
    if (!CloseHandle(hMap)) {
        fprintf(stderr, "CloseHandle failed with error %lu\n", GetLastError());
        return -1;
    }
    return 0;
}

// To differentiate “create” versus “open” calls we define two macros:
#define shm_open_create(name, mode, size)                                      \
    shm_open_wrapper(name, O_CREAT | O_RDWR, mode, size)
#define shm_open_existing(name, mode) shm_open_wrapper(name, O_RDWR, mode, 0)
#define ftruncate(fd, size) ftruncate_win(fd, size)
#define mmap(addr, length, prot, flags, fd, offset)                            \
    mmap_win(addr, length, prot, flags, fd, offset)
#define munmap(addr, length) munmap_win(addr, length)
#define close(fd) close_win(fd)
#define shm_unlink(name) shm_unlink_win(name)

#else
// On POSIX systems use the normal headers.
#include "posix_shm.h"
#include <sys/mman.h>
#include <unistd.h>
#include <csignal>
#endif

#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif

bool shared_memory_exists(const std::string &name) {
#ifdef _WIN32
    HANDLE hMap = OpenFileMapping(FILE_MAP_READ, FALSE, name.c_str());
    if (hMap == NULL)
        return false;
    CloseHandle(hMap);
    return true;
#else
    int fd = shm_open(name.c_str(), O_RDONLY, 0666);
    if (fd == -1)
        return false;
    close(fd);
    return true;
#endif
}

std::string make_shared_memory_name(int port, const std::string &suffix) {
    return SHMEM_PREFIX "craftground_" + std::to_string(port) + "_" + suffix;
}

std::unordered_map<std::string, int> shm_fd_cache;
std::mutex shm_fd_cache_mutex;

int get_shared_memory_fd(const std::string &memory_name) {
    std::lock_guard<std::mutex> lock(shm_fd_cache_mutex);
    if (shm_fd_cache.count(memory_name))
        return shm_fd_cache[memory_name];
#ifdef _WIN32
    int fd = shm_open_existing(memory_name.c_str(), 0666);
#else
    int fd = shm_open(memory_name.c_str(), O_RDWR, 0666);
#endif
    if (fd == -1) {
        perror(("shm_open failed for " + memory_name).c_str());
        return -1;
    }
    shm_fd_cache[memory_name] = fd;
    return fd;
}

void close_shared_memory_fd(const std::string &memory_name) {
    std::lock_guard<std::mutex> lock(shm_fd_cache_mutex);
    if (shm_fd_cache.count(memory_name)) {
        close(shm_fd_cache[memory_name]);
        shm_fd_cache.erase(memory_name);
    }
}

void *map_shared_memory(int fd, size_t size) {
    void *ptr =
        mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap failed while mapping shared memory");
        return nullptr;
    }
    return ptr;
}

std::unordered_map<std::thread::id, std::pair<std::string, std::string>>
    shm_map;
std::mutex shm_map_mutex;

void signal_handler(int signal) {
    std::cout << "Received signal " << signal
              << ", cleaning up shared memory..." << std::endl;
    std::lock_guard<std::mutex> lock(shm_map_mutex);

    for (const auto &[tid, names] : shm_map) {
        shm_unlink(names.first.c_str());
        shm_unlink(names.second.c_str());
    }

    exit(1);
}

void register_signal_handlers() {
#ifndef _WIN32
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGTERM, &sa, nullptr); // kill, systemd
    sigaction(SIGINT, &sa, nullptr);  // Ctrl+C
    sigaction(SIGHUP, &sa, nullptr);  // Terminal closed
    sigaction(SIGQUIT, &sa, nullptr); // Quit Signal
#endif
}

int create_shared_memory_impl(
    int port,
    const char *initial_data,
    size_t data_size,
    size_t action_size,
    bool find_free_port
) {
    std::string p2j_memory_name, j2p_memory_name;
    int p2jFd;
    const size_t shared_memory_size =
        sizeof(SharedMemoryLayout) + action_size + data_size;

    for (;;) {
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
#ifdef _WIN32
        p2jFd =
            shm_open_create(p2j_memory_name.c_str(), 0666, shared_memory_size);
#else
        p2jFd =
            shm_open(p2j_memory_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (p2jFd < 0 && errno == EEXIST && find_free_port) {
            ++port;
            continue;
        }
#endif
        break;
    }
#ifndef _WIN32
    if (p2jFd != -1 && ftruncate(p2jFd, shared_memory_size) == -1) {
        perror("ftruncate failed for p2jFd");
        close(p2jFd);
        shm_unlink(p2j_memory_name.c_str());
        return -1;
    }
#endif
    if (p2jFd == -1) {
        perror("shm_open failed while creating shared memory p2j");
        return -1;
    }

#ifdef _WIN32
    int j2pFd = shm_open_create(
        j2p_memory_name.c_str(), 0666, sizeof(J2PSharedMemoryLayout) + data_size
    );
#else
    int j2pFd =
        shm_open(j2p_memory_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (j2pFd != -1 &&
        ftruncate(j2pFd, sizeof(J2PSharedMemoryLayout) + data_size) == -1) {
        perror("ftruncate failed for j2pFd");
        close(j2pFd);
        close(p2jFd);
        shm_unlink(j2p_memory_name.c_str());
        shm_unlink(p2j_memory_name.c_str());
        return -1;
    }
#endif

    if (j2pFd == -1) {
        close(p2jFd);
        shm_unlink(p2j_memory_name.c_str());
        throw std::runtime_error("Could not create observation shared memory");
    }
    std::lock_guard<std::mutex> lock(shm_map_mutex);
    shm_map[std::this_thread::get_id()] = {p2j_memory_name, j2p_memory_name};

    void *ptr = mmap(
        0, shared_memory_size, PROT_READ | PROT_WRITE, MAP_SHARED, p2jFd, 0
    );
    if (ptr == MAP_FAILED) {
        perror("mmap failed while creating shared memory");
        close(j2pFd);
        close(p2jFd);
        shm_unlink(p2j_memory_name.c_str());
        shm_unlink(j2p_memory_name.c_str());
        return -1;
    }

    SharedMemoryLayout *p2jLayout = static_cast<SharedMemoryLayout *>(ptr);
    new (p2jLayout) SharedMemoryLayout();

#ifndef _WIN32
    try {
        // The exclusive control object reserves this port, including its
        // semaphore names.
        sem_unlink((j2p_memory_name + "f").c_str());
        craftground::NamedSemaphore slot(j2p_memory_name + "f", true);
    } catch (...) {
        munmap(ptr, shared_memory_size);
        close(p2jFd);
        close(j2pFd);
        shm_unlink(p2j_memory_name.c_str());
        shm_unlink(j2p_memory_name.c_str());
        throw;
    }
    p2jLayout->layout_size = shared_memory_size;
#endif
    std::string sema_obs_ready_name =
        SHMEM_PREFIX "cg_sem_obs" + std::to_string(port);
    std::string sema_action_ready_name =
        SHMEM_PREFIX "cg_sem_act" + std::to_string(port);

    rk_sema_init(&p2jLayout->sem_obs_ready, sema_obs_ready_name.c_str(), 0, 1);
    rk_sema_init(
        &p2jLayout->sem_action_ready, sema_action_ready_name.c_str(), 0, 1
    );
    // std::cout << "Initialized semaphore for Python"
    //           << p2jLayout->sem_obs_ready.name << std::endl;
    // std::cout << "Initialized semaphore for Python"
    //           << p2jLayout->sem_action_ready.name << std::endl;

    p2jLayout->action_offset = sizeof(SharedMemoryLayout);
    p2jLayout->action_size = action_size;
    p2jLayout->initial_environment_offset =
        sizeof(SharedMemoryLayout) + action_size;
    p2jLayout->initial_environment_size = data_size;

    void *action_start = (char *)ptr + p2jLayout->action_offset;
    void *data_start = (char *)ptr + p2jLayout->initial_environment_offset;
    std::memcpy(data_start, initial_data, data_size);
    // std::cout << "Wrote initial data to shared memory:" << std::endl;

    // printHex(initial_data, data_size);
    // std::cout << "Data size: " << data_size << std::endl;
    // std::cout << "Action size: " << action_size << std::endl;
    // std::cout << "Initial environment offset: "
    //           << p2jLayout->initial_environment_offset << std::endl;
    // std::cout << "Initial environment size: "
    //           << p2jLayout->initial_environment_size << std::endl;
    // std::cout << "Action offset: " << p2jLayout->action_offset << std::endl;
    // std::cout << "Action size: " << p2jLayout->action_size << std::endl;

    if (munmap(ptr, shared_memory_size) == -1) {
        perror("munmap failed while creating shared memory");
    }
    close(p2jFd);
    close(j2pFd);
    return port;
}

void write_to_shared_memory_impl(
    const std::string &p2j_memory_name, const char *data, size_t action_size
) {
    // p2jFd is owned by the shm_fd_cache in get_shared_memory_fd() and is
    // reused across calls - it must not be close()d here, only by
    // close_shared_memory_fd() (via destroy_shared_memory_impl). Closing it
    // here left later calls reusing a cached-but-already-closed fd number,
    // breaking action delivery after the first write (surfaced as spurious
    // EACCES/EBADF errors from mmap).
#ifndef _WIN32
    // The JVM has already consumed the initial configuration when actions
    // begin. Reuse the inline payload space; larger actions get a separate
    // named object.
    craftground::ShmMapping control(
        p2j_memory_name.c_str(),
        sizeof(SharedMemoryLayout),
        craftground::ShmAccess::Write
    );
    auto *header = reinterpret_cast<SharedMemoryLayout *>(control.data());
    size_t offset = sizeof(*header);
    if (header->layout_size < sizeof(*header))
        throw std::runtime_error("Invalid shared-memory control header");
    if (action_size > header->layout_size - sizeof(*header)) {
        // Older runtime wheels only understand inline actions.
        std::string j2p_name =
            p2j_memory_name.substr(0, p2j_memory_name.size() - 3) + "j2p";
        craftground::ShmMapping last_frame(
            j2p_name.c_str(),
            sizeof(J2PSharedMemoryLayout),
            craftground::ShmAccess::Read
        );
        if (!reinterpret_cast<const J2PSharedMemoryLayout *>(last_frame.data())
                 ->layout_size) {
            throw std::runtime_error(
                "This action requires an updated shared-memory runtime"
            );
        }
        offset = 0;
    }
    if (action_size) {
        const std::string name =
            offset ? p2j_memory_name : p2j_memory_name + "a";
        craftground::ShmMapping payload(
            name.c_str(),
            offset + action_size,
            offset ? craftground::ShmAccess::Write
                   : craftground::ShmAccess::Grow
        );
        std::memcpy(payload.data() + offset, data, action_size);
    }
    header->action_offset = offset;
    header->action_size = action_size;
    craftground::NamedSemaphore(header->sem_action_ready.name).post();
#else
    int p2jFd = get_shared_memory_fd(p2j_memory_name.c_str());
    if (p2jFd == -1) {
        perror("shm_open failed while writing to shared memory");
        return;
    }
    void *ptr = mmap(
        0,
        sizeof(SharedMemoryLayout) + action_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        p2jFd,
        0
    );
    if (ptr == MAP_FAILED) {
        perror("mmap failed while writing to shared memory");
        return;
    }
    SharedMemoryLayout *layout = static_cast<SharedMemoryLayout *>(ptr);
    layout->action_size = action_size;
    layout->action_offset = sizeof(SharedMemoryLayout);
    std::memcpy((char *)ptr + layout->action_offset, data, layout->action_size);
    rk_sema_open(&layout->sem_obs_ready);
    // Must be synchronous: async_rk_sema_post() posts from a detached thread
    // that captures a pointer into `ptr`, but this function munmap()s `ptr`
    // right after - the detached thread would then race to dereference
    // already-unmapped memory, segfaulting nondeterministically (observed on
    // the 2nd+ step in ZEROCOPY/JAX mode, once earlier bugs stopped masking
    // this path).
    if (rk_sema_post(&layout->sem_action_ready) < 0) {
        perror("Failed to post semaphore");
    }
    // std::cout << "Wrote action to shared memory" << std::endl;
    munmap(ptr, sizeof(SharedMemoryLayout) + action_size);
#endif
}

py::bytes read_from_shared_memory_impl(
    const std::string &p2j_memory_name,
    const std::string &j2p_memory_name,
    py::object server_event, int timeout_ms
) {
#ifndef _WIN32
    craftground::ShmMapping control(
        p2j_memory_name.c_str(),
        sizeof(SharedMemoryLayout),
        craftground::ShmAccess::Read
    );
    auto *layout = reinterpret_cast<const SharedMemoryLayout *>(control.data());
    {
        py::gil_scoped_release release;
        craftground::NamedSemaphore(layout->sem_obs_ready.name).waitFor(timeout_ms);
    }
    if (!server_event.is_none() && server_event.attr("is_set")().cast<bool>()) {
        throw std::runtime_error(
            "Minecraft exited while waiting for shared memory"
        );
    }
    if (!layout->initial_environment_offset)
        throw std::runtime_error("Minecraft failed to write the shared-memory frame");
    craftground::ShmMapping head(
        j2p_memory_name.c_str(),
        sizeof(J2PSharedMemoryLayout),
        craftground::ShmAccess::Read
    );
    auto *header = reinterpret_cast<const J2PSharedMemoryLayout *>(head.data());
    if (header->data_offset != sizeof(*header) ||
        header->data_size > INT32_MAX) {
        throw std::runtime_error("Invalid shared-memory message header");
    }
    auto copy_frame = [&]() {
        craftground::ShmMapping frame(
            j2p_memory_name.c_str(), sizeof(*header) + header->data_size,
            craftground::ShmAccess::Read
        );
        return py::bytes(frame.data() + sizeof(*header), header->data_size);
    };
    if (!header->layout_size)
        return copy_frame();
    craftground::NamedSemaphore slot(j2p_memory_name + "f");
    py::bytes data;
    try {
        data = copy_frame();
    } catch (...) {
        slot.post();
        throw;
    }
    slot.post();
    return data;
#else
    int p2jFd = get_shared_memory_fd(p2j_memory_name.c_str());
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
        return py::bytes();
    }
    SharedMemoryLayout *layout = static_cast<SharedMemoryLayout *>(p2jPtr);
    size_t action_size = layout->action_size;
    size_t data_size = layout->initial_environment_size;

    // std::cout << "Waiting for java to write observation" << std::endl;

    // Wait for the observation to be ready
    rk_sema_open(&layout->sem_obs_ready);
    // rk_sema_open(&layout->sem_action_ready);
    // rk_sema_post(&layout->sem_action_ready);
    {
        py::gil_scoped_release release;
        rk_sema_wait(&layout->sem_obs_ready);
    }

    int j2pFd;
    j2pFd = shm_open_existing(j2p_memory_name.c_str(), 0666);
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
        close(j2pFd);
        return py::bytes();
    }

    j2pLayout = static_cast<J2PSharedMemoryLayout *>(j2pPtr);
    const char *data_start = (char *)j2pPtr + j2pLayout->data_offset;

    py::bytes data(data_start, j2pLayout->data_size);

    // std::cout << "Read observation from shared memory. Notified to java read
    // "
    //              "finish observation"
    //           << std::endl;
    munmap(j2pPtr, sizeof(J2PSharedMemoryLayout) + obs_length);
    munmap(p2jPtr, sizeof(SharedMemoryLayout));
    close(j2pFd);
    return data;
#endif
}

#ifndef _WIN32
void notify_shared_memory_exit(const std::string &p2j_memory_name) {
    craftground::ShmMapping control(
        p2j_memory_name.c_str(),
        sizeof(SharedMemoryLayout),
        craftground::ShmAccess::Read
    );
    auto *layout = reinterpret_cast<const SharedMemoryLayout *>(control.data());
    craftground::NamedSemaphore(layout->sem_obs_ready.name).post();
}
#endif

// remove shared memory
void destroy_shared_memory_impl(
    const std::string &memory_name, bool release_semaphores
) {
    if (release_semaphores) {
        int p2jFd = get_shared_memory_fd(memory_name.c_str());
        if (p2jFd == -1) {
            perror("shm_open failed while destroying shared memory");
            return;
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
            close_shared_memory_fd(memory_name);
        }
    }
#ifndef _WIN32
    if (release_semaphores)
        shm_unlink((memory_name + "a").c_str());
    else
        sem_unlink((memory_name + "f").c_str());
#endif
    shm_unlink(memory_name.c_str());
}
