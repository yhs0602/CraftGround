// POSIX shared-memory objects on macOS can only be sized once. The producer
// replaces an undersized object while it owns the slot; the consumer opens
// the name after the ready semaphore is posted.
#pragma once
#include <cerrno>
#include <fcntl.h>
#include <semaphore.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace craftground {
inline void shm_error(const char *operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}

enum class ShmAccess { Read, Write, Grow };

class ShmMapping {
    void *address;
    size_t length;

  public:
    ShmMapping(const char *name, size_t size, ShmAccess access) : length(size) {
        const bool write = access != ShmAccess::Read;
        int fd = shm_open(name, write ? O_RDWR : O_RDONLY, 0600);
        struct stat st{};
        if (fd >= 0) {
            if (fstat(fd, &st) < 0) {
                int error = errno;
                close(fd);
                throw std::system_error(
                    error, std::generic_category(), "fstat"
                );
            }
            if (static_cast<size_t>(st.st_size) < size) {
                close(fd);
                if (access != ShmAccess::Grow)
                    throw std::runtime_error("Incomplete shared-memory message"
                    );
                if (shm_unlink(name) < 0)
                    shm_error("shm_unlink");
                fd = -1;
            }
        } else if (access != ShmAccess::Grow || errno != ENOENT) {
            shm_error("shm_open");
        }
        if (fd < 0) {
            fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
            if (fd < 0)
                shm_error("shm_open");
            if (ftruncate(fd, size) < 0) {
                int error = errno;
                close(fd);
                shm_unlink(name);
                throw std::system_error(
                    error, std::generic_category(), "ftruncate"
                );
            }
        }
        address = mmap(
            nullptr,
            size,
            write ? PROT_READ | PROT_WRITE : PROT_READ,
            MAP_SHARED,
            fd,
            0
        );
        int error = errno;
        close(fd);
        if (address == MAP_FAILED)
            throw std::system_error(error, std::generic_category(), "mmap");
    }
    ShmMapping(const ShmMapping &) = delete;
    ~ShmMapping() { munmap(address, length); }
    char *data() const { return static_cast<char *>(address); }
};

class NamedSemaphore {
    sem_t *sem;

  public:
    explicit NamedSemaphore(const std::string &name, bool create = false) {
        sem = create ? sem_open(name.c_str(), O_CREAT | O_EXCL, 0600, 1)
                     : sem_open(name.c_str(), 0);
        if (sem == SEM_FAILED)
            shm_error("sem_open");
    }
    NamedSemaphore(const NamedSemaphore &) = delete;
    ~NamedSemaphore() { sem_close(sem); }
    void wait() {
        while (sem_wait(sem) < 0) {
            if (errno != EINTR)
                shm_error("sem_wait");
        }
    }
    void post() {
        if (sem_post(sem) < 0)
            shm_error("sem_post");
    }
};
} // namespace craftground
