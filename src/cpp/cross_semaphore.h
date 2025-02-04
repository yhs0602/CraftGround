#pragma once

// https://stackoverflow.com/a/27847103/8614565
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <thread>
#include <fcntl.h>
#if IS_WINDOWS
#include <windows.h>
#else
#include <semaphore.h>
#endif

struct rk_sema {
#if IS_WINDOWS
    HANDLE sem_python;
    HANDLE sem_java;
#else
    sem_t *sem_python;
    sem_t *sem_java;
#endif
    char name[30]; // Save the name of the semaphore
};

static inline void rk_sema_init(
    struct rk_sema *s, const char *name, uint32_t value, uint32_t max
) {
#if IS_WINDOWS
    s->sem_python = CreateSemaphore(NULL, value, max, name);
#else
    snprintf(s->name, sizeof(s->name), "%s", name);
    sem_unlink(s->name); // Remove any existing semaphore with the same name
    s->sem_python = sem_open(s->name, O_CREAT, 0644, value); // Binary semaphore
    if (s->sem_python == SEM_FAILED) {
        perror("sem_open failed in create");
        return;
    }
#endif
}

static inline void rk_sema_open(struct rk_sema *s) {
#if IS_WINDOWS
    s->sem_python = OpenSemaphoreA(
        SEMAPHORE_ALL_ACCESS, FALSE, s->name
    ); // Open existing semaphore
    if (s->sem_python == NULL) {
        std::cerr << "OpenSemaphore failed: " << GetLastError() << std::endl;
    }
#else
    if (s->sem_python != nullptr) {
        return;
    }
    s->sem_python = sem_open(s->name, 0); // Open existing semaphore
    if (s->sem_python == SEM_FAILED) {
        std::cout << "sem_open failed in python open" << s->name << std::endl;
        perror("sem_open failed in open");
        return;
    }
#endif
}

static inline int rk_sema_wait(struct rk_sema *s) {
#if IS_WINDOWS
    DWORD r;
    do {
        r = WaitForSingleObject(s->sem_python, INFINITE);
    } while (r == WAIT_FAILED && GetLastError() == ERROR_IO_PENDING
    ); // 적절한 오류 코드로 변경
    return r;
#else
    int r;
    do {
        r = sem_wait(s->sem_python);
    } while (r == -1 && errno == EINTR);
    return r;
#endif
}

static inline int rk_sema_post(struct rk_sema *s) {

#if IS_WINDOWS
    return ReleaseSemaphore(s->sem_python, 1, NULL);
#else
    return sem_post(s->sem_python);
#endif
}

static inline void async_rk_sema_post(struct rk_sema *s) {
    std::thread([s]() {
        if (rk_sema_post(s) < 0) {
            perror("Failed to post semaphore");
        };
    }).detach();
}

static inline void rk_sema_destroy(struct rk_sema *s) {
#if IS_WINDOWS
    CloseHandle(s->sem_python);
    CloseHandle(s->sem_java);
#else
    sem_close(s->sem_python);
    sem_close(s->sem_java);
    sem_unlink(s->name); // Named semaphores are removed after unlinking
#endif
}