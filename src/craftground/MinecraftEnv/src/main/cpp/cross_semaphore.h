#pragma once

// https://stackoverflow.com/a/27847103/8614565
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <iostream>
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
    char name[30]; // Save the name of the semaphore
#endif
};

static inline void rk_sema_init(
    struct rk_sema *s, const char *name, uint32_t value, uint32_t max
) {
#if IS_WINDOWS
    // TODO: Open the semaphore with the same name
    s->sem_java = CreateSemaphore(NULL, value, max, name);
#else
    snprintf(s->name, sizeof(s->name), "%s", name);
    s->sem_java = sem_open(s->name, O_CREAT); // Binary semaphore
    if (s->sem_java == SEM_FAILED) {
        perror("sem_open failed in java init");
        return;
    }
#endif
}

static inline void rk_sema_open(struct rk_sema *s) {
#if IS_WINDOWS
    s->sem_java = OpenSemaphore(SEMAPHORE_ALL_ACCESS, FALSE, s->name);
#else
    s->sem_java = sem_open(s->name, 0); // Open existing semaphore
    if (s->sem_java == SEM_FAILED) {
        std::cout << "sem_open failed in java open" << s->name << std::endl;
        perror("sem_open failed in java open");
        return;
    }
#endif
}

static inline void rk_sema_wait(struct rk_sema *s) {

#if IS_WINDOWS
    DWORD r;
    do {
        r = WaitForSingleObject(s->sem_java, INFINITE);
    } while (r == WAIT_FAILED && GetLastError() == ERROR_INTERRUPT);
#else
    int r;

    do {
        r = sem_wait(s->sem_java);
    } while (r == -1 && errno == EINTR);
#endif
}

static inline int rk_sema_post(struct rk_sema *s) {

#if IS_WINDOWS
    return ReleaseSemaphore(s->sem_java, 1, NULL);
#else
    return sem_post(s->sem_java);
#endif
}

static inline void rk_sema_destroy(struct rk_sema *s) {
#if IS_WINDOWS
    CloseHandle(s->sem);
#else
    sem_close(s->sem_java);
    sem_close(s->sem_python);
    sem_unlink(s->name); // Named semaphores are removed after unlinking
#endif
}