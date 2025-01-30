#pragma once

// https://stackoverflow.com/a/27847103/8614565
#include <cerrno>
#include <cstdint>
#include <cstdio>
#if IS_WINDOWS
#include <windows.h>
#else
#include <semaphore.h>
#endif

struct rk_sema {
#if IS_WINDOWS
    HANDLE sem;
#else
    sem_t *sem;
    char name[30]; // Save the name of the semaphore
#endif
};

static inline void rk_sema_init(
    struct rk_sema *s, const char *name, uint32_t value, uint32_t max
) {
#if IS_WINDOWS
    s->sem = CreateSemaphore(NULL, value, max, name);
#else
    snprintf(s->name, sizeof(s->name), "/%s", name);
    sem_unlink(s->name); // Remove any existing semaphore with the same name
    s->sem = sem_open(s->name, O_CREAT, 0644, value); // Binary semaphore
    if (s->sem == SEM_FAILED) {
        perror("sem_open failed");
        return;
    }
#endif
}

static inline void rk_sema_wait(struct rk_sema *s) {

#if IS_WINDOWS
    DWORD r;
    do {
        r = WaitForSingleObject(s->sem, INFINITE);
    } while (r == WAIT_FAILED && GetLastError() == ERROR_INTERRUPT);
#else
    int r;

    do {
        r = sem_wait(s->sem);
    } while (r == -1 && errno == EINTR);
#endif
}

static inline void rk_sema_post(struct rk_sema *s) {

#if IS_WINDOWS
    ReleaseSemaphore(s->sem, 1, NULL);
#else
    sem_post(s->sem);
#endif
}

static inline void rk_sema_destroy(struct rk_sema *s) {
#if IS_WINDOWS
    CloseHandle(s->sem);
#else
    sem_close(s->sem);
    sem_unlink(s->name); // Named semaphores are removed after unlinking
#endif
}