#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <jni.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#if defined(WIN32) || defined(_WIN32) ||                                       \
    defined(__WIN32) && !defined(__CYGWIN__)
#define IS_WINDOWS 1
#define SHMEM_PREFIX "Global\\"
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#define SHMEM_PREFIX "/"
#define IS_WINDOWS 0
#endif
#include "cross_semaphore.h"

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

void printHex(const char *data, size_t data_size) {
    for (size_t i = 0; i < data_size; ++i) {
        // Print the hexadecimal representation of the byte
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << (static_cast<unsigned int>(data[i]) & 0xFF) << " ";

        // Print a newline every 16 bytes
        if ((i + 1) % 16 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::dec << std::endl; // Reset the output format
}

std::string make_shared_memory_name(int port, const std::string &suffix) {
    return SHMEM_PREFIX "craftground_" + std::to_string(port) + "_" + suffix;
}

// Returns ByteArray object containing the initial environment message
jobject read_initial_environment(
    JNIEnv *env, jclass clazz, const std::string &p2j_memory_name, int port
) {
#ifdef _WIN32
    HANDLE hMapFile =
        OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, p2j_memory_name.c_str());
    if (!hMapFile) {
        std::cerr << "OpenFileMapping failed: " << GetLastError() << std::endl;
        return nullptr;
    }

    void *p2jPtr = MapViewOfFile(
        hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedMemoryLayout)
    );
    if (!p2jPtr) {
        std::cerr << "MapViewOfFile failed: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return nullptr;
    }
#else
    int p2jFd = shm_open(p2j_memory_name.c_str(), O_RDWR, 0666);
    if (p2jFd == -1) {
        perror("shm_open p2j failed while reading from shared memory");
        return nullptr;
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
        return nullptr;
    }
#endif

    SharedMemoryLayout *p2jLayout = static_cast<SharedMemoryLayout *>(p2jPtr);
    const size_t initial_environment_size = p2jLayout->initial_environment_size;
    const size_t action_size = p2jLayout->action_size;

#ifdef _WIN32
    UnmapViewOfFile(p2jPtr);
    p2jPtr = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(SharedMemoryLayout) + action_size + initial_environment_size
    );
    if (!p2jPtr) {
        std::cerr << "MapViewOfFile failed on second mapping: "
                  << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return nullptr;
    }
#else
    munmap(p2jPtr, sizeof(SharedMemoryLayout));
    p2jPtr = mmap(
        0,
        sizeof(SharedMemoryLayout) + action_size + initial_environment_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        p2jFd,
        0
    );
    if (p2jPtr == MAP_FAILED) {
        perror("mmap p2j failed while reading from shared memory");
        close(p2jFd);
        return nullptr;
    }
#endif

    p2jLayout = static_cast<SharedMemoryLayout *>(p2jPtr);
    char *data_startInitialEnvironment =
        static_cast<char *>(p2jPtr) + p2jLayout->initial_environment_offset;
    size_t data_size = p2jLayout->initial_environment_size;

    jbyteArray byteArray = env->NewByteArray(data_size);
    if (byteArray == nullptr || env->ExceptionCheck()) {
#ifdef _WIN32
        UnmapViewOfFile(p2jPtr);
        CloseHandle(hMapFile);
#else
        munmap(
            p2jPtr,
            sizeof(SharedMemoryLayout) + action_size + initial_environment_size
        );
        close(p2jFd);
#endif
        return nullptr;
    }

    env->SetByteArrayRegion(
        byteArray,
        0,
        data_size,
        reinterpret_cast<jbyte *>(data_startInitialEnvironment)
    );

#ifdef _WIN32
    UnmapViewOfFile(p2jPtr);
    CloseHandle(hMapFile);
#else
    munmap(
        p2jPtr,
        sizeof(SharedMemoryLayout) + action_size + initial_environment_size
    );
    close(p2jFd);
#endif

    return byteArray;
}

jbyteArray read_action(
    JNIEnv *env,
    jclass clazz,
    const std::string &p2j_memory_name,
    jbyteArray data
) {
#ifdef _WIN32
    HANDLE hMapFile =
        OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, p2j_memory_name.c_str());
    if (!hMapFile) {
        std::cerr << "OpenFileMapping failed: " << GetLastError() << std::endl;
        return nullptr;
    }

    void *p2jPtr = MapViewOfFile(
        hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedMemoryLayout)
    );
    if (!p2jPtr) {
        std::cerr << "MapViewOfFile failed: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return nullptr;
    }
#else
    int p2jFd = shm_open(p2j_memory_name.c_str(), O_RDWR, 0666);
    if (p2jFd == -1) {
        perror("shm_open p2j failed while reading from shared memory");
        return nullptr;
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
        return nullptr;
    }
#endif

    SharedMemoryLayout *p2jHeader = static_cast<SharedMemoryLayout *>(p2jPtr);
    size_t action_size = p2jHeader->action_size;

#ifdef _WIN32
    UnmapViewOfFile(p2jPtr);
    p2jPtr = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(SharedMemoryLayout) + action_size
    );
    if (!p2jPtr) {
        std::cerr << "MapViewOfFile failed on second mapping: "
                  << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return nullptr;
    }
#else
    munmap(p2jPtr, sizeof(SharedMemoryLayout));
    p2jPtr = mmap(
        0,
        sizeof(SharedMemoryLayout) + action_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        p2jFd,
        0
    );
    if (p2jPtr == MAP_FAILED) {
        perror("mmap p2j failed while reading from shared memory");
        close(p2jFd);
        return nullptr;
    }
#endif

    p2jHeader = static_cast<SharedMemoryLayout *>(p2jPtr);
    char *data_start = static_cast<char *>(p2jPtr) + p2jHeader->action_offset;

    if (data != nullptr) {
        jsize oldSize = env->GetArrayLength(data);
        if (oldSize != p2jHeader->action_size) {
            env->DeleteLocalRef(data);
            data = nullptr;
            data = env->NewByteArray(p2jHeader->action_size);
        }
    } else {
        data = env->NewByteArray(p2jHeader->action_size);
    }

    if (data == nullptr || env->ExceptionCheck()) {
#ifdef _WIN32
        UnmapViewOfFile(p2jPtr);
        CloseHandle(hMapFile);
#else
        munmap(p2jPtr, sizeof(SharedMemoryLayout) + action_size);
        close(p2jFd);
#endif
        return nullptr;
    }

    if (action_size > 0) {
        env->SetByteArrayRegion(
            data,
            0,
            p2jHeader->action_size,
            reinterpret_cast<jbyte *>(data_start)
        );
    }

#ifdef _WIN32
    UnmapViewOfFile(p2jPtr);
    CloseHandle(hMapFile);
#else
    munmap(p2jPtr, sizeof(SharedMemoryLayout) + action_size);
    close(p2jFd);
#endif
    return data;
}

void write_observation(
    const char *p2j_memory_name,
    const char *j2p_memory_name,
    const char *data,
    const size_t observation_size
) {
#ifdef _WIN32
    HANDLE p2jFd =
        OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, p2j_memory_name);
    if (!p2jFd) {
        std::cerr << "OpenFileMapping failed for p2j: " << GetLastError()
                  << std::endl;
        return;
    }
    void *p2jPtr = MapViewOfFile(
        p2jFd, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedMemoryLayout)
    );
    if (!p2jPtr) {
        std::cerr << "MapViewOfFile failed for p2j: " << GetLastError()
                  << std::endl;
        CloseHandle(p2jFd);
        return;
    }
#else
    int p2jFd = shm_open(p2j_memory_name, O_RDWR, 0666);
    if (p2jFd == -1) {
        perror("shm_open p2j failed while writing to shared memory");
        return;
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
        perror("mmap p2j failed while writing to shared memory");
        close(p2jFd);
        return;
    }
#endif
    SharedMemoryLayout *p2jLayout = static_cast<SharedMemoryLayout *>(p2jPtr);

#ifdef _WIN32
    HANDLE j2pFd =
        OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, j2p_memory_name);
    if (!j2pFd) {
        std::cerr << "OpenFileMapping failed for j2p: " << GetLastError()
                  << std::endl;
        UnmapViewOfFile(p2jPtr);
        CloseHandle(p2jFd);
        return;
    }
    void *j2pPtr = MapViewOfFile(
        j2pFd, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(J2PSharedMemoryLayout)
    );
    if (!j2pPtr) {
        std::cerr << "MapViewOfFile failed for j2p: " << GetLastError()
                  << std::endl;
        UnmapViewOfFile(p2jPtr);
        CloseHandle(p2jFd);
        CloseHandle(j2pFd);
        return;
    }
#else
    int j2pFd = shm_open(j2p_memory_name, O_RDWR, 0666);
    if (j2pFd == -1) {
        perror("shm_open j2p failed while writing to shared memory");
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
        close(p2jFd);
        return;
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
        perror("mmap j2p failed while writing to shared memory");
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
        close(p2jFd);
        close(j2pFd);
        return;
    }
#endif
    J2PSharedMemoryLayout *j2pLayout =
        static_cast<J2PSharedMemoryLayout *>(j2pPtr);
    j2pLayout->data_offset = sizeof(J2PSharedMemoryLayout);

    char *data_start = static_cast<char *>(j2pPtr) + j2pLayout->data_offset;
    std::memcpy(data_start, data, observation_size);
    j2pLayout->data_size = observation_size;

    rk_sema_open(&p2jLayout->sem_obs_ready);
    if (rk_sema_post(&p2jLayout->sem_obs_ready) < 0) {
        perror("rk_sema_post failed while notifying python");
    }

#ifdef _WIN32
    UnmapViewOfFile(j2pPtr);
    UnmapViewOfFile(p2jPtr);
    CloseHandle(j2pFd);
    CloseHandle(p2jFd);
#else
    munmap(j2pPtr, sizeof(J2PSharedMemoryLayout));
    munmap(p2jPtr, sizeof(SharedMemoryLayout));
    close(j2pFd);
    close(p2jFd);
#endif
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_readInitialEnvironmentImpl(
    JNIEnv *env, jclass clazz, jstring p2j_memory_name, jint port
) {
    const char *p2j_memory_name_cstr =
        env->GetStringUTFChars(p2j_memory_name, nullptr);
    jobject result = nullptr;
    try {
        result =
            read_initial_environment(env, clazz, p2j_memory_name_cstr, port);
    } catch (const std::exception &e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
        return nullptr;
    }
    env->ReleaseStringUTFChars(p2j_memory_name, p2j_memory_name_cstr);
    return result;
}

// fun readAction(action_memory_name: String, action_data: ByteArray?):
// ByteArray
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_readActionImpl(
    JNIEnv *env, jclass clazz, jstring p2j_memory_name, jbyteArray action_data
) {
    const char *j2p_memory_name_cstr =
        env->GetStringUTFChars(p2j_memory_name, nullptr);

    size_t data_size = 0;
    jbyteArray data;
    try {
        data = read_action(env, clazz, j2p_memory_name_cstr, action_data);
    } catch (const std::exception &e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
        return nullptr;
    }
    env->ReleaseStringUTFChars(p2j_memory_name, j2p_memory_name_cstr);
    return data;
}

// fun writeObservation(
//     observation_memory_name: String,
//     synchronization_memory_name: String,
//     observation_data: ByteArray
// )
extern "C" JNIEXPORT void JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_writeObservationImpl(
    JNIEnv *env,
    jclass clazz,
    jstring p2j_memory_name,
    jstring j2p_memory_name,
    jbyteArray observation_data
) {
    const char *j2p_memory_name_cstr =
        env->GetStringUTFChars(j2p_memory_name, nullptr);
    const char *p2j_memory_name_cstr =
        env->GetStringUTFChars(p2j_memory_name, nullptr);

    if (j2p_memory_name_cstr == nullptr || p2j_memory_name_cstr == nullptr) {
        std::cerr << "Failed to get memory name: " << j2p_memory_name_cstr
                  << ";" << p2j_memory_name_cstr << std::endl;
        return;
    }
    jbyte *observation_data_ptr =
        env->GetByteArrayElements(observation_data, nullptr);

    if (observation_data_ptr == nullptr) {
        std::cerr << "Failed to get observation data" << std::endl;
        return;
    }
    jsize observation_data_size = env->GetArrayLength(observation_data);
    // std::cout << "Writing observation to shared memory with length"
    //           << std::to_string(observation_data_size) << std::endl;

    try {
        write_observation(
            p2j_memory_name_cstr,
            j2p_memory_name_cstr,
            reinterpret_cast<const char *>(observation_data_ptr),
            static_cast<size_t>(observation_data_size)
        );
    } catch (const std::exception &e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }

    env->ReleaseStringUTFChars(j2p_memory_name, j2p_memory_name_cstr);
    env->ReleaseStringUTFChars(p2j_memory_name, p2j_memory_name_cstr);
    env->ReleaseByteArrayElements(
        observation_data, observation_data_ptr, JNI_ABORT
    );
}