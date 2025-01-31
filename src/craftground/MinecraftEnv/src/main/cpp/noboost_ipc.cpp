#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <jni.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(WIN32) || defined(_WIN32) ||                                       \
    defined(__WIN32) && !defined(__CYGWIN__)
#define IS_WINDOWS 1
#define SHMEM_PREFIX "Global\\"
#else
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
    // std::cout << "Reading initial environment from shared memory 1"
    //           << std::endl;
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
    SharedMemoryLayout *p2jLayout = static_cast<SharedMemoryLayout *>(p2jPtr);

    const size_t initial_environment_size = p2jLayout->initial_environment_size;
    const size_t action_size = p2jLayout->action_size;
    // std::cout << "Reading initial environment from shared memory 2"
    //           << std::endl;

    munmap(p2jPtr, sizeof(SharedMemoryLayout));
    // Note: action_size is 0 when dummy is provided; actually python overwrites
    // on initial environment section.
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

    p2jLayout = static_cast<SharedMemoryLayout *>(p2jPtr);

    char *data_startInitialEnvironment =
        static_cast<char *>(p2jPtr) + p2jLayout->initial_environment_offset;
    size_t data_size = p2jLayout->initial_environment_size;

    // std::cout << "Java read data_size: " <<
    // p2jLayout->initial_environment_size
    //           << std::endl;
    // std::cout << "Java initial environment offset:"
    //           << p2jLayout->initial_environment_offset << std::endl;
    // std::cout << "Java layout size:" << p2jLayout->layout_size << std::endl;
    // std::cout << "Java action offset:" << p2jLayout->action_offset <<
    // std::endl; std::cout << "Java action size:" << p2jLayout->action_size <<
    // std::endl; std::cout << "Java read data_size: " << data_size <<
    // std::endl;

    jbyteArray byteArray = env->NewByteArray(data_size);
    if (byteArray == nullptr || env->ExceptionCheck()) {
        munmap(
            p2jPtr,
            sizeof(SharedMemoryLayout) + action_size + initial_environment_size
        );
        close(p2jFd);
        return nullptr;
    }
    // std::cout << "Java read array: ";
    // printHex(data_startInitialEnvironment, data_size);
    env->SetByteArrayRegion(
        byteArray,
        0,
        data_size,
        reinterpret_cast<jbyte *>(data_startInitialEnvironment)
    );

    std::string sema_obs_ready_name =
        SHMEM_PREFIX "cg_sem_obs" + std::to_string(port);
    std::string sema_action_ready_name =
        SHMEM_PREFIX "cg_sem_act" + std::to_string(port);

    rk_sema_init(&p2jLayout->sem_obs_ready, sema_obs_ready_name.c_str(), 0, 1);
    rk_sema_init(
        &p2jLayout->sem_action_ready, sema_action_ready_name.c_str(), 0, 1
    );
    // std::cout << "Initialied semaphore for Java"
    //           << p2jLayout->sem_obs_ready.name << std::endl;
    // std::cout << "Initialied semaphore for Java"
    //           << p2jLayout->sem_action_ready.name << std::endl;
    return byteArray;
}

jbyteArray read_action(
    JNIEnv *env,
    jclass clazz,
    const std::string &p2j_memory_name,
    jbyteArray data
) {
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

    SharedMemoryLayout *p2jHeader = static_cast<SharedMemoryLayout *>(p2jPtr);
    size_t action_size = p2jHeader->action_size;

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
    p2jHeader = static_cast<SharedMemoryLayout *>(p2jPtr);

    char *data_start = static_cast<char *>(p2jPtr) + p2jHeader->action_offset;

    // std::cout << "Waiting for Python to write the action" << std::endl;
    // printHex((const char *)p2jHeader, sizeof(SharedMemoryLayout));
    // OK
    rk_sema_open(&p2jHeader->sem_action_ready);
    rk_sema_wait(&p2jHeader->sem_action_ready);
    if (data != nullptr) {
        jsize oldSize = env->GetArrayLength(data);
        if (oldSize != p2jHeader->action_size) {
            // std::cout << "Resizing byte array to"
            //           << std::to_string(p2jHeader->action_size) << std::endl;
            env->DeleteLocalRef(data);
            data = nullptr;
            data = env->NewByteArray(p2jHeader->action_size);
        }
    } else {
        // std::cout << "Creating new byte array"
        //           << std::to_string(p2jHeader->action_size) << std::endl;
        data = env->NewByteArray(p2jHeader->action_size);
    }

    if (data == nullptr || env->ExceptionCheck()) {
        munmap(p2jPtr, sizeof(SharedMemoryLayout) + action_size);
        close(p2jFd);
        return nullptr;
    }
    // Read the action message
    if (action_size > 0) {
        env->SetByteArrayRegion(
            data,
            0,
            p2jHeader->action_size,
            reinterpret_cast<jbyte *>(data_start)
        );
    }
    // std::cout << "Read action from shared memory" << std::endl;
    return data;
}

void write_observation(
    const char *p2j_memory_name,
    const char *j2p_memory_name,
    const char *data,
    const size_t observation_size
) {
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
    SharedMemoryLayout *p2jLayout = static_cast<SharedMemoryLayout *>(p2jPtr);

    int j2pFd = shm_open(j2p_memory_name, O_RDWR, 0666);
    if (j2pFd == -1) {
        perror("shm_open j2p failed while writing to shared memory");
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
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
    // std::cout << "Writing observation to shared memory adfsadf" << std::endl;
    J2PSharedMemoryLayout *j2pLayout =
        static_cast<J2PSharedMemoryLayout *>(j2pPtr);
    j2pLayout->data_offset = sizeof(J2PSharedMemoryLayout);

    // std::cout << "Writing observation to shared memory adfsad222sdsfsasdff"
    //           << std::endl;

    struct stat statbuf;
    if (fstat(j2pFd, &statbuf) == -1) {
        perror("fstat failed while getting shared memory size");
        munmap(j2pPtr, sizeof(J2PSharedMemoryLayout));
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
        close(j2pFd);
        close(p2jFd);
        return;
    } else {
        // std::cout << "Shared memory size: " << statbuf.st_size << std::endl;
    }
    const size_t current_shmem_size = statbuf.st_size;
    size_t requiredSize = observation_size + sizeof(J2PSharedMemoryLayout);
    requiredSize = requiredSize > 1024 ? requiredSize : 1024;

    if (current_shmem_size < requiredSize) {
        // Unmap existing memory before resizing
        munmap(j2pPtr, sizeof(J2PSharedMemoryLayout));

        // Resize the shared memory
        if (ftruncate(j2pFd, requiredSize) == -1) {
            perror("ftruncate failed while resizing shared memory");
            munmap(p2jPtr, sizeof(SharedMemoryLayout));
            close(j2pFd);
            close(p2jFd);
            return;
        }

        // Remap with new size
        j2pPtr =
            mmap(0, requiredSize, PROT_READ | PROT_WRITE, MAP_SHARED, j2pFd, 0);
        if (j2pPtr == MAP_FAILED) {
            perror("mmap failed after resizing shared memory");
            munmap(p2jPtr, sizeof(SharedMemoryLayout));
            close(j2pFd);
            close(p2jFd);
            return;
        }

        // Initialize the header
        j2pLayout = static_cast<J2PSharedMemoryLayout *>(j2pPtr);
        j2pLayout->layout_size = sizeof(J2PSharedMemoryLayout);
        j2pLayout->data_offset = sizeof(J2PSharedMemoryLayout);
        j2pLayout->data_size = observation_size;
        // std::cout << "Resized shared memory to " << requiredSize <<
        // std::endl;
    }
    // std::cout << "Writing observation to shared memory KKKAKAAKAK" <<
    // std::endl;

    // Write the observation to shared memory
    char *data_start = static_cast<char *>(j2pPtr) + j2pLayout->data_offset;
    std::memcpy(data_start, data, observation_size);
    j2pLayout->data_size = observation_size;

    // Notify Python that the observation is ready
    // printHex((const char *)p2jLayout, sizeof(SharedMemoryLayout));
    rk_sema_open(&p2jLayout->sem_obs_ready);
    if (rk_sema_post(&p2jLayout->sem_obs_ready) < 0) {
        perror("rk_sema_post failed while notifying python");
    }
    // std::cout << "Wrote and notified observation to python" << std::endl;

    // Clean up resources
    munmap(j2pPtr, requiredSize);
    munmap(p2jPtr, sizeof(SharedMemoryLayout));
    close(j2pFd);
    close(p2jFd);
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