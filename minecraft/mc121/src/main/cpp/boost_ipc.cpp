#include "boost/interprocess/detail/os_file_functions.hpp"
#include "boost/interprocess/mapped_region.hpp"
#include "boost/interprocess/shared_memory_object.hpp"
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <jni.h>
#include <mutex>
#include <iostream>

using namespace boost::interprocess;

struct SharedMemoryLayout {
    size_t layout_size;                // to be set on initialization
    size_t action_offset;              // Always sizeof(SharedMemoryLayout)
    size_t action_size;                // to be set on initialization
    size_t initial_environment_offset; // Always action_size +
                                       // sizeof(SharedDataHeader)
    size_t initial_environment_size;   // to be set on initialization
    interprocess_mutex mutex;
    interprocess_condition condition;
    bool p2j_ready;
    bool j2p_ready;
    bool p2j_recv_ready;
    bool j2p_recv_ready;
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

// Returns ByteArray object containing the initial environment message
jobject read_initial_environment(
    JNIEnv *env, jclass clazz, const std::string &p2j_memory_name
) {
    shared_memory_object p2jMemory(
        open_only, p2j_memory_name.c_str(), read_only
    );
    mapped_region p2jRegion(p2jMemory, read_only);
    SharedMemoryLayout *p2jLayout =
        reinterpret_cast<SharedMemoryLayout *>(p2jRegion.get_address());
    char *data_startInitialEnvironment = reinterpret_cast<char *>(p2jLayout) +
                                         p2jLayout->initial_environment_offset;
    size_t data_size = p2jLayout->initial_environment_size;

    std::cout << "Java read data_size: " << p2jLayout->initial_environment_size
              << std::endl;
    std::cout << "Java initial environment offset:"
              << p2jLayout->initial_environment_offset << std::endl;
    std::cout << "Java layout size:" << p2jLayout->layout_size << std::endl;
    std::cout << "Java action offset:" << p2jLayout->action_offset << std::endl;
    std::cout << "Java action size:" << p2jLayout->action_size << std::endl;
    std::cout << "Java p2j ready:" << p2jLayout->j2p_ready << std::endl;
    std::cout << "Java j2p ready:" << p2jLayout->p2j_ready << std::endl;
    std::cout << "Java read data_size: " << data_size << std::endl;

    jbyteArray byteArray = env->NewByteArray(data_size);
    if (byteArray == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    std::cout << "Java read array: ";
    printHex(data_startInitialEnvironment, data_size);
    env->SetByteArrayRegion(
        byteArray,
        0,
        data_size,
        reinterpret_cast<jbyte *>(data_startInitialEnvironment)
    );
    return byteArray;
}

jbyteArray read_action(
    JNIEnv *env,
    jclass clazz,
    const std::string &p2j_memory_name,
    jbyteArray data
) {
    std::cout << "Reading action from shared memory 1" << std::endl;
    shared_memory_object p2jMemory(
        open_only, p2j_memory_name.c_str(), read_write
    );
    mapped_region p2jRegion(p2jMemory, read_write);
    SharedMemoryLayout *p2jHeader =
        static_cast<SharedMemoryLayout *>(p2jRegion.get_address());
    char *data_start =
        reinterpret_cast<char *>(p2jHeader) + p2jHeader->action_offset;

    std::unique_lock<interprocess_mutex> actionLock(p2jHeader->mutex);
    p2jHeader->p2j_recv_ready = true;
    p2jHeader->condition.notify_all();
    actionLock.unlock();
    actionLock.lock();
    std::cout << "Reading action from shared memory: Acquired Lock"
              << std::endl;
    p2jHeader->condition.wait(actionLock, [&] { return p2jHeader->p2j_ready; });
    p2jHeader->p2j_ready = false;

    if (data == nullptr) {
        data = env->NewByteArray(p2jHeader->action_size);
    }
    if (data == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    // Read the action message
    env->SetByteArrayRegion(
        data, 0, p2jHeader->action_size, reinterpret_cast<jbyte *>(data_start)
    );
    p2jHeader->p2j_ready = false;
    p2jHeader->j2p_ready = false;
    p2jHeader->p2j_recv_ready = false;
    p2jHeader->j2p_recv_ready = false;
    actionLock.unlock();
    std::cout << "Read action from shared memory 2" << std::endl;
    return data;
}

void write_observation(
    const std::string &p2j_memory_name,
    const std::string &j2p_memory_name,
    const char *data,
    const size_t observation_size
) {
    std::cout << "Writing observation to shared memory 1" << std::endl;
    shared_memory_object p2jMemory(
        open_only, p2j_memory_name.c_str(), read_write
    );
    mapped_region p2jRegion(p2jMemory, read_write);
    SharedMemoryLayout *p2jLayout =
        static_cast<SharedMemoryLayout *>(p2jRegion.get_address());

    std::unique_lock<interprocess_mutex> lockSynchronization(p2jLayout->mutex);
    p2jLayout->condition.wait(lockSynchronization, [&] {
        return p2jLayout->j2p_recv_ready;
    });
    p2jLayout->j2p_recv_ready = false;

    shared_memory_object j2pMemory(
        open_only, j2p_memory_name.c_str(), read_write
    );
    mapped_region j2pMemoryRegion(j2pMemory, read_write);
    // Resize the shared memory if needed
    offset_t size = 0;
    const size_t currentSize = j2pMemory.get_size(size);
    size_t requiredSize = observation_size + sizeof(J2PSharedMemoryLayout);
    if (currentSize < requiredSize) {
        try {
            shared_memory_object::remove(j2p_memory_name.c_str());
        } catch (const interprocess_exception &e) {
            std::cerr << e.what() << std::endl;
            std::cerr << "Failed to remove shared memory to write observation: "
                      << j2p_memory_name << " errno=" << errno << std::endl;
            throw std::runtime_error(e.what());
        }
        try {
            j2pMemory = shared_memory_object(
                create_only, j2p_memory_name.c_str(), read_write
            );
        } catch (const interprocess_exception &e) {
            std::cerr << e.what() << std::endl;
            std::cerr << "Failed to create shared memory to write observation: "
                      << j2p_memory_name << " errno=" << errno << std::endl;
            throw std::runtime_error(e.what());
        }
        requiredSize = requiredSize > 1024 ? requiredSize : 1024;
        j2pMemory.truncate(requiredSize);
        j2pMemoryRegion = mapped_region(j2pMemory, read_write);
        J2PSharedMemoryLayout *j2pHeader =
            static_cast<J2PSharedMemoryLayout *>(j2pMemoryRegion.get_address());
        j2pHeader->layout_size = sizeof(J2PSharedMemoryLayout);
        j2pHeader->data_offset = sizeof(J2PSharedMemoryLayout);
        j2pHeader->data_size = observation_size;
        // j2pMemory.grow(j2p_memory_name.c_str(), (requiredSize -
        // currentSize));
    }
    std::cout << "Writing observation to shared memory 2" << std::endl;
    // Write the observation to shared memory
    J2PSharedMemoryLayout *j2pHeader =
        static_cast<J2PSharedMemoryLayout *>(j2pMemoryRegion.get_address());
    char *data_start =
        reinterpret_cast<char *>(j2pHeader) + j2pHeader->data_offset;
    std::memcpy(data_start, data, observation_size);
    j2pHeader->data_size = observation_size;

    // Notify Python that the observation is ready
    std::cout << "Writing observation to shared memory 3" << std::endl;
    lockSynchronization.lock();
    p2jLayout->j2p_ready = true;
    p2jLayout->p2j_ready = false;
    p2jLayout->j2p_recv_ready = false;
    p2jLayout->p2j_recv_ready = false;
    p2jLayout->condition.notify_all();
    lockSynchronization.unlock();
    std::cout << "Wrote observation to shared memory 4" << std::endl;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_readInitialEnvironmentImpl(
    JNIEnv *env, jclass clazz, jstring p2j_memory_name
) {
    const char *p2j_memory_name_cstr =
        env->GetStringUTFChars(p2j_memory_name, nullptr);
    jobject result = nullptr;
    try {
        result = read_initial_environment(env, clazz, p2j_memory_name_cstr);
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
    jbyte *observation_data_ptr =
        env->GetByteArrayElements(observation_data, nullptr);
    jsize observation_data_size = env->GetArrayLength(observation_data);

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