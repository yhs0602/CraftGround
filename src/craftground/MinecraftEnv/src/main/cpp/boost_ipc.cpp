#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <cstddef>
#include <cstring>
#include <jni.h>

using namespace boost::interprocess;

// explicit structure of the object is handled using protobuf
struct SharedDataHeader {
    interprocess_mutex mutex;
    interprocess_condition condition;
    size_t size;
    bool ready;
};
// Message follows the header

// Returns ByteArray object containing the initial environment message
jobject read_initial_environment(
    JNIEnv *env,
    jclass clazz,
    const std::string &initial_environment_memory_name
) {
    managed_shared_memory sharedMemoryInitialEnvironment(
        open_only, initial_environment_memory_name.c_str()
    );
    void *addrInitialEnvironment = sharedMemoryInitialEnvironment.get_address();
    auto *headerInitialEnvironment =
        reinterpret_cast<SharedDataHeader *>(addrInitialEnvironment);
    // Read the initial environment message
    char *data_startInitialEnvironment =
        reinterpret_cast<char *>(headerInitialEnvironment) +
        sizeof(SharedDataHeader);
    size_t data_size = headerInitialEnvironment->size;

    jbyteArray byteArray = env->NewByteArray(data_size);
    if (byteArray == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    env->SetByteArrayRegion(
        byteArray,
        0,
        data_size,
        reinterpret_cast<jbyte *>(data_startInitialEnvironment)
    );
    // Delete the shared memory
    shared_memory_object::remove(initial_environment_memory_name.c_str());
    return byteArray;
}

jbyteArray read_action(
    JNIEnv *env,
    jclass clazz,
    const std::string &action_memory_name,
    jbyteArray data
) {
    managed_shared_memory actionSharedMemory(
        open_only, action_memory_name.c_str()
    );
    void *addr = actionSharedMemory.get_address();
    auto *actionHeader = reinterpret_cast<SharedDataHeader *>(addr);

    std::unique_lock<interprocess_mutex> actionLock(actionHeader->mutex);
    actionHeader->condition.wait(actionLock, [&] {
        return actionHeader->ready;
    });

    if (data == nullptr) {
        data = env->NewByteArray(actionHeader->size);
    }
    if (data == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    // Read the action message
    char *data_start =
        reinterpret_cast<char *>(actionHeader) + sizeof(SharedDataHeader);
    env->SetByteArrayRegion(
        data, 0, actionHeader->size, reinterpret_cast<jbyte *>(data_start)
    );
    actionHeader->ready = false;
    actionLock.unlock();
    return data;
}

void write_observation(
    const std::string &observation_memory_name,
    const std::string &synchronization_memory_name,
    const char *data,
    const size_t observation_size
) {
    managed_shared_memory synchronizationSharedMemory(
        open_only, synchronization_memory_name.c_str()
    );
    void *addrSynchronization = synchronizationSharedMemory.get_address();
    auto *headerSynchronization =
        reinterpret_cast<SharedDataHeader *>(addrSynchronization);
    std::unique_lock<interprocess_mutex> lockSynchronization(
        headerSynchronization->mutex
    );
    headerSynchronization->ready = false;

    managed_shared_memory observationSharedMemory(
        open_only, observation_memory_name.c_str()
    );

    // Resize the shared memory if needed
    const size_t currentSize = observationSharedMemory.get_size();
    const size_t requiredSize = observation_size + sizeof(SharedDataHeader);
    if (currentSize < requiredSize) {
        observationSharedMemory.grow(
            observation_memory_name.c_str(), (requiredSize - currentSize)
        );
    }

    // Write the observation to shared memory
    void *observationAddr = observationSharedMemory.get_address();
    auto *observationHeader =
        reinterpret_cast<SharedDataHeader *>(observationAddr);
    char *data_start =
        reinterpret_cast<char *>(observationHeader) + sizeof(SharedDataHeader);
    std::memcpy(data_start, data, observation_size);
    observationHeader->size = observation_size;
    observationHeader->ready = true;
    // mutex, condition of observationHeader SHOULD NOT BE USED. Use
    // synchronizationSharedMemory instead

    // Notify Python that the observation is ready
    headerSynchronization->ready = true;
    headerSynchronization->condition.notify_one();
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_readInitialEnvironmentImpl(
    JNIEnv *env, jclass clazz, jstring initial_environment_memory_name
) {
    const char *initial_environment_memory_name_cstr =
        env->GetStringUTFChars(initial_environment_memory_name, nullptr);
    jobject result = read_initial_environment(
        env, clazz, initial_environment_memory_name_cstr
    );
    env->ReleaseStringUTFChars(
        initial_environment_memory_name, initial_environment_memory_name_cstr
    );
    return result;
}

// fun readAction(action_memory_name: String, action_data: ByteArray?):
// ByteArray
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_kyhsgeekcode_minecraftenv_FramebufferCapturer_readActionImpl(
    JNIEnv *env,
    jclass clazz,
    jstring action_memory_name,
    jbyteArray action_data
) {
    const char *action_memory_name_cstr =
        env->GetStringUTFChars(action_memory_name, nullptr);

    size_t data_size = 0;
    jbyteArray data =
        read_action(env, clazz, action_memory_name_cstr, action_data);
    env->ReleaseStringUTFChars(action_memory_name, action_memory_name_cstr);
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
    jstring observation_memory_name,
    jstring synchronization_memory_name,
    jbyteArray observation_data
) {
    const char *observation_memory_name_cstr =
        env->GetStringUTFChars(observation_memory_name, nullptr);
    const char *synchronization_memory_name_cstr =
        env->GetStringUTFChars(synchronization_memory_name, nullptr);
    jbyte *observation_data_ptr =
        env->GetByteArrayElements(observation_data, nullptr);
    jsize observation_data_size = env->GetArrayLength(observation_data);

    write_observation(
        observation_memory_name_cstr,
        synchronization_memory_name_cstr,
        reinterpret_cast<const char *>(observation_data_ptr),
        static_cast<size_t>(observation_data_size)
    );

    env->ReleaseStringUTFChars(
        observation_memory_name, observation_memory_name_cstr
    );
    env->ReleaseStringUTFChars(
        synchronization_memory_name, synchronization_memory_name_cstr
    );
    env->ReleaseByteArrayElements(
        observation_data, observation_data_ptr, JNI_ABORT
    );
}