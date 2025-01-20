#include "boost/interprocess/shared_memory_object.hpp"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <cstring>
#include <iostream>

using namespace boost::interprocess;

// explicit structure of the object is handled using protobuf
struct SharedDataHeader {
    interprocess_mutex mutex;
    interprocess_condition condition;
    bool ready;     // Data ready flag
};

// char data[8]; // Message buffer (Acutally, protobuf object size, which can
//                     // be calcluated using protobuf)
//                     // TODO: use buffer size of protobuf object, not 256

// 1. Receive the initial environment message from Python (Python -> Java)
// 2. Send observation to python (Java -> Python)
// 3. Receive action from python (Python -> Java)
// Prepare Shared memory
void loop(
    std::string initial_environment_memory_name,
    std::string memory_name,
    int initial_environment_size,
    int observation_size,
    int action_size
) {
    // Prepare initial environment shared memory
    shared_memory_object::remove(initial_environment_memory_name.c_str());
    managed_shared_memory sharedMemoryInitialEnvironment(
        create_only,
        initial_environment_memory_name.c_str(),
        sizeof(SharedDataHeader) + initial_environment_size
    );

    // Read the initial environment
    SharedDataHeader *sharedDataInitialEnvironment =
        sharedMemoryInitialEnvironment.construct<SharedDataHeader>("SharedDataInitialEnvironment")();
    sharedDataInitialEnvironment->ready = false;

    sharedDataInitialEnvironment->condition.notify_one();



    shared_memory_object::remove(memory_name.c_str());
    managed_shared_memory sharedMemory(
        create_only,
        memory_name.c_str(),
        1024 // TODO: Size of protobufdata sizes
    );

    // Shared memory data for initial environment message
    SharedData *sharedData = sharedMemory.construct<SharedData>("SharedData")();
    sharedData->ready = false;
    // Shared memory data for observation message
    SharedData *sharedDataObs =
        sharedMemory.construct<SharedData>("SharedDataObs")();
    sharedDataObs->ready = false;
    // Shared memory data for action message
    SharedData *sharedDataAction =
        sharedMemory.construct<SharedData>("SharedDataAction")();
    sharedDataAction->ready = false;

    // 1. Receive the initial environment message from Python (Python -> Java)

    // Wait for Python to send a message
    std::unique_lock<interprocess_mutex> lock(sharedData->mutex);
    sharedData->condition.wait(lock, [&] { return sharedData->ready; });

    // Read the message from Python
    std::cout << "Received from Python: " << sharedData->data << std::endl;

    // Do something with the initial configuration message

    while (true) {
        // Send the observation to Python
        std::strcpy(sharedDataObs->data, "Observation from Java");
        sharedDataObs->ready = true;
        // Notify Python that the observation has been sent
        sharedDataObs->condition.notify_one();

        // Wait for Python to send a message
        std::unique_lock<interprocess_mutex> lock(sharedDataAction->mutex);
        sharedDataAction->condition.wait(lock, [&] {
            return sharedDataAction->ready;
        });

        // Read the message from Python
        std::cout << "Received Action from Python: " << sharedDataAction->data
                  << std::endl;
    }

    // Cleanup
    shared_memory_object::remove("PythonJavaSharedMemory");
    return 0;
}
