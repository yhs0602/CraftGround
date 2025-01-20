#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <cstring>
#include <iostream>

using namespace boost::interprocess;

struct SharedData {
    interprocess_mutex mutex;
    interprocess_condition condition;
    bool ready;               // Data ready flag
    char data[256];           // Message buffer
};

int main() {
    // Shared memory
    shared_memory_object::remove("PythonJavaSharedMemory");
    managed_shared_memory sharedMemory(create_only, "PythonJavaSharedMemory", 1024);

    // Shared memory data
    SharedData* sharedData = sharedMemory.construct<SharedData>("SharedData")();
    sharedData->ready = false;

    while (true) {
        // Wait for Python to send a message
        std::unique_lock<interprocess_mutex> lock(sharedData->mutex);
        sharedData->condition.wait(lock, [&] { return sharedData->ready; });

        // Read the message from Python
        std::cout << "Received from Python: " << sharedData->data << std::endl;

        // Send a message to Python
        std::strcpy(sharedData->data, "Acknowledged from Java");
        sharedData->ready = false;

        // Notify Python that the message has been sent
        sharedData->condition.notify_one();
    }

    // Cleanup
    shared_memory_object::remove("PythonJavaSharedMemory");
    return 0;
}
