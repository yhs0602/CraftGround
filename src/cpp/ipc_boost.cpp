#include "ipc_boost.hpp"
#include "boost/interprocess/shared_memory_object.hpp"

// Create shared memory and write initial environment data
void create_shared_memory_impl(
    const std::string &initial_memory_name,
    const std::string &synchronization_memory_name,
    const std::string &action_memory_name,
    const char *initial_data,
    size_t data_size,
    size_t action_size
) {
    shared_memory_object::remove(initial_memory_name.c_str());
    managed_shared_memory sharedMemory(
        create_only,
        initial_memory_name.c_str(),
        sizeof(SharedDataHeader) + data_size
    );
    void *addr = sharedMemory.allocate(sizeof(SharedDataHeader) + data_size);

    auto *header = new (addr) SharedDataHeader();
    header->ready = false;

    char *data_start =
        reinterpret_cast<char *>(header) + sizeof(SharedDataHeader);
    std::memcpy(data_start, initial_data, data_size);
    header->size = data_size;

    header->ready = true;
    // Java will remove the initial environment shared memory

    // Create synchronization shared memory (fixed size, the size field )
    shared_memory_object::remove(synchronization_memory_name.c_str());
    managed_shared_memory sharedMemorySynchronization(
        create_only,
        synchronization_memory_name.c_str(),
        sizeof(SharedDataHeader)
    );
    void *addrSyncrhonization =
        sharedMemorySynchronization.allocate(sizeof(SharedDataHeader));
    auto *headerSynchronization = new (addrSyncrhonization) SharedDataHeader();
    headerSynchronization->size = 0;
    headerSynchronization->ready = true;

    // Allocate shared memory for action
    shared_memory_object::remove(action_memory_name.c_str());
    managed_shared_memory sharedMemoryAction(
        create_only,
        action_memory_name.c_str(),
        sizeof(SharedDataHeader) + action_size
    );
    void *addrAction =
        sharedMemoryAction.allocate(sizeof(SharedDataHeader) + action_size);
    auto *headerAction = new (addrAction) SharedDataHeader();
    headerAction->size = action_size;
    headerAction->ready = true;
}

// Write action to shared memory
void write_to_shared_memory_impl(
    const std::string &action_memory_name,
    const char *data,
    const size_t action_size
) {
    managed_shared_memory actionMemory(open_only, action_memory_name.c_str());
    void *addr = actionMemory.get_address();
    auto *actionHeader = reinterpret_cast<SharedDataHeader *>(addr);

    std::unique_lock<interprocess_mutex> actionLock(actionHeader->mutex);
    char *data_start =
        reinterpret_cast<char *>(actionHeader) + sizeof(SharedDataHeader);
    std::memcpy(data_start, data, action_size);
    actionHeader->ready = true;
    actionHeader->condition.notify_one();
    actionLock.unlock();
}

// Read observation from shared memory
const char *read_from_shared_memory_impl(
    const std::string &memory_name,
    const std::string &synchronization_memory_name,
    size_t &data_size
) {
    // Synchronize with Java using synchronization shared memory
    managed_shared_memory synchronizationSharedMemory(
        open_only, synchronization_memory_name.c_str()
    );
    void *addrSynchronization = synchronizationSharedMemory.get_address();
    auto *headerSynchronization =
        reinterpret_cast<SharedDataHeader *>(addrSynchronization);
    std::unique_lock<interprocess_mutex> lockSynchronization(
        headerSynchronization->mutex
    );
    headerSynchronization->condition.wait(lockSynchronization, [&] {
        return headerSynchronization->ready;
    });

    // Read the observation from shared memory
    managed_shared_memory sharedMemory(open_only, memory_name.c_str());
    void *addr = sharedMemory.get_address();
    auto *header = reinterpret_cast<SharedDataHeader *>(addr);
    // Read the message from Java
    char *data_start =
        reinterpret_cast<char *>(header) + sizeof(SharedDataHeader);
    header->ready = false;
    lockSynchronization.unlock();

    data_size = header->size;
    return data_start;
}

// Destroy shared memory
void destroy_shared_memory_impl(const std::string &memory_name) {
    shared_memory_object::remove(memory_name.c_str());
}
