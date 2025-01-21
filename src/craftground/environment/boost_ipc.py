from craftground_native import (  # noqa
    initialize_shared_memory,  # noqa
    write_to_shared_memory,  # noqa
    read_from_shared_memory,  # noqa
    destroy_shared_memory,  # noqa
)
from environment.ipc_interface import IPCInterface
from proto.action_space_pb2 import ActionSpaceMessageV2
from proto.initial_environment_pb2 import InitialEnvironmentMessage


class BoostIPC(IPCInterface):
    def __init__(self, port: str, find_free_port: bool, initial_environment: InitialEnvironmentMessage):
        self.port = port
        self.initial_environment_shared_memory_name = (
            f"craftground_{port}_initial_environment"
        )
        self.action_shared_memory_name = f"craftground_{port}_action"
        self.observation_shared_memory_name = f"craftground_{port}_observation"
        self.synchronization_shared_memory_name = f"craftground_{port}_synchronization"

        initial_environment_bytes: bytes = initial_environment.SerializeToString()

        # Get the length of the action space message
        dummy_action: ActionSpaceMessageV2 = ActionSpaceMessageV2()
        dummy_action_bytes: bytes = dummy_action.SerializeToString()
        self.port = initialize_shared_memory(
            self.initial_environment_shared_memory_name,
            self.synchronization_shared_memory_name,
            self.action_shared_memory_name,
            initial_environment_bytes,
            len(initial_environment_bytes),
            len(dummy_action_bytes),
            find_free_port,
        )

    def write_action(self, action: ActionSpaceMessageV2):
        action_bytes: bytes = action.SerializeToString()
        write_to_shared_memory(
            self.action_shared_memory_name, action_bytes, len(action_bytes)
        )

    def read_observation(self) -> bytes:
        return read_from_shared_memory(
            self.observation_shared_memory_name, self.synchronization_shared_memory_name
        )

    def destroy(self):
        destroy_shared_memory(self.action_shared_memory_name)
        destroy_shared_memory(self.observation_shared_memory_name)
        destroy_shared_memory(self.synchronization_shared_memory_name)
        # Java destroys the initial environment shared memory
        # destroy_shared_memory(self.initial_environment_shared_memory_name)
