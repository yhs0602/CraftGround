import time
from ..environment.ipc_interface import IPCInterface
from ..proto.action_space_pb2 import ActionSpaceMessageV2
from ..proto.initial_environment_pb2 import InitialEnvironmentMessage

# Torch should be imported first before craftground_native to avoid segfaults
try:
    import torch  # noqa
except ImportError:
    pass

from ..craftground_native import (  # noqa
    initialize_shared_memory,  # noqa
    write_to_shared_memory,  # noqa
    read_from_shared_memory,  # noqa
    destroy_shared_memory,  # noqa
    shared_memory_exists,  # noqa
)


class BoostIPC(IPCInterface):
    def __init__(
        self,
        port: int,
        find_free_port: bool,
        initial_environment: InitialEnvironmentMessage,
    ):
        self.port = port
        initial_environment_bytes: bytes = initial_environment.SerializeToString()

        # Get the length of the action space message
        dummy_action: ActionSpaceMessageV2 = ActionSpaceMessageV2()
        dummy_action_bytes: bytes = dummy_action.SerializeToString()
        self.find_free_port = find_free_port
        self.port = initialize_shared_memory(
            int(self.port),
            initial_environment_bytes,
            len(initial_environment_bytes),
            len(dummy_action_bytes),
            find_free_port,
        )
        self.p2j_shared_memory_name = f"craftground_{port}_p2j"
        self.j2p_shared_memory_name = f"craftground_{port}_j2p"

    def send_action(self, action: ActionSpaceMessageV2):
        action_bytes: bytes = action.SerializeToString()
        write_to_shared_memory(self.p2j_shared_memory_name, action_bytes)

    def read_observation(self) -> bytes:
        return read_from_shared_memory(
            self.p2j_shared_memory_name, self.j2p_shared_memory_name
        )

    def destroy(self):
        destroy_shared_memory(self.p2j_shared_memory_name)
        destroy_shared_memory(self.j2p_shared_memory_name)
        # Java destroys the initial environment shared memory
        # destroy_shared_memory(self.initial_environment_shared_memory_name)

    def is_alive(self) -> bool:
        return True

    def remove_orphan_java_processes(self):
        pass

    def start_communication(self):
        # wait until the j2p shared memory is created
        wait_time = 1
        next_output = 1  # 3 7 15 31 63 127 255  seconds passed
        while True:
            if shared_memory_exists(self.j2p_shared_memory_name):
                break
            self.logger.log(
                f"Waiting for Java process to create shared memory {self.j2p_shared_memory_name}"
            )
            time.sleep(wait_time)
            wait_time *= 2
            if wait_time > 1024:
                raise Exception(
                    f"Java process failed to create shared memory {self.j2p_shared_memory_name}"
                )
        self.logger.log(
            f"Java process created shared memory {self.j2p_shared_memory_name}"
        )

    def __del__(self):
        self.destroy()
