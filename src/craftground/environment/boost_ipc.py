import platform
import time
from typing import List, Optional

from ..proto.observation_space_pb2 import ObservationSpaceMessage

from ..csv_logger import CsvLogger
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

from ..environment.action_space import no_op_v2, action_v2_dict_to_message


class BoostIPC(IPCInterface):
    def __init__(
        self,
        port: int,
        find_free_port: bool,
        initial_environment: InitialEnvironmentMessage,
        logger: CsvLogger,
    ):
        self.port = port
        self.logger = logger
        initial_environment_bytes: bytes = initial_environment.SerializeToString()

        # Get the length of the action space message
        dummy_action: ActionSpaceMessageV2 = action_v2_dict_to_message(no_op_v2())
        dummy_action_bytes: bytes = dummy_action.SerializeToString()
        print(f"Length of Dummy_action_bytes: {len(dummy_action_bytes)}")
        self.find_free_port = find_free_port
        self.port = initialize_shared_memory(
            int(self.port),
            initial_environment_bytes,
            len(initial_environment_bytes),
            len(dummy_action_bytes),
            find_free_port,
        )
        self.SHMEM_PREFIX = "Global\\" if platform.system() == "Windows" else "/"
        self.p2j_shared_memory_name = f"{self.SHMEM_PREFIX}craftground_{self.port}_p2j"
        self.j2p_shared_memory_name = f"{self.SHMEM_PREFIX}craftground_{self.port}_j2p"

    def send_action(
        self, action: ActionSpaceMessageV2, commands: Optional[List[str]] = None
    ):
        if not commands:
            commands = []
        action.commands.extend(commands)

        action_bytes: bytes = action.SerializeToString()
        # self.logger.log(f"Sending action to shared memory: {len(action_bytes)} bytes")
        write_to_shared_memory(
            self.p2j_shared_memory_name, action_bytes, len(action_bytes)
        )

    def read_observation(self) -> ObservationSpaceMessage:
        # self.logger.log("Reading observation from shared memory")
        observation_space = ObservationSpaceMessage()
        data_bytes = read_from_shared_memory(
            self.p2j_shared_memory_name, self.j2p_shared_memory_name
        )
        observation_space.ParseFromString(data_bytes)
        return observation_space

    def destroy(self):
        destroy_shared_memory(self.p2j_shared_memory_name, True)
        destroy_shared_memory(self.j2p_shared_memory_name, True)
        # Java destroys the initial environment shared memory
        # destroy_shared_memory(self.initial_environment_shared_memory_name)

    def is_alive(self) -> bool:
        return True

    def remove_orphan_java_processes(self):
        pass

    def send_fastreset2(self, extra_commands: List[str] = None):
        extra_cmd_str = ""
        if extra_commands is not None:
            extra_cmd_str = ";".join(extra_commands)
        self.send_commands([f"fastreset {extra_cmd_str}"])

    def send_commands(self, commands: List[str]):
        # print("Sending command")
        action_space = action_v2_dict_to_message(no_op_v2())
        action_space.commands.extend(commands)
        v = action_space.SerializeToString()
        self.logger.log(f"Sending action to shared memory: {len(v)} bytes")
        write_to_shared_memory(self.p2j_shared_memory_name, v, len(v))

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
