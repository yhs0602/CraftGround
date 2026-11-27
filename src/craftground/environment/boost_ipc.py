import platform
import threading
from typing import List, Optional

from ..proto.observation_space_pb2 import ObservationSpaceMessage

from ..csv_logger import CsvLogger
from ..environment.ipc_interface import IPCInterface
from ..proto.action_space_pb2 import ActionSpaceMessageV2
from ..proto.initial_environment_pb2 import HandshakeAck, InitialEnvironmentMessage
from ..protocol_version import validate_handshake_ack

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
        self.closed = True
        self._connected = False
        self.port = port
        self.logger = logger
        self.initial_environment = initial_environment
        initial_environment_bytes: bytes = initial_environment.SerializeToString()

        initial_action_size = action_v2_dict_to_message(no_op_v2()).ByteSize()
        self.find_free_port = find_free_port
        self.port = initialize_shared_memory(
            int(self.port),
            initial_environment_bytes,
            len(initial_environment_bytes),
            initial_action_size,
            find_free_port,
        )
        if self.port < 0:
            raise RuntimeError("Could not create shared memory")
        self.SHMEM_PREFIX = "Global\\" if platform.system() == "Windows" else "/"
        self.p2j_shared_memory_name = f"{self.SHMEM_PREFIX}craftground_{self.port}_p2j"
        self.j2p_shared_memory_name = f"{self.SHMEM_PREFIX}craftground_{self.port}_j2p"
        self.closed = False

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
        data_bytes = self._read_message()
        observation_space.ParseFromString(data_bytes)
        return observation_space

    def destroy(self):
        if not self.closed:
            self.closed = True
            self._connected = False
            destroy_shared_memory(self.j2p_shared_memory_name, False)
            destroy_shared_memory(self.p2j_shared_memory_name, True)

    def is_alive(self) -> bool:
        return self._connected

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

    def _read_message(self) -> bytes:
        if self._server_event.is_set():
            raise RuntimeError("Minecraft exited while waiting for shared memory")
        return read_from_shared_memory(
            self.p2j_shared_memory_name, self.j2p_shared_memory_name, self._server_event
        )

    def _read_handshake_ack(self) -> HandshakeAck:
        # Java writes exactly one HandshakeAck payload over the same j2p channel/protocol as a
        # real observation (SharedMemoryMessageIO.writeHandshakeAck reuses writeObservationImpl),
        # so this consumes it with the same read_from_shared_memory call used by
        # read_observation(), before the caller starts treating j2p reads as observations.
        data_bytes = self._read_message()
        ack = HandshakeAck()
        ack.ParseFromString(data_bytes)
        return ack

    def start_communication(self, server_event: threading.Event):
        self._server_event = server_event
        ack = self._read_handshake_ack()
        self.logger.log(
            f"Received handshake ack: protocol_version={ack.protocol_version} "
            f"minecraft_version={ack.minecraft_version} render_backend={ack.render_backend} "
            f"capabilities={list(ack.capabilities)}"
        )
        validate_handshake_ack(ack, self.initial_environment.screen_encoding_mode)

        self._connected = True

    def notify_process_exit(self):
        if not self.closed and platform.system() != "Windows":
            from ..craftground_native import notify_shared_memory_exit

            notify_shared_memory_exit(self.p2j_shared_memory_name)

    def __del__(self):
        self.destroy()
