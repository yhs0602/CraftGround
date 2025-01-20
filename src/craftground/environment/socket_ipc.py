import os
import signal
import struct
from typing import List, Optional, Tuple

import psutil
from csv_logger import CsvLogger
from environment.ipc_interface import IPCInterface
from minecraft import action_v2_dict_to_message
from print_with_time import print_with_time
from proto.action_space_pb2 import ActionSpaceMessageV2
from proto.initial_environment_pb2 import InitialEnvironmentMessage
from proto.observation_space_pb2 import ObservationSpaceMessage


class SocketIPC(IPCInterface):
    def __init__(self, logger: CsvLogger, port: int, find_free_port: bool = False):
        self.logger = logger
        self.find_free_port = find_free_port
        self.remove_orphan_java_processes()
        self.port = self.check_port(port)

    def check_port(self, port: int) -> int:
        if os.name == "nt":
            while True:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(("127.0.0.1", port)) == 0:  # The port is in use
                        if self.find_free_port:
                            print(
                                f"[Warning]: Port {port} is already in use. Trying another port."
                            )
                            port += 1
                        else:
                            raise ConnectionError(
                                f"Port {port} is already in use. Please choose another port."
                            )
                    else:
                        return port
        else:
            socket_path = f"/tmp/minecraftrl_{port}.sock"
            if os.path.exists(socket_path):
                if self.find_free_port:
                    print(
                        f"[Warning]: Socket file {socket_path} already exists. Trying another port."
                    )
                    while os.path.exists(socket_path):
                        port += 1
                        socket_path = f"/tmp/minecraftrl_{port}.sock"
                    print(f"Using port {socket_path}")
                    return port
                else:
                    raise FileExistsError(
                        f"Socket file {socket_path} already exists. Please choose another port."
                    )

    def send_initial_environment(self, initial_env: InitialEnvironmentMessage):
        v = initial_env.SerializeToString()
        self.sock.send(struct.pack("<I", len(v)))
        self.sock.sendall(v)

    def send_action(
        self, action: ActionSpaceMessageV2, commands: Optional[List[str]] = None
    ):
        if not commands:
            commands = []

        self.logger.log("Sending action and commands")
        action.commands.extend(commands)
        v = action.SerializeToString()
        self.sock.send(struct.pack("<I", len(v)))
        self.sock.sendall(v)
        self.logger.log("Sent action and commands")

    def read_observation(self) -> ObservationSpaceMessage:
        data_len_bytes = self.buffered_socket.read(4, True)
        data_len = struct.unpack("<I", data_len_bytes)[0]
        data_bytes = self.buffered_socket.read(data_len, True)
        observation_space = ObservationSpaceMessage()
        observation_space.ParseFromString(data_bytes)
        return observation_space

    def destroy(self):
        pass

    def remove_orphan_java_processes(self):  # noqa: C901
        self.logger.log("Removing orphan Java processes...")
        target_directory = "/tmp"
        file_pattern = "minecraftrl_"
        file_usage = {}
        no_such_processes = 0
        access_denied_processes = 0
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                for file in proc.open_files():
                    if (
                        file.path.startswith(target_directory)
                        and file_pattern in file.path
                    ):
                        if file.path not in file_usage:
                            file_usage[file.path] = []
                        file_usage[file.path].append(proc.info)
            except psutil.NoSuchProcess:
                no_such_processes += 1
                continue
            except psutil.AccessDenied:
                access_denied_processes += 1
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue

        for file_path, processes in file_usage.items():
            if all(proc["name"].lower() == "java" for proc in processes):
                for proc in processes:
                    os.kill(proc["pid"], signal.SIGTERM)
                    print(f"Killed Java process {proc['pid']} using file {file_path}")
                os.remove(file_path)
                print(f"Removed {file_path}")
        print(
            f"Removed orphan Java processes: {access_denied_processes} access denied, {no_such_processes} no such process"
        )
