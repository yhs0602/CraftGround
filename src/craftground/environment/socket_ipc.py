import os
import signal
import struct
import time
from typing import Dict, List, Optional, Union

import psutil
from craftground.buffered_socket import BufferedSocket
from craftground.csv_logger import CsvLogger
from craftground.environment.action_space import action_v2_dict_to_message, no_op_v2
from craftground.environment.ipc_interface import IPCInterface
from craftground.proto.action_space_pb2 import ActionSpaceMessageV2
from craftground.proto.initial_environment_pb2 import InitialEnvironmentMessage
from craftground.proto.observation_space_pb2 import ObservationSpaceMessage
import socket


class SocketIPC(IPCInterface):
    def __init__(
        self,
        logger: CsvLogger,
        initial_environment: InitialEnvironmentMessage,
        port: int,
        find_free_port: bool = True,
    ):
        self.logger = logger
        self.find_free_port = find_free_port
        self.remove_orphan_java_processes()
        self.port = self.check_port(port)
        self.sock = None
        self.initial_environment = initial_environment

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
            else:
                return port

    def _send_initial_environment(self, initial_env: InitialEnvironmentMessage):
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
        self.logger.log("Reading response...")
        data_len_bytes = self.buffered_socket.read(4, True)
        data_len = struct.unpack("<I", data_len_bytes)[0]
        data_bytes = self.buffered_socket.read(data_len, True)
        observation_space = ObservationSpaceMessage()
        observation_space.ParseFromString(data_bytes)
        self.logger.log(f"Got response with size {data_len}")
        return observation_space

    def is_alive(self) -> bool:
        return self.sock is not None

    def destroy(self):
        if self.sock:
            self.send_exit()
            self.sock.close()
            self.sock = None

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

    def send_commands(self, commands: List[str]):
        # print("Sending command")
        action_space = action_v2_dict_to_message(no_op_v2())
        action_space.commands.extend(commands)
        v = action_space.SerializeToString()
        self.sock.send(struct.pack("<I", len(v)))
        self.sock.sendall(v)
        # print("Sent command")

    def send_action_and_commands(
        self,
        action_v2: Dict[str, Union[bool, float]],
        commands: List[str],
    ):
        self.logger.log("Sending action and commands")
        action_space = self.action_v2_dict_to_message(action_v2)

        action_space.commands.extend(commands)
        v = action_space.SerializeToString()
        self.sock.send(struct.pack("<I", len(v)))
        self.sock.sendall(v)
        self.logger.log("Sent action and commands")

    def send_fastreset2(self, extra_commands: List[str] = None):
        extra_cmd_str = ""
        if extra_commands is not None:
            extra_cmd_str = ";".join(extra_commands)
        self.send_commands([f"fastreset {extra_cmd_str}"])

    def send_respawn2(self):
        self.send_commands(["respawn"])

    def send_exit(self):
        self.send_commands(["exit"])

    def start_communication(self):
        self._connect_server()
        self.buffered_socket = BufferedSocket(self.sock)
        self._send_initial_environment(self.initial_environment)
        self.logger.log("Sent initial environment")

    def _connect_server(self):
        wait_time = 1
        next_output = 1  # 3 7 15 31 63 127 255  seconds passed
        while True:
            try:
                if os.name == "nt":
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(("127.0.0.1", self.port))
                    s.settimeout(30)
                    self.sock = s
                    return
                else:
                    socket_path = f"/tmp/minecraftrl_{self.port}.sock"
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(socket_path)
                    # s.connect(("127.0.0.1", port))
                    s.settimeout(30)
                    self.sock = s
                    return
            except (ConnectionRefusedError, FileNotFoundError):
                if wait_time == next_output:
                    print(
                        f"Waiting for server on port {self.port}...",
                    )
                    next_output *= 2
                    if next_output > 1024:
                        raise Exception("Server not started within 1024 seconds")
                wait_time += 1
                time.sleep(1)
