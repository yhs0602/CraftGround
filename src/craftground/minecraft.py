import os
import socket
import struct

# import pdb
import time
from typing import List, Dict, Union

from environment.action_space import action_v2_dict_to_message, no_op_v2

from .print_with_time import print_with_time
from .proto import action_space_pb2


def wait_for_server(port: int) -> socket.socket:
    wait_time = 1
    next_output = 1  # 3 7 15 31 63 127 255  seconds passed

    while True:
        try:
            if os.name == "nt":
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(("127.0.0.1", port))
                s.settimeout(30)
                return s
            else:
                socket_path = f"/tmp/minecraftrl_{port}.sock"
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(socket_path)
                # s.connect(("127.0.0.1", port))
                s.settimeout(30)
                return s
        except (ConnectionRefusedError, FileNotFoundError):
            if wait_time == next_output:
                print(
                    f"Waiting for server on port {port}...",
                )
                next_output *= 2
                if next_output > 1024:
                    raise Exception("Server not started within 1024 seconds")
            wait_time += 1
            time.sleep(1)


def send_commands(sock: socket.socket, commands: List[str]):
    # print("Sending command")
    action_space = action_v2_dict_to_message(no_op_v2())
    action_space.commands.extend(commands)
    v = action_space.SerializeToString()
    sock.send(struct.pack("<I", len(v)))
    sock.sendall(v)
    # print("Sent command")


def send_action_and_commands(
    sock: socket.socket,
    action_v2: Dict[str, Union[bool, float]],
    commands: List[str],
    verbose: bool = False,
):
    if verbose:
        print_with_time("Sending action and commands")
    action_space = action_v2_dict_to_message(action_v2)

    action_space.commands.extend(commands)
    v = action_space.SerializeToString()
    sock.send(struct.pack("<I", len(v)))
    sock.sendall(v)
    if verbose:
        print_with_time("Sent actions and commands")


def send_fastreset2(sock: socket.socket, extra_commands: List[str] = None):
    extra_cmd_str = ""
    if extra_commands is not None:
        extra_cmd_str = ";".join(extra_commands)
    send_commands(sock, [f"fastreset {extra_cmd_str}"])


def send_respawn2(sock: socket.socket):
    send_commands(sock, ["respawn"])


def send_exit(sock: socket.socket):
    send_commands(sock, ["exit"])
