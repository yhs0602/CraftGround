import socket
import struct

# import pdb
import time
from typing import List, Dict, Union

from .print_with_time import print_with_time
from .proto import action_space_pb2


def wait_for_server(port: int) -> socket.socket:
    wait_time = 1
    next_output = 1  # 3 7 15 31 63 127 255  seconds passed

    while True:
        try:
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


def no_op() -> List[int]:
    r = [0] * 8
    r[3] = 12
    r[4] = 12
    return r


def no_op_v2() -> Dict[str, Union[bool, float]]:
    noop_dict = {}
    for bool_key in [
        "attack",
        "back",
        "forward",
        "jump",
        "left",
        "right",
        "sneak",
        "sprint",
        "use",
        "drop",
        "inventory",
    ]:
        noop_dict[bool_key] = False
    for i in range(1, 10):
        noop_dict[f"hotbar.{i}"] = False
    noop_dict["camera_pitch"] = 0.0
    noop_dict["camera_yaw"] = 0.0
    return noop_dict


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


def action_v2_dict_to_message(action_v2):
    action_space = action_space_pb2.ActionSpaceMessageV2()
    action_space.attack = action_v2["attack"]
    action_space.back = action_v2["back"]
    action_space.forward = action_v2["forward"]
    action_space.jump = action_v2["jump"]
    action_space.left = action_v2["left"]
    action_space.right = action_v2["right"]
    action_space.sneak = action_v2["sneak"]
    action_space.sprint = action_v2["sprint"]
    action_space.use = action_v2["use"]
    action_space.drop = action_v2["drop"]
    action_space.inventory = action_v2["inventory"]
    action_space.hotbar_1 = action_v2["hotbar.1"]
    action_space.hotbar_2 = action_v2["hotbar.2"]
    action_space.hotbar_3 = action_v2["hotbar.3"]
    action_space.hotbar_4 = action_v2["hotbar.4"]
    action_space.hotbar_5 = action_v2["hotbar.5"]
    action_space.hotbar_6 = action_v2["hotbar.6"]
    action_space.hotbar_7 = action_v2["hotbar.7"]
    action_space.hotbar_8 = action_v2["hotbar.8"]
    action_space.hotbar_9 = action_v2["hotbar.9"]
    action_space.camera_pitch = action_v2["camera_pitch"]
    action_space.camera_yaw = action_v2["camera_yaw"]
    return action_space


def send_fastreset2(sock: socket.socket, extra_commands: List[str] = None):
    extra_cmd_str = ""
    if extra_commands is not None:
        extra_cmd_str = ";".join(extra_commands)
    send_commands(sock, [f"fastreset {extra_cmd_str}"])


def send_respawn2(sock: socket.socket):
    send_commands(sock, ["respawn"])


def send_exit(sock: socket.socket):
    send_commands(sock, ["exit"])
