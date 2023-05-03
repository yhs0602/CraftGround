import socket
import struct

# import pdb
import time
from typing import List

from mydojo.json_socket import JSONSocket
from mydojo.proto import action_space_pb2


def wait_for_server() -> socket.socket:
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", 8000))
            s.settimeout(30)
            return s
        except ConnectionRefusedError:
            print("Waiting for server...")
            time.sleep(1)


def no_op() -> List[int]:
    r = [0] * 8
    r[3] = 12
    r[4] = 12
    return r


def int_to_action(input_act: int):
    act = no_op()
    if input_act == 0:  # go forward
        act[0] = 1  # 0: noop 1: forward 2 : back
    elif input_act == 1:  # go backward
        act[0] = 2  # 0: noop 1: forward 2 : back
    elif input_act == 2:  # move right
        act[1] = 1  # 0: noop 1: move right 2: move left
    elif input_act == 3:  # move left
        act[1] = 2  # 0: noop 1: move right 2: move left
    elif input_act == 4:  # Turn left
        act[4] = 12 - 1  # Camera delta yaw (0: -180, 24: 180)
    elif input_act == 5:  # Turn right
        act[0] = 12 + 1  # Camera delta yaw (0: -180, 24: 180)
    elif input_act == 6:  # Jump
        act[2] = 1  # 0: noop 1: jump 2: sneak 3: sprint
    elif input_act == 7:  # Attack
        act[5] = 3
        # 0: noop 1: use 2: drop 3: attack 4: craft 5: equip 6: place 7: destroy
    elif input_act == 8:  # Look up
        act[3] = 12 + 1  # Camera delta pitch (0: -180, 24: 180)
    elif input_act == 9:  # Look down
        act[3] = 12 - 1  # Camera delta pitch (0: -180, 24: 180)
    elif input_act == 10:
        act[5] = 1  # use
    elif input_act == 11:
        act[5] = 4  # craft
        act[6] = 331  # iron bar?
    return act


def send_action(sock: JSONSocket, action_array: List[int]):
    sock.send_json_as_base64({"action": action_array, "command": ""})


def send_action2(sock: socket.socket, action_array: List[int]):
    # print("Sending action")
    action_space = action_space_pb2.ActionSpaceMessage()
    action_space.action.extend(action_array)
    action_space.command = ""
    v = action_space.SerializeToString()
    sock.send(struct.pack("<I", len(v)))
    sock.sendall(v)
    # print("Sent action")


def send_command(sock: socket.socket, command: str):
    # print("Sending command")
    action_space = action_space_pb2.ActionSpaceMessage()
    action_space.action.extend(no_op())
    action_space.command = command
    v = action_space.SerializeToString()
    sock.send(struct.pack("<I", len(v)))
    sock.sendall(v)
    # print("Sent command")


def send_fastreset2(sock: socket.socket):
    send_command(sock, "fastreset")


def send_respawn2(sock: socket.socket):
    send_command(sock, "respawn")


def send_respawn(sock: JSONSocket):
    sock.send_json_as_base64({"action": no_op(), "command": "respawn"})


def send_fastreset(sock: JSONSocket):
    sock.send_json_as_base64({"action": no_op(), "command": "fastreset"})
