import socket

# import pdb
import time
from typing import List

from mydojo.json_socket import JSONSocket


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
    if input_act == 0:
        act[0] = 1  # 0: noop 1: forward 2 : back
    elif input_act == 1:
        act[4] = 12 - 1  # Camera delta yaw (0: -180, 24: 180)
    elif input_act == 2:
        act[0] = 2  # go backward
    elif input_act == 3:
        act[4] = 12 + 1  # Camera delta yaw (0: -180, 24: 180)
    elif input_act == 4:
        act[2] = 1  # 0: noop 1: jump 2: sneak 3: sprint
    elif input_act == 5:
        act[
            5
        ] = 3  # 0: noop 1: use 2: drop 3: attack 4: craft 5: equip 6: place 7: destroy
    elif input_act == 6:
        act[1] = 1  # 0: noop 1: move right 2: move left
    elif input_act == 7:
        act[1] = 2  # 0: noop 1: move right 2: move left
    elif input_act == 8:
        act[5] = 4  # craft
        act[6] = 331  # iron bar?
    return act


def send_action(sock: JSONSocket, action_array: List[int]):
    sock.send_json_as_base64({"action": action_array, "command": ""})


def send_respawn(sock: JSONSocket):
    sock.send_json_as_base64({"action": no_op(), "command": "respawn"})


def send_fastreset(sock: JSONSocket):
    sock.send_json_as_base64({"action": no_op(), "command": "fastreset"})
