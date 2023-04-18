import base64
import random
import socket

# import pdb
import time
from typing import List

from initial_environment import InitialEnvironment
from json_socket import JSONSocket


def wait_for_server() -> socket.socket:
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", 8000))
            s.settimeout(15)
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
        act[4] = 12 - 2  # Camera delta yaw (0: -180, 24: 180)
    elif input_act == 2:
        act[0] = 2  # go backward
    elif input_act == 3:
        act[4] = 12 + 2  # Camera delta yaw (0: -180, 24: 180)
    elif input_act == 4:
        act[2] = 1  # 0: noop 1: jump 2: sneak 3: sprint
    elif input_act == 5:
        act[
            5
        ] = 3  # 0: noop 1: attack 2: use 3: attack 4: craft 5: equip 6: place 7: destroy
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


def main():
    # pdb.set_trace()
    sock: socket.socket = wait_for_server()
    # buffered_reader = BufferedReader(sock)
    json_socket = JSONSocket(sock)
    img_seq: int = 0
    # send initial environment
    initial_env = InitialEnvironment(
        initialInventoryCommands=["minecraft:diamond_sword", "minecraft:shield"],
        initialPosition=None,  # nullable
        initialMobsCommands=["minecraft:sheep"],
        imageSizeX=800,
        imageSizeY=449,
        seed=123456,  # nullable
        allowMobSpawn=True,
        alwaysDay=False,
        alwaysNight=False,
        initialWeather="clear",  # nullable
    )
    json_socket.send_json_as_base64(initial_env.to_dict())
    print("Sent initial environment")
    # send_initial_environment(sock, initial_env)

    while True:
        random_action = random.randint(0, 8)
        action_arr = int_to_action(random_action)

        # send the action
        print("Sending action...")
        send_action(json_socket, action_arr)
        # read the response
        print("Reading response...")
        res = json_socket.receive_json()
        if res is None:
            # server is not responding
            # send_action(sock, no_op())
            continue
        # save this png byte array to a file
        img = base64.b64decode(res["image"])
        with open(f"{img_seq}.png", "wb") as f:
            f.write(img)
        img_seq += 1

    sock.close()


if __name__ == "__main__":
    main()
