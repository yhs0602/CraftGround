import base64
import json
import random

# import pdb
import socket
import time
from typing import List, Dict, Optional, Any


def recvall(sock):
    BUFF_SIZE = 1024  # 1 KiB
    data = b""
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    return data


def wait_for_server() -> socket.socket:
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", 8000))
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
    # elif input_act == 6:
    #     act[3] = 12 - 1  # Camera delta pitch (0: -180, 24: 180)
    # elif input_act == 7:
    #     act[3] = 12 + 1  # Camera delta pitch (0: -180, 24: 180)
    return act


def send_action(sock: socket.socket, action_array: List[int]):
    send_payload(sock, {"action": action_array, "command": ""})


def send_payload(sock: socket.socket, action_payload):
    dumped = json.dumps(action_payload)
    message_bytes = dumped.encode("utf-8")
    base64_bytes = base64.b64encode(message_bytes)
    base64_message = base64_bytes.decode("utf-8")
    sock.send(bytes(base64_message + "\n", "utf-8"))


def decode_response(sock: socket.socket) -> Optional[Dict[str, Any]]:
    sock.settimeout(5)
    try:
        re = recvall(sock)
    except socket.timeout:
        return None
    print(len(re))
    sr = re.decode("utf-8")
    data = json.loads(sr)
    print(data["x"], data["y"], data["z"])
    img = data["image"]  # wxh|base64
    decoded_img = base64.b64decode(img)
    return {
        "x": data["x"],
        "y": data["y"],
        "z": data["z"],
        "image": decoded_img,
    }


def main():
    # pdb.set_trace()
    sock: socket.socket = wait_for_server()
    img_seq: int = 0
    while True:
        random_action = random.randint(0, 7)  # 0, 1, 2, 3..7
        action_arr = int_to_action(random_action)

        # send the action
        print("Sending action...")
        send_action(sock, action_arr)
        # read the response
        print("Reading response...")
        res = decode_response(sock)
        if res is None:
            # server is not responding
            # send_action(sock, no_op())
            continue

        # save this png byte array to a file
        with open(f"{img_seq}.png", "wb") as f:
            f.write(res["image"])
        img_seq += 1

    sock.close()


# send a single character to 127.0.0.1:8000
if __name__ == "__main__":
    main()
