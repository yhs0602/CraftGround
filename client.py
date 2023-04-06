import base64
import json
import random
import socket

# import pdb
import time
from typing import List, Dict, Optional, Any


class BufferedReader:
    def __init__(self, sock):
        self.sock = sock
        self.buffer = b""

    def readline(self):
        while b"\n" not in self.buffer:  # 일단 \n이 있거나 더 받을 게 없으면 break
            chunk = self.sock.recv(1024)
            if not chunk:
                break
            self.buffer += chunk
        if b"\n" in self.buffer:  # \n이 있으면 그 전까지 잘라서 리턴
            line, self.buffer = self.buffer.split(b"\n", 1)
            return line  # \n은 포함하지 않음
        else:
            line = self.buffer
            self.buffer = b""
            return line


BUFFER_SIZE = 1024


class JSONSocket:
    def __init__(self, sock):
        self.sock = sock
        self.buffer = ""
        self.extra = ""
        self.decoder = json.JSONDecoder()

    def receive_json(self, wait=False) -> Optional[Dict[str, Any]]:
        while True:
            # need more data: from extra or from socket
            if self.extra:
                # consume extra
                self.buffer += self.extra
                self.extra = ""
            else:
                # consume socket
                try:
                    byte_buffer = self.sock.recv(BUFFER_SIZE)
                    if not byte_buffer:
                        # Connection closed by remote end
                        raise ValueError("Incomplete JSON object")
                    self.buffer += byte_buffer.decode(
                        "utf-8"
                    )  # assume only ascii, so that decode never fails
                except socket.timeout:
                    if wait:
                        print("Waiting")
                        time.sleep(0.01)
                        continue
                    else:
                        return None
            # extra = "", buffer = "new data"
            # extra = "", buffer = "data that was in extra"
            try:
                obj, index = self.decoder.raw_decode(self.buffer)
                self.extra = self.buffer[index:]  # .strip()
                self.buffer = ""
                return obj
            except ValueError:
                # Incomplete JSON object received, continue reading from socket
                # extra is empty, because we didn't find a valid JSON object yet
                continue

    def send_json_as_base64(self, obj):
        dumped = json.dumps(obj)
        message_bytes = dumped.encode("utf-8")
        base64_bytes = base64.b64encode(message_bytes)
        base64_message = base64_bytes.decode("utf-8")
        self.sock.sendall(bytes(base64_message + "\n", "utf-8"))


class BufferedJsonReader:
    def __init__(self, sock):
        self.sock = sock
        self.buffer = b""
        self.decoder = json.JSONDecoder()

    def read_one_object(self):
        while b"\n" not in self.buffer:  # 일단 \n이 있거나 더 받을 게 없으면 break
            chunk = self.sock.recv(1024)
            if not chunk:
                break
            self.buffer += chunk
        if b"\n" in self.buffer:  # \n이 있으면 그 전까지 잘라서 리턴
            line, self.buffer = self.buffer.split(b"\n", 1)
            return json.loads(line)  # \n은 포함하지 않음
        else:
            line = self.buffer
            self.buffer = b""
            return json.loads(line)


class InitialEnvironment:
    def __init__(
        self,
        initialInventoryCommands,
        initialPosition,
        initialMobsCommands,
        imageSizeX,
        imageSizeY,
        seed,
        allowMobSpawn,
        alwaysNight,
        alwaysDay,
        initialWeather,
    ):
        self.initialInventoryCommands = initialInventoryCommands
        self.initialPosition = initialPosition
        self.initialMobsCommands = initialMobsCommands
        self.imageSizeX = imageSizeX
        self.imageSizeY = imageSizeY
        self.seed = seed
        self.allowMobSpawn = allowMobSpawn
        self.alwaysNight = alwaysNight
        self.alwaysDay = alwaysDay
        self.initialWeather = initialWeather

    def to_dict(self) -> Dict[str, Any]:
        initial_env_dict = {
            "initialInventoryCommands": self.initialInventoryCommands,
            "initialPosition": self.initialPosition,
            "initialMobsCommands": self.initialMobsCommands,
            "imageSizeX": self.imageSizeX,
            "imageSizeY": self.imageSizeY,
            "seed": self.seed,
            "allowMobSpawn": self.allowMobSpawn,
            "alwaysNight": self.alwaysNight,
            "alwaysDay": self.alwaysDay,
            "initialWeather": self.initialWeather,
        }
        return {k: v for k, v in initial_env_dict.items() if v is not None}


# initial_env = InitialEnvironment(["sword", "shield"], [10, 20], ["summon ", "killMob"], 800, 600, 123456, True, False, False, "sunny")


def recvline(sock, leftover=b"") -> (List[bytes], bytes):
    CHUNK_SIZE = 1024  # 1 KiB
    data = leftover
    while True:
        part = sock.recv(CHUNK_SIZE)
        data += part
        if b"\n" in part:
            break
    lines = data.split(b"\n")
    leftover = lines[-1]
    complete_lines = lines[:-1]
    return (complete_lines, leftover)


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
    # elif input_act == 6:
    #     act[3] = 12 - 1  # Camera delta pitch (0: -180, 24: 180)
    # elif input_act == 7:
    #     act[3] = 12 + 1  # Camera delta pitch (0: -180, 24: 180)
    return act


def send_action(sock: JSONSocket, action_array: List[int]):
    sock.send_json_as_base64({"action": action_array, "command": ""})


# def send_initial_environment(sock: socket.socket, environment: InitialEnvironment):
#     send_payload(sock, environment.to_dict())


# def send_payload(sock: socket.socket, payload):
#     dumped = json.dumps(payload)
#     message_bytes = dumped.encode("utf-8")
#     base64_bytes = base64.b64encode(message_bytes)
#     base64_message = base64_bytes.decode("utf-8")
#     sock.send(bytes(base64_message + "\n", "utf-8"))


def decode_response(buffered_reader: BufferedReader) -> Optional[Dict[str, Any]]:
    try:
        re = buffered_reader.readline()
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
    # buffered_reader = BufferedReader(sock)
    json_socket = JSONSocket(sock)
    img_seq: int = 0
    # send initial environment
    initial_env = InitialEnvironment(
        initialInventoryCommands=["minecraft:diamond_sword", "minecraft:shield"],
        initialPosition=None,  # nullable
        initialMobsCommands=["minecraft:sheep"],
        imageSizeX=800,
        imageSizeY=600,
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
        random_action = random.randint(0, 7)  # 0, 1, 2, 3..7
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


# send a single character to 127.0.0.1:8000
if __name__ == "__main__":
    main()
