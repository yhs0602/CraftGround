import base64
import json
import random

# import pdb
import socket
import time


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


# send a single character to 127.0.0.1:8000
if __name__ == "__main__":
    # pdb.set_trace()
    sock: socket.socket = wait_for_server()
    img_seq: int = 0
    while True:
        # send a single character
        random_action = random.randint(0, 9)  # 0, 1, 2, 3..9
        sock.send(bytes(str(random_action), "utf-8"))
        # read the response
        re = recvall(sock)
        print(len(re))
        sr = re.decode("utf-8")
        data = json.loads(sr)
        print(data["x"], data["y"], data["z"])
        img = data["image"]  # wxh|base64
        decoded_img = base64.b64decode(img)
        # save this png byte array to a file
        with open(f"{img_seq}.png", "wb") as f:
            f.write(decoded_img)
            img_seq += 1

    sock.close()
