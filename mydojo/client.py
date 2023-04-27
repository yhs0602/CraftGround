import base64
import random
import socket

from initial_environment import InitialEnvironment
from mydojo.json_socket import JSONSocket
from mydojo.minecraft import wait_for_server, int_to_action, send_action


# import pdb


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
        isHardCore=False,
    )
    json_socket.send_json_as_base64(initial_env.to_dict())
    print("Sent initial environment")
    # send_initial_environment(sock, initial_env)

    while True:
        random_action = random.randint(0, 8)
        action_arr = int_to_action(random_action)

        # send the action
        # print("Sending action...")
        send_action(json_socket, action_arr)
        # read the response
        # print("Reading response...")
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
