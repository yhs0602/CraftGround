import base64
import json
import socket

# import pdb
import time
from typing import Dict, Optional, Any

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

    def close(self):
        self.sock.close()
