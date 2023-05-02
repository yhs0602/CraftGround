import time
import socket


class BufferedSocket:
    def __init__(self, sock: socket.socket):
        self.socket = sock
        self.buffer = b""

    def read(self, n: int, wait=False) -> bytes:
        # print("Reading", n, "bytes")
        if n == 0:
            return b""
        data = self.buffer
        while len(data) < n:
            try:
                chunk = self.socket.recv(n - len(data))
                if not chunk:
                    break
                data += chunk
            except socket.timeout:
                if wait:  # wait for more data
                    # print("Waiting")
                    time.sleep(0.01)
                    continue
                else:
                    raise
        result, self.buffer = data[:n], data[n:]
        return result
