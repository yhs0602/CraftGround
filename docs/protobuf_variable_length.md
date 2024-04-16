# Enhancing Cross-Language Protobuf Protocol for Improved Data Handling

## Introduction
In modern software development, ensuring efficient and reliable data exchange between different programming languages is crucial. This project presents a significant enhancement to the protobuf protocol used between Python and Kotlin, addressing critical issues in the existing implementation and introducing a lightweight solution.

## Problem in Existing Protobuf Python SDK
The previous implementation of the protobuf Python SDK did not transmit the message length separately. This omission made it impossible to read only the message length from a stream, leading to problems in scenarios involving continuous data exchange. Without knowing the exact size of incoming messages, the system was prone to errors from redundant data, especially in persistent connections.

Actually, there are methods like `parseFromDelimited` and `writeToDelimited` in the Java SDK that can be used to read and write length-delimited messages. However, these methods are not available in the Python SDK.

## Related Posts
Here are some related posts that I found while researching this issue:

- [Similar issues in C#](https://github.com/protocolbuffers/protobuf/issues/4303)
- [Receive delimited Protobuf message in python via TCP](https://stackoverflow.com/questions/43897955/receive-delimited-protobuf-message-in-python-via-tcp/43898459#43898459)
- [Length-Delimited Protobuf Streams](https://seb-nyberg.medium.com/length-delimited-protobuf-streams-a39ebc4a4565)


## Proposed Lightweight Solution
To address the aforementioned issue, I introduced a straightforward yet effective enhancement: prefixing each message with its length. This simple modification ensures that both the sending and receiving ends of a communication link can accurately determine the size of the data being exchanged, thus preventing the transmission of excess data and reducing the likelihood of data corruption.

### Implementation Details
Below are snippets of the code used in both Python and Kotlin to implement this solution:

#### Python (Sender)
```python
import struct
import socket
import action_space_pb2
from typing import List
def send_action2(sock: socket.socket, action_array: List[int]):
    # Make a message
    action_space = action_space_pb2.ActionSpaceMessage()
    action_space.action.extend(action_array)
    action_space.command = ""
    # Serialize the message
    v = action_space.SerializeToString()
    # First send the length of the message
    sock.send(struct.pack("<I", len(v)))
    # Then send the message
    sock.sendall(v)
```
#### Python (Reader)
```python
import struct
import socket
import observation_space_pb2
def read_one_observation(sock: socket.socket) -> (int, ObsType):
    # First read the length of the message
    data_len_bytes = sock.read(4, True)
    # Unpack the length
    data_len = struct.unpack("<I", data_len_bytes)[0]
    # Then read the message
    data_bytes = sock.read(data_len, True)
    # Parse the message
    observation_space = observation_space_pb2.ObservationSpaceMessage()
    observation_space.ParseFromString(data_bytes)
    return data_len, observation_space
```

#### Kotlin(Sender)
```kotlin
fun writeObservation(observationSpace: ObservationSpace.ObservationSpaceMessage) {
    // Prepare buffer for writing
    val bufferSize = 4 + observationSpace.serializedSize
    val buffer = ByteBuffer.allocate(bufferSize).order(ByteOrder.LITTLE_ENDIAN)
    // Write message length
    buffer.putInt(observationSpace.serializedSize)
    // Write message
    val byteArrayOutputStream = ByteArrayOutputStream()
    observationSpace.writeTo(byteArrayOutputStream)
    buffer.put(byteArrayOutputStream.toByteArray())
    // Flip the buffer to read mode (buffer -> channel)
    buffer.flip()
    // Write buffer to SocketChannel
    while (buffer.hasRemaining()) {
        socketChannel.write(buffer)
    }
}
```

#### Kotlin(Reader)
```kotlin
fun readAction(): ActionSpace.ActionSpaceMessage {
    // First read the length of the message
    val buffer = ByteBuffer.allocate(Integer.BYTES) // 4 bytes
    socketChannel.fillBuffer(buffer)
    buffer.flip()
    val len = buffer.order(ByteOrder.LITTLE_ENDIAN).int
    // Then read the message
    val bytes = socketChannel.readNBytes(len)
    // Parse the message
    val actionSpace = ActionSpace.ActionSpaceMessage.parseFrom(bytes)
    return actionSpace
}
```

In Kotlin side, we used an extension function `readNBytes` to read `len` bytes from the socket channel. The `fillBuffer` function is used to fill the buffer with data from the socket channel.

```kotlin
fun SocketChannel.fillBuffer(buffer: ByteBuffer) {
    while (buffer.hasRemaining()) {
        read(buffer)
    }
}

fun SocketChannel.readNBytes(len: Int): ByteArray {
    val buffer = ByteBuffer.allocate(len)
    var readLen = 0
    while (readLen < len) {
        val bytesRead = read(buffer)
        if (bytesRead == -1) {
            throw IOException("EOF")
        }
        readLen += bytesRead
    }
    return buffer.array()
}
```

# Conclusion
This enhancement to the protobuf protocol not only improves the reliability of data transmission between Python and Kotlin but also serves as a model for similar improvements in other cross-language implementations. By ensuring precise data handling and reducing redundancy, this solution contributes to more robust and scalable systems.