package com.kyhsgeekcode.minecraftenv

import com.google.common.io.LittleEndianDataOutputStream
import com.kyhsgeekcode.minecraftenv.proto.ActionSpace
import com.kyhsgeekcode.minecraftenv.proto.InitialEnvironment
import com.kyhsgeekcode.minecraftenv.proto.ObservationSpace
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.SocketTimeoutException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.SocketChannel

interface MessageIO {
    fun readAction(): ActionSpace.ActionSpaceMessageV2

    fun readInitialEnvironment(): InitialEnvironment.InitialEnvironmentMessage

    /**
     * Sent once per session, right after [readInitialEnvironment] succeeds, so Python can reject
     * an incompatible server immediately instead of hanging on the first read_observation() call
     * or finding out only when Java crashes mid-init (docs/26_2_MigrationPlan.md item (f)).
     */
    fun writeHandshakeAck(ack: InitialEnvironment.HandshakeAck)

    fun writeObservation(observation: com.kyhsgeekcode.minecraftenv.proto.ObservationSpace.ObservationSpaceMessage)
}

class TCPSocketMessageIO(
    private val socket: java.net.Socket,
) : MessageIO {
    private val outputStream = socket.getOutputStream()
    private val inputStream = socket.getInputStream()

    override fun readAction(): ActionSpace.ActionSpaceMessageV2 {
        printWithTime("Reading action space")
        // read action from inputStream using protobuf
        val buffer = ByteBuffer.allocate(Integer.BYTES) // 4 bytes
        inputStream.read(buffer.array())
        val len = buffer.order(ByteOrder.LITTLE_ENDIAN).int
        val bytes = inputStream.readNBytes(len)
        //        println("Read action space bytes $len")
        val actionSpace = ActionSpace.ActionSpaceMessageV2.parseFrom(bytes)
        printWithTime("Read action space")
        return actionSpace
    }

    override fun readInitialEnvironment(): InitialEnvironment.InitialEnvironmentMessage {
        while (true) {
            try {
                printWithTime("Reading initial environment")
                // read a single int from input stream
                val buffer = ByteBuffer.allocate(Integer.BYTES) // 4 bytes
                inputStream.read(buffer.array())
                val len = buffer.order(ByteOrder.LITTLE_ENDIAN).int
                printWithTime("$len")
                val bytes = inputStream.readNBytes(len.toInt())
                val initialEnvironment =
                    InitialEnvironment.InitialEnvironmentMessage.parseFrom(bytes)
                printWithTime(
                    "Read initial environment ${initialEnvironment!!.imageSizeX} ${initialEnvironment!!.imageSizeY}",
                )
                return initialEnvironment
            } catch (e: SocketTimeoutException) {
                println("Socket timeout")
                // wait and try again
            } catch (e: IOException) {
                e.printStackTrace()
                throw RuntimeException(e)
            }
        }
    }

    override fun writeHandshakeAck(ack: InitialEnvironment.HandshakeAck) {
        printWithTime("Writing handshake ack with size ${ack.serializedSize}")
        val dataOutputStream = LittleEndianDataOutputStream(outputStream)
        dataOutputStream.writeInt(ack.serializedSize)
        ack.writeTo(outputStream)
        outputStream.flush()
        printWithTime("Flushed handshake ack")
    }

    override fun writeObservation(observationSpace: ObservationSpace.ObservationSpaceMessage) {
        printWithTime("Writing observation with size ${observationSpace.serializedSize}")
        val dataOutputStream = LittleEndianDataOutputStream(outputStream)
        dataOutputStream.writeInt(observationSpace.serializedSize)
        //        println("Wrote observation size ${observationSpace.serializedSize}")
        observationSpace.writeTo(outputStream)
        //        println("Wrote observation ${observationSpace.serializedSize}")
        outputStream.flush()
        printWithTime("Flushed")
    }
}

class DomainSocketMessageIO(
    private val socketChannel: SocketChannel,
) : MessageIO {
    override fun readAction(): ActionSpace.ActionSpaceMessageV2 {
        printWithTime("Reading action space")
        // read action from inputStream using protobuf
        val buffer = ByteBuffer.allocate(Integer.BYTES) // 4 bytes
        socketChannel.fillBuffer(buffer)
        buffer.flip()
        val len = buffer.order(ByteOrder.LITTLE_ENDIAN).int
        val bytes = socketChannel.readNBytes(len)
        //        println("Read action space bytes $len")
        val actionSpaceMessageV2 = ActionSpace.ActionSpaceMessageV2.parseFrom(bytes)
        printWithTime("Read action space")
        return actionSpaceMessageV2
    }

    override fun readInitialEnvironment(): InitialEnvironment.InitialEnvironmentMessage {
        // read client environment settings
        while (true) {
            try {
                printWithTime("Reading initial environment")
                // read a single int from input stream
                val buffer = ByteBuffer.allocate(Integer.BYTES) // 4 bytes
                socketChannel.fillBuffer(buffer)
                buffer.flip()
                val len = buffer.order(ByteOrder.LITTLE_ENDIAN).int
                printWithTime("$len")
                val dataBuffer = socketChannel.readNBytes(len)
                val initialEnvironment =
                    InitialEnvironment.InitialEnvironmentMessage.parseFrom(dataBuffer)
                printWithTime(
                    "Read initial environment ${initialEnvironment.imageSizeX} ${initialEnvironment.imageSizeY}",
                )
                return initialEnvironment
            } catch (e: SocketTimeoutException) {
                println("Socket timeout")
                // wait and try again
            } catch (e: IOException) {
                e.printStackTrace()
                throw RuntimeException(e)
            }
        }
    }

    override fun writeHandshakeAck(ack: InitialEnvironment.HandshakeAck) {
        printWithTime("Writing handshake ack with size ${ack.serializedSize}")
        val bufferSize = 4 + ack.serializedSize
        val buffer = ByteBuffer.allocate(bufferSize).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(ack.serializedSize)
        val byteArrayOutputStream = ByteArrayOutputStream()
        ack.writeTo(byteArrayOutputStream)
        buffer.put(byteArrayOutputStream.toByteArray())
        buffer.flip()
        while (buffer.hasRemaining()) {
            socketChannel.write(buffer)
        }
        printWithTime("Flushed handshake ack")
    }

    override fun writeObservation(observation: ObservationSpace.ObservationSpaceMessage) {
        printWithTime("Writing observation with size ${observation.serializedSize}")
        val bufferSize = 4 + observation.serializedSize
        val buffer = ByteBuffer.allocate(bufferSize).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(observation.serializedSize)
        val byteArrayOutputStream = ByteArrayOutputStream()
        observation.writeTo(byteArrayOutputStream)
        buffer.put(byteArrayOutputStream.toByteArray())
        // to read mode
        buffer.flip()
        // Write buffer to SocketChannel
        while (buffer.hasRemaining()) {
            socketChannel.write(buffer)
        }
        printWithTime("Flushed")
    }
}

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

class NamedPipeMessageIO(
    private val readPipePath: String,
    private val writePipePath: String,
) : MessageIO {
    override fun readAction(): ActionSpace.ActionSpaceMessageV2 {
        FileInputStream(readPipePath).use { fis ->
            DataInputStream(fis).use { dis ->
                val len = dis.readInt()
                val bytes = dis.readNBytes(len)
                return ActionSpace.ActionSpaceMessageV2.parseFrom(bytes)
            }
        }
    }

    override fun readInitialEnvironment(): InitialEnvironment.InitialEnvironmentMessage {
        FileInputStream(readPipePath).use { fis ->
            DataInputStream(fis).use { dis ->
                val len = dis.readInt()
                val bytes = dis.readNBytes(len)
                return InitialEnvironment.InitialEnvironmentMessage.parseFrom(bytes)
            }
        }
    }

    override fun writeHandshakeAck(ack: InitialEnvironment.HandshakeAck) {
        FileOutputStream(writePipePath).use { fos ->
            LittleEndianDataOutputStream(fos).use { dos ->
                dos.writeInt(ack.serializedSize)
                ack.writeTo(dos)
            }
        }
    }

    override fun writeObservation(observation: ObservationSpace.ObservationSpaceMessage) {
        FileOutputStream(writePipePath).use { fos ->
            LittleEndianDataOutputStream(fos).use { dos ->
                dos.writeInt(observation.serializedSize)
                observation.writeTo(dos)
            }
        }
    }
}

class SharedMemoryMessageIO(
    val port: Int,
) : MessageIO {
    val shmemPrefix =
        if (System.getProperty("os.name").contains("Windows", ignoreCase = true)) {
            "Global\\"
        } else {
            "/"
        }
    private val p2jMemoryName = "${shmemPrefix}craftground_${port}_p2j"
    private val j2pMemoryName = "${shmemPrefix}craftground_${port}_j2p"

    override fun readAction(): ActionSpace.ActionSpaceMessageV2 = FramebufferCapturer.readAction(p2jMemoryName)

    override fun readInitialEnvironment(): InitialEnvironment.InitialEnvironmentMessage =
        FramebufferCapturer.readInitialEnvironment(p2jMemoryName, port)

    // Not wired up on the Python side yet: BoostIPC (the shared-memory IPCInterface) has no
    // pre-read_observation() step to consume a standalone ack from the j2p channel, and writing
    // one there would just corrupt the first real observation read. Left a no-op rather than
    // silently breaking BoostIPC until that Python-side read is added - see
    // docs/26_2_MigrationPlan.md item (f) for the TCP/domain-socket path that IS wired up.
    override fun writeHandshakeAck(ack: InitialEnvironment.HandshakeAck) {}

    override fun writeObservation(observation: ObservationSpace.ObservationSpaceMessage) =
        FramebufferCapturer.writeObservation(p2jMemoryName, j2pMemoryName, observation)
}
