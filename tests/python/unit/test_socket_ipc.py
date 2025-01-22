import signal
import pytest
from unittest.mock import MagicMock, patch
import struct

from craftground.buffered_socket import BufferedSocket
from craftground.csv_logger import CsvLogger
from craftground.environment.socket_ipc import SocketIPC
from craftground.proto.action_space_pb2 import ActionSpaceMessageV2
from craftground.proto.initial_environment_pb2 import InitialEnvironmentMessage
from craftground.proto.observation_space_pb2 import ObservationSpaceMessage


@pytest.fixture
def mock_logger():
    return MagicMock(spec=CsvLogger)


@pytest.fixture
def mock_initial_environment():
    return MagicMock(spec=InitialEnvironmentMessage)


@pytest.fixture
def socket_ipc(mock_logger, mock_initial_environment):
    return SocketIPC(
        logger=mock_logger, initial_environment=mock_initial_environment, port=8000
    )


@patch("os.path.exists")
@patch("os.name", "posix")
def test_check_port_unix(mock_exists, socket_ipc):
    mock_exists.side_effect = lambda path: path == "/tmp/minecraftrl_8000.sock"

    socket_ipc.find_free_port = False
    with pytest.raises(FileExistsError):
        socket_ipc.check_port(8000)

    socket_ipc.find_free_port = True
    port = socket_ipc.check_port(8000)
    assert port > 8000


@patch("socket.socket")
def test_send_initial_environment(mock_socket, socket_ipc, mock_initial_environment):
    mock_sock = MagicMock()
    mock_socket.return_value = mock_sock
    socket_ipc.sock = mock_sock

    socket_ipc._send_initial_environment(mock_initial_environment)

    mock_sock.send.assert_called()
    mock_sock.sendall.assert_called()


@patch("socket.socket")
def test_send_action(mock_socket, socket_ipc):
    mock_sock = MagicMock()
    mock_socket.return_value = mock_sock
    socket_ipc.sock = mock_sock

    action = MagicMock(spec=ActionSpaceMessageV2)
    action.commands = []
    socket_ipc.send_action(action, ["command1", "command2"])

    assert "command1" in action.commands
    assert "command2" in action.commands
    mock_sock.send.assert_called()
    mock_sock.sendall.assert_called()


@patch("socket.socket")
def test_read_observation(mock_socket, socket_ipc):
    mock_sock = MagicMock()
    mock_socket.return_value = mock_sock
    socket_ipc.buffered_socket = MagicMock(spec=BufferedSocket)

    example_observation = ObservationSpaceMessage().SerializeToString()
    socket_ipc.buffered_socket.read.side_effect = [
        struct.pack("<I", len(example_observation)),
        example_observation,
    ]

    observation = socket_ipc.read_observation()

    assert isinstance(observation, ObservationSpaceMessage)


@patch("os.remove")
@patch("os.kill")
def test_remove_orphan_java_processes(mock_kill, mock_remove, socket_ipc):
    mock_remove.side_effect = lambda path: path == "/tmp/minecraftrl_8000.sock"
    with patch("psutil.process_iter") as mock_process_iter:
        mock_process = MagicMock()
        mock_process.open_files.return_value = [
            MagicMock(path="/tmp/minecraftrl_8000.sock")
        ]
        mock_process.info = {"name": "java", "pid": 1234}
        mock_process_iter.return_value = [mock_process]

        socket_ipc.remove_orphan_java_processes()

        mock_kill.assert_called_with(1234, signal.SIGTERM)


@patch("craftground.buffered_socket.socket.socket")
def test_start_communication(mock_socket, socket_ipc):
    mock_sock = MagicMock()
    mock_socket.return_value = mock_sock
    socket_ipc.sock = mock_sock
    socket_ipc.buffered_socket = MagicMock(spec=BufferedSocket)

    socket_ipc.start_communication()

    mock_sock.send.assert_called()
    mock_sock.sendall.assert_called()


@patch("socket.socket")
def test_send_commands(mock_socket, socket_ipc):
    mock_sock = MagicMock()
    mock_socket.return_value = mock_sock
    socket_ipc.sock = mock_sock

    socket_ipc.send_commands(["command1", "command2"])

    mock_sock.send.assert_called()
    mock_sock.sendall.assert_called()


@patch("socket.socket")
def test_send_fastreset2(mock_socket, socket_ipc):
    mock_sock = MagicMock()
    mock_socket.return_value = mock_sock
    socket_ipc.sock = mock_sock

    socket_ipc.send_fastreset2(["extra_command"])

    mock_sock.send.assert_called()
    mock_sock.sendall.assert_called()


@patch("socket.socket")
def test_send_respawn2(mock_socket, socket_ipc):
    mock_sock = MagicMock()
    mock_socket.return_value = mock_sock
    socket_ipc.sock = mock_sock

    socket_ipc.send_respawn2()

    mock_sock.send.assert_called()
    mock_sock.sendall.assert_called()


@patch("socket.socket")
def test_send_exit(mock_socket, socket_ipc):
    mock_sock = MagicMock()
    mock_socket.return_value = mock_sock
    socket_ipc.sock = mock_sock

    socket_ipc.send_exit()

    mock_sock.send.assert_called()
    mock_sock.sendall.assert_called()


@patch("socket.socket")
def test_connect_server(mock_socket, socket_ipc):
    mock_sock = MagicMock()
    mock_socket.return_value = mock_sock
    socket_ipc._connect_server()

    assert socket_ipc.sock is not None
