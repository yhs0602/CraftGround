"""Tests for IPC communication with mocking."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from craftground.buffered_socket import BufferedSocket
from craftground.csv_logger import CsvLogger, LogBackend
from craftground.environment.socket_ipc import SocketIPC
from craftground.exceptions import (
    ConnectionTimeoutError,
    InvalidPortError,
    PortInUseError,
)
from craftground.proto.action_space_pb2 import ActionSpaceMessageV2
from craftground.proto.initial_environment_pb2 import InitialEnvironmentMessage
from craftground.proto.observation_space_pb2 import ObservationSpaceMessage


class TestSocketIPC:
    """Test SocketIPC functionality with mocking."""

    def test_check_port_validation(self):
        """Test port number validation."""
        logger = Mock(spec=CsvLogger)
        initial_env = Mock(spec=InitialEnvironmentMessage)

        # Invalid port (too low)
        with pytest.raises(InvalidPortError):
            SocketIPC(logger, initial_env, port=0, find_free_port=False)

        # Invalid port (too high)
        with pytest.raises(InvalidPortError):
            SocketIPC(logger, initial_env, port=70000, find_free_port=False)

        # Invalid type
        with pytest.raises(TypeError):
            SocketIPC(logger, initial_env, port="8000", find_free_port=False)

    @patch("socket.socket")
    def test_send_action(self, mock_socket_class):
        """Test sending action through IPC."""
        logger = Mock(spec=CsvLogger)
        initial_env = Mock(spec=InitialEnvironmentMessage)
        ipc = SocketIPC(logger, initial_env, port=8000, find_free_port=False)

        # Mock socket
        mock_sock = MagicMock()
        ipc.sock = mock_sock

        # Create action
        action = ActionSpaceMessageV2()
        action.attack = True
        action.forward = True

        # Send action
        ipc.send_action(action, commands=["test_command"])

        # Verify socket was called
        assert mock_sock.send.called
        assert mock_sock.sendall.called
        assert logger.log.called

    @patch("socket.socket")
    def test_send_action_no_commands(self, mock_socket_class):
        """Test sending action without commands."""
        logger = Mock(spec=CsvLogger)
        initial_env = Mock(spec=InitialEnvironmentMessage)
        ipc = SocketIPC(logger, initial_env, port=8000, find_free_port=False)

        # Mock socket
        mock_sock = MagicMock()
        ipc.sock = mock_sock

        # Create action
        action = ActionSpaceMessageV2()

        # Send action without commands
        ipc.send_action(action, commands=None)

        # Verify socket was called
        assert mock_sock.send.called
        assert mock_sock.sendall.called

    def test_is_alive(self):
        """Test is_alive method."""
        logger = Mock(spec=CsvLogger)
        initial_env = Mock(spec=InitialEnvironmentMessage)
        ipc = SocketIPC(logger, initial_env, port=8000, find_free_port=False)

        # Initially not alive
        assert not ipc.is_alive()

        # After setting socket, should be alive
        ipc.sock = MagicMock()
        assert ipc.is_alive()
