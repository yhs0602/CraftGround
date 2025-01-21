import signal
import pytest
from unittest.mock import MagicMock, patch

from environment.environment import CraftGroundEnvironment
from environment.observation_converter import ObservationConverter
from environment.socket_ipc import SocketIPC
from initial_environment_config import InitialEnvironmentConfig


@pytest.fixture
def mock_initial_env():
    return InitialEnvironmentConfig(
        imageSizeX=1280,
        imageSizeY=720,
        screen_encoding_mode="rgb",
        eye_distance=0,
    )


@pytest.fixture
def environment(mock_initial_env):
    return CraftGroundEnvironment(initial_env=mock_initial_env)


@patch("environment.craftground_environment.SocketIPC")
def test_initialize_environment(mock_ipc_class, mock_initial_env):
    mock_ipc_instance = MagicMock()
    mock_ipc_class.return_value = mock_ipc_instance

    env = CraftGroundEnvironment(initial_env=mock_initial_env)

    assert env.initial_env == mock_initial_env
    assert env.action_space is not None
    assert env.observation_space is not None
    assert isinstance(env.observation_converter, ObservationConverter)
    assert isinstance(env.ipc, SocketIPC)
    assert env.sock is None


@patch("subprocess.Popen")
def test_start_server(mock_popen, environment):
    mock_process = MagicMock()
    mock_popen.return_value = mock_process

    environment.start_server(
        port=8000, use_vglrun=False, track_native_memory=False, ld_preload=None
    )

    assert mock_popen.called
    assert environment.process is not None


@patch("socket.socket")
def test_socket_connection(mock_socket, environment):
    mock_sock_instance = MagicMock()
    mock_socket.return_value = mock_sock_instance

    mock_sock_instance.connect.return_value = True

    environment.ipc.wait_for_server = MagicMock(return_value=mock_sock_instance)

    environment.start_server(
        port=8000, use_vglrun=False, track_native_memory=False, ld_preload=None
    )

    assert environment.sock is not None
    assert environment.sock == mock_sock_instance


@patch("builtins.open", new_callable=MagicMock)
def test_update_override_resolutions(mock_open, environment):
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    options_path = environment.get_env_option_path()

    mock_file.read.return_value = "overrideWidth:800\noverrideHeight:600"

    environment.update_override_resolutions(options_path)

    mock_file.write.assert_called()


@patch("environment.craftground_environment.BufferedSocket")
def test_read_one_observation(mock_buffered_socket, environment):
    mock_buffered_socket_instance = MagicMock()
    mock_buffered_socket.return_value = mock_buffered_socket_instance

    mock_buffered_socket_instance.read.side_effect = [
        struct.pack("<I", 1024),
        b"fake_observation_data",
    ]

    environment.buffered_socket = mock_buffered_socket_instance

    length, obs = environment.read_one_observation()

    assert length == 1024
    assert obs is not None


@patch("environment.craftground_environment.CraftGroundEnvironment.reset")
def test_reset_environment(mock_reset, environment):
    mock_reset.return_value = ({"observation": "data"}, {})

    obs, info = environment.reset()

    assert "observation" in obs
    assert isinstance(info, dict)


@patch("environment.craftground_environment.CraftGroundEnvironment.step")
def test_step_action(mock_step, environment):
    mock_step.return_value = ("observation", 1.0, False, False, {})

    obs, reward, done, truncated, info = environment.step(action=0)

    assert obs == "observation"
    assert reward == 1.0
    assert not done
    assert not truncated
    assert isinstance(info, dict)


@patch("environment.craftground_environment.CraftGroundEnvironment.close")
def test_close_environment(mock_close, environment):
    environment.close()
    mock_close.assert_called()


@patch("os.kill")
def test_terminate_environment(mock_kill, environment):
    environment.process = MagicMock()
    environment.process.pid = 1234

    environment.terminate()

    mock_kill.assert_called_with(1234, signal.SIGKILL)
