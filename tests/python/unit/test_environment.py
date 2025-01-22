import signal
import pytest
from unittest.mock import MagicMock, patch


from craftground.environment.environment import CraftGroundEnvironment
from craftground.environment.observation_converter import ObservationConverter
from craftground.environment.socket_ipc import SocketIPC
from craftground.initial_environment_config import InitialEnvironmentConfig
from craftground.screen_encoding_modes import ScreenEncodingMode
from craftground.environment.boost_ipc import BoostIPC


@pytest.fixture
def mock_initial_env():
    return InitialEnvironmentConfig(
        image_width=640,
        image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        eye_distance=0,
    )


@pytest.fixture
def environment(mock_initial_env):
    return CraftGroundEnvironment(initial_env=mock_initial_env)


@patch("craftground.environment.environment.SocketIPC")
def test_initialize_socket_environment(mock_ipc_class, mock_initial_env):
    mock_ipc_instance = MagicMock(spec=SocketIPC)
    mock_ipc_class.return_value = mock_ipc_instance

    assert mock_ipc_instance is not None
    print(mock_initial_env)
    env = CraftGroundEnvironment(initial_env=mock_initial_env, use_shared_memory=False)

    assert env.initial_env == mock_initial_env
    assert env.action_space is not None
    assert env.observation_space is not None
    assert isinstance(env.observation_converter, ObservationConverter)
    assert isinstance(env.ipc, SocketIPC)


@patch("craftground.environment.boost_ipc.BoostIPC")
def test_initialize_boost_environment(mock_ipc_class, mock_initial_env):
    mock_ipc_instance = MagicMock(spec=BoostIPC)
    mock_ipc_class.return_value = mock_ipc_instance

    assert mock_ipc_instance is not None
    print(mock_initial_env)
    env = CraftGroundEnvironment(initial_env=mock_initial_env, use_shared_memory=True)

    assert env.initial_env == mock_initial_env
    assert env.action_space is not None
    assert env.observation_space is not None
    assert isinstance(env.observation_converter, ObservationConverter)
    assert isinstance(env.ipc, BoostIPC)


@patch("socket.socket")
@patch("subprocess.Popen")
def test_start_server(mock_popen, mock_socket, environment):
    mock_process = MagicMock()
    mock_popen.return_value = mock_process
    mock_sock_instance = MagicMock()
    mock_socket.return_value = mock_sock_instance

    environment.start_server(seed=1234)

    assert mock_popen.called
    assert environment.process is not None


@patch("builtins.open", new_callable=MagicMock)
def test_update_override_resolutions(mock_open, environment):
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    options_path = environment.get_env_option_path()

    mock_file.read.return_value = "overrideWidth:800\noverrideHeight:600"

    environment.update_override_resolutions(options_path)

    mock_file.write.assert_called()


@patch("craftground.environment.environment.CraftGroundEnvironment.reset")
def test_reset_environment(mock_reset, environment):
    mock_reset.return_value = ({"observation": "data"}, {})

    obs, info = environment.reset()

    assert "observation" in obs
    assert isinstance(info, dict)


@patch("craftground.environment.environment.CraftGroundEnvironment.step")
def test_step_action(mock_step, environment):
    mock_step.return_value = ("observation", 1.0, False, False, {})

    obs, reward, done, truncated, info = environment.step(action=0)

    assert obs == "observation"
    assert reward == 1.0
    assert not done
    assert not truncated
    assert isinstance(info, dict)


@patch("craftground.environment.environment.CraftGroundEnvironment.close")
def test_close_environment(mock_close, environment):
    environment.close()
    mock_close.assert_called()


@patch("os.getpgid")
@patch("os.killpg")
def test_terminate_environment(mock_kill, mock_getpgid, environment):
    environment.process = MagicMock()
    environment.process.pid = 1234
    mock_getpgid.return_value = -1234

    environment.terminate()

    mock_getpgid.assert_called_with(1234)
    mock_kill.assert_called_with(-1234, signal.SIGKILL)
