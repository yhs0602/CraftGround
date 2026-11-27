import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

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


@pytest.mark.parametrize("version", ["1.21", "26.2", "26.3"])
def test_windows_shared_memory_reports_socket_transport(mock_initial_env, monkeypatch, version):
    monkeypatch.setattr("craftground.environment.environment.sys.platform", "win32")
    with pytest.raises(ValueError, match="use_shared_memory=False"):
        CraftGroundEnvironment(mock_initial_env, mc_version=version, use_shared_memory=True)


@pytest.mark.parametrize("mode", [ScreenEncodingMode.ZEROCOPY_TORCH, ScreenEncodingMode.ZEROCOPY_JAX])
@patch("craftground.environment.environment.resolve_runtime_env_path", return_value="unused")
def test_binocular_zero_copy_fails_before_opening_ipc(mock_runtime_path, mode):
    config = InitialEnvironmentConfig(screen_encoding_mode=mode, eye_distance=0.1)
    with patch("craftground.environment.environment.SocketIPC") as ipc:
        with pytest.raises(ValueError, match="binocular"):
            CraftGroundEnvironment(config, mc_version="26.3")
        ipc.assert_not_called()


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


@pytest.mark.skipif(sys.platform == "win32", reason="Shared memory requires POSIX")
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


@pytest.mark.skipif(sys.platform == "win32", reason="Shared memory requires POSIX")
def test_exit_notification_finishes_before_shared_memory_is_destroyed(mock_initial_env):
    publishing, resume, destroying = (threading.Event() for _ in range(3))

    def notify(_):
        publishing.set()
        assert resume.wait(5)

    def destroy():
        destroying.set()
        ipc.destroy()

    with patch(
        "craftground.environment.boost_ipc.initialize_shared_memory", return_value=8000
    ):
        ipc = BoostIPC(
            8000, False, mock_initial_env.to_initial_environment_message(), MagicMock()
        )
    with patch(
        "craftground.craftground_native.notify_shared_memory_exit", side_effect=notify
    ) as post:
        with patch("craftground.environment.boost_ipc.destroy_shared_memory") as unlink:
            with ThreadPoolExecutor(max_workers=2) as pool:
                notified = pool.submit(ipc.notify_process_exit)
                try:
                    assert publishing.wait(5)
                    destroyed = pool.submit(destroy)
                    assert destroying.wait(5)
                    assert not ipc.closed
                    unlink.assert_not_called()
                finally:
                    resume.set()
                notified.result(timeout=5)
                destroyed.result(timeout=5)
            ipc.notify_process_exit()
            post.assert_called_once()
            assert unlink.call_count == 2


@patch("socket.socket")
@patch("subprocess.Popen")
def test_reset(mock_popen, mock_socket, environment):
    mock_process = MagicMock()
    mock_popen.return_value = mock_process
    mock_sock_instance = MagicMock()
    mock_socket.return_value = mock_sock_instance

    with pytest.raises(RuntimeError):
        environment.reset(seed=1234)

    assert mock_popen.called
    # assert mock_socket.called
    # assert environment.process is not None


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
    environment.process.poll.return_value = None
    environment.process.pid = 1234
    environment.process.wait.side_effect = subprocess.TimeoutExpired("java", 10)
    mock_getpgid.return_value = -1234

    environment.terminate()

    mock_getpgid.assert_called_with(1234)
    mock_kill.assert_called_with(-1234, signal.SIGKILL)
