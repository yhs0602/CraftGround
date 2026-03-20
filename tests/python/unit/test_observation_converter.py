import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from craftground.environment.observation_converter import ObservationConverter
from craftground.environment.observation_converter import ObservationTensorType
from craftground.screen_encoding_modes import ScreenEncodingMode


def test_macos_zerocopy_initializes_raw_tensor_and_returns_normalized_output():
    raw_bgr = torch.tensor(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=torch.uint8,
    )
    normalized = torch.tensor(
        [
            [[90, 80, 70], [120, 110, 100]],
            [[30, 20, 10], [60, 50, 40]],
        ],
        dtype=torch.uint8,
    )
    normalize_calls = []
    fake_native = SimpleNamespace(
        initialize_from_mach_port=lambda mach_port, width, height: raw_bgr,
        mtl_tensor_from_cuda_mem_handle=lambda *args, **kwargs: None,
        normalize_apple_mtl_tensor=lambda tensor: (
            normalize_calls.append(tensor) or normalized
        ),
    )
    observation = SimpleNamespace(ipc_handle=(1234).to_bytes(4, byteorder="little"))
    logger = MagicMock()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(
            sys.modules,
            "craftground.environment.craftground_native",
            fake_native,
        )

        converter = ObservationConverter(
            output_type=ScreenEncodingMode.ZEROCOPY,
            image_width=2,
            image_height=2,
            logger=logger,
        )

        obs, _ = converter.convert(observation)

    assert converter.last_observations[0] is raw_bgr
    assert converter.internal_type == ObservationTensorType.APPLE_TENSOR
    assert normalize_calls == [raw_bgr]
    assert torch.equal(obs, normalized)


def test_macos_zerocopy_falls_back_to_python_normalization_without_native_helper():
    raw_bgr = torch.tensor(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=torch.uint8,
    )
    fake_native = SimpleNamespace(
        initialize_from_mach_port=lambda mach_port, width, height: raw_bgr,
        mtl_tensor_from_cuda_mem_handle=lambda *args, **kwargs: None,
    )
    observation = SimpleNamespace(ipc_handle=(1234).to_bytes(4, byteorder="little"))
    logger = MagicMock()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(
            sys.modules,
            "craftground.environment.craftground_native",
            fake_native,
        )

        converter = ObservationConverter(
            output_type=ScreenEncodingMode.ZEROCOPY,
            image_width=2,
            image_height=2,
            logger=logger,
        )

        obs, _ = converter.convert(observation)

    expected = torch.tensor(
        [
            [[90, 80, 70], [120, 110, 100]],
            [[30, 20, 10], [60, 50, 40]],
        ],
        dtype=torch.uint8,
    )

    assert converter.last_observations[0] is raw_bgr
    assert converter.internal_type == ObservationTensorType.APPLE_TENSOR
    assert torch.equal(obs, expected)


def test_jax_mach_port_returns_normalized_tensor_and_caches_it():
    raw_bgra = np.array(
        [
            [[10, 20, 30, 255], [40, 50, 60, 255]],
            [[70, 80, 90, 255], [100, 110, 120, 255]],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [[90, 80, 70], [120, 110, 100]],
            [[30, 20, 10], [60, 50, 40]],
        ],
        dtype=np.uint8,
    )
    fake_jnp = SimpleNamespace(
        from_dlpack=lambda capsule: raw_bgra.copy(),
        flip=lambda array, axis: np.flip(array, axis=axis).copy(),
    )
    fake_native = SimpleNamespace(
        mtl_dlpack_from_mach_port=lambda mach_port, width, height: object(),
        mtl_tensor_from_cuda_mem_handle=lambda *args, **kwargs: None,
    )
    observation = SimpleNamespace(ipc_handle=(1234).to_bytes(4, byteorder="little"))
    logger = MagicMock()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(numpy=fake_jnp))
        monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
        monkeypatch.setitem(
            sys.modules,
            "craftground.environment.craftground_native",
            fake_native,
        )

        converter = ObservationConverter(
            output_type=ScreenEncodingMode.JAX,
            image_width=2,
            image_height=2,
            logger=logger,
        )

        first, _ = converter.convert(observation)
        second, _ = converter.convert(observation)

    assert converter.internal_type == ObservationTensorType.JAX_NP
    assert np.array_equal(first, expected)
    assert np.array_equal(second, expected)
    assert np.array_equal(converter.last_observations[0], expected)


def test_jax_cuda_handle_returns_normalized_tensor_and_caches_it():
    raw_rgba = np.array(
        [
            [[1, 2, 3, 255], [4, 5, 6, 255]],
            [[7, 8, 9, 255], [10, 11, 12, 255]],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [[7, 8, 9], [10, 11, 12]],
            [[1, 2, 3], [4, 5, 6]],
        ],
        dtype=np.uint8,
    )
    fake_jnp = SimpleNamespace(
        from_dlpack=lambda capsule: raw_rgba.copy(),
        flip=lambda array, axis: np.flip(array, axis=axis).copy(),
    )
    fake_native = SimpleNamespace(
        mtl_dlpack_from_mach_port=lambda *args, **kwargs: None,
        mtl_tensor_from_cuda_mem_handle=lambda *args, **kwargs: object(),
    )
    observation = SimpleNamespace(ipc_handle=b"cuda-handle")
    logger = MagicMock()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(numpy=fake_jnp))
        monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
        monkeypatch.setitem(
            sys.modules,
            "craftground.environment.craftground_native",
            fake_native,
        )

        converter = ObservationConverter(
            output_type=ScreenEncodingMode.JAX,
            image_width=2,
            image_height=2,
            logger=logger,
        )

        first, _ = converter.convert(observation)
        second, _ = converter.convert(observation)

    assert converter.internal_type == ObservationTensorType.JAX_NP
    assert np.array_equal(first, expected)
    assert np.array_equal(second, expected)
    assert np.array_equal(converter.last_observations[0], expected)
