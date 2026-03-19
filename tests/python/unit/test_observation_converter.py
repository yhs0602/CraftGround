import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

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
