from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from craftground.environment.observation_converter import (
    ObservationConverter,
    ObservationTensorType,
)
from craftground.proto.observation_space_pb2 import ObservationSpaceMessage
from craftground.screen_encoding_modes import ScreenEncodingMode


@pytest.mark.parametrize(
    "tensor_type,channels",
    [(ObservationTensorType.APPLE_TENSOR, [2, 1, 0]),
     (ObservationTensorType.CUDA_DLPACK, [0, 1, 2])],
)
def test_render_uses_processed_zero_copy_observation(tensor_type, channels):
    torch = pytest.importorskip("torch")
    converter = ObservationConverter(
        ScreenEncodingMode.ZEROCOPY_TORCH, 2, 2, MagicMock(), render_mode="rgb_array"
    )
    backing = torch.arange(16, dtype=torch.uint8).reshape(2, 2, 4)

    def import_frame(_):
        converter.internal_type = tensor_type
        converter._shared_frame = backing

    observation = ObservationSpaceMessage()
    with patch.object(converter, "initialize_zerocopy", side_effect=import_frame) as attach:
        first, _ = converter.convert(observation)
        expected = backing.flip(0)[..., channels].numpy()
        np.testing.assert_array_equal(converter.render(), expected)

        backing.add_(16)
        second, _ = converter.convert(observation)
        np.testing.assert_array_equal(converter.render(), expected + 16)
        np.testing.assert_array_equal(first.numpy(), expected)
        np.testing.assert_array_equal(second.numpy(), expected + 16)
        attach.assert_called_once()

    converter.release_shared_frame()
    assert converter.render() is None
