from enum import Enum
import io
from typing import TYPE_CHECKING, Optional, Tuple, Union

from proto.observation_space_pb2 import ObservationSpaceMessage
from screen_encoding_modes import ScreenEncodingMode
import numpy as np
from PIL import Image


class ObservationTensorType(Enum):
    NONE = 0
    CUDA_DLPACK = 1
    APPLE_TENSOR = 2
    JAX_NP = 3


if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp

    TorchArrayType = torch.Tensor
    JaxArrayType = jnp.ndarray
ImageOutputType = Union[np.ndarray, "TorchArrayType", "JaxArrayType"]


class ObservationConverter:
    def __init__(
        self, output_type: ScreenEncodingMode, is_binocular: bool = False
    ) -> None:
        self.output_type = output_type
        self.internal_type = ObservationTensorType.NONE
        self.last_observations = [None, None]
        self.last_images = [None, None]
        self.is_binocular = is_binocular

    def convert(
        self, observation: ObservationSpaceMessage
    ) -> Tuple[Optional[ImageOutputType]]:
        if self.output_type == ScreenEncodingMode.PNG:
            obs_1, obs_2 = None, None
            img_1, img_2 = None, None
            obs_1, img_1 = self.convert_png_observation(observation.image)
            if self.is_binocular:
                obs_2, img_2 = self.convert_png_observation(observation.image_2)
            self.last_observations[0], self.last_observations[1] = obs_1, obs_2
            self.last_images[0], self.last_images[1] = img_1, img_2
            return (obs_1, obs_2)
        elif self.output_type == ScreenEncodingMode.RAW:
            obs_1, obs_2 = None, None
            img_1, img_2 = None, None
            obs_1 = self.convert_raw_observation(observation.image)
            if self.is_binocular:
                obs_2 = self.convert_raw_observation(observation.image_2)
            self.last_observations[0], self.last_observations[1] = obs_1, obs_2
            return (obs_1, obs_2)
        elif self.output_type == ScreenEncodingMode.ZEROCOPY:
            if self.is_binocular:
                raise ValueError("Zerocopy mode does not support binocular vision")
            if self.internal_type == ObservationTensorType.APPLE_TENSOR:
                pass
            elif self.internal_type == ObservationTensorType.CUDA_DLPACK:
                pass
            elif self.internal_type == ObservationTensorType.NONE:
                obs_1 = self.convert_torch_zerocopy(observation.image)
            else:
                raise ValueError(
                    f"Invalid internal type for output {self.output_type}: {self.internal_type}"
                )

        elif self.output_type == ScreenEncodingMode.JAX:
            return self.convert_jax_zerocopy(observation)
        else:
            raise ValueError(f"Unknown output type: {self.output_type}")

    def render():
        pass

    def convert_png_observation(
        self, image_bytes: bytes
    ) -> Tuple[np.ndarray, Image.Image]:
        # decode png byte array to numpy array
        # Create a BytesIO object from the byte array
        with self.csv_logger.profile("convert_observation/decode_png"):
            bytes_io = io.BytesIO(image_bytes)
            # Use PIL to open the image from the BytesIO object
            img = Image.open(bytes_io).convert("RGB")
            # Flip y axis
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        with self.csv_logger.profile("convert_observation/convert_to_numpy"):
            # Convert the PIL image to a numpy array
            last_rgb_frame = np.array(img)
            arr = np.transpose(last_rgb_frame, (2, 1, 0))
            rgb_array_or_tensor = arr.astype(np.uint8)

        return rgb_array_or_tensor, img

    def convert_raw_observation(self, image_bytes: bytes) -> np.ndarray:
        # decode raw byte array to numpy array
        with self.csv_logger.profile("convert_observation/decode_raw"):
            last_rgb_frame = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
                (self.initial_env.imageSizeY, self.initial_env.imageSizeX, 3)
            )
            # Flip y axis using np
            last_rgb_frame = np.flip(last_rgb_frame, axis=0)
            rgb_array_or_tensor = last_rgb_frame
            # arr = np.transpose(last_rgb_frame, (2, 1, 0))  # channels, width, height
        return rgb_array_or_tensor

    def convert_torch_zerocopy(
        self, observation: Union[bytes]
    ) -> Tuple[np.ndarray, Optional[Image.Image]]:
        pass
