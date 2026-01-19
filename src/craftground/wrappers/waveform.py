"""
Waveform Audio Wrapper for CraftGround

This wrapper provides raw audio waveform observations from Minecraft,
captured via the OpenAL SOFT_loopback extension.

Unlike the subtitle-based SoundWrapper which provides discrete sound events,
this wrapper provides continuous audio signals that can be processed by
audio-based neural networks (e.g., for spectrogram analysis, audio classification).

Usage:
    env = craftground.make(...)
    env = WaveformWrapper(env, output_format="raw")  # or "spectrogram"
    
    obs, info = env.reset()
    # obs["waveform"] contains the audio data
"""

from typing import SupportsFloat, Any, Optional, Literal
import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType


class WaveformWrapper(gym.Wrapper):
    """
    Wrapper that extracts audio waveform data from CraftGround observations.

    The audio data is captured via OpenAL SOFT_loopback extension on the
    Minecraft side and transmitted via Protobuf.

    Attributes:
        sample_rate: Audio sample rate in Hz (typically 44100)
        channels: Number of audio channels (1=mono, 2=stereo)
        buffer_samples: Number of samples per channel in each observation
        output_format: "raw" for PCM data, "spectrogram" for mel spectrogram
    """

    def __init__(
        self,
        env: gym.Env,
        output_format: Literal["raw", "spectrogram", "mfcc"] = "raw",
        normalize: bool = True,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        include_subtitle: bool = False,
        **kwargs,
    ):
        """
        Initialize the WaveformWrapper.

        Args:
            env: The CraftGround environment to wrap
            output_format: Output format for audio data:
                - "raw": Raw PCM samples as float32 array
                - "spectrogram": Mel spectrogram (requires librosa)
                - "mfcc": MFCC features (requires librosa)
            normalize: Whether to normalize audio to [-1, 1] range
            n_mels: Number of mel bands for spectrogram
            n_fft: FFT window size for spectrogram
            hop_length: Hop length for spectrogram
            include_subtitle: Whether to also include subtitle-based sound info
        """
        super().__init__(env)
        self.output_format = output_format
        self.normalize = normalize
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.include_subtitle = include_subtitle

        # Default audio parameters (will be updated from actual data)
        self.sample_rate = 44100
        self.channels = 2
        self.buffer_samples = 2205  # ~50ms at 44100Hz

        # Setup observation space based on output format
        self._setup_observation_space()

        # Lazy load librosa if needed
        self._librosa = None
        if output_format in ["spectrogram", "mfcc"]:
            try:
                import librosa

                self._librosa = librosa
            except ImportError:
                raise ImportError(
                    f"librosa is required for output_format='{output_format}'. "
                    "Install it with: pip install librosa"
                )

    def _setup_observation_space(self):
        """Setup the observation space based on output format."""
        if self.output_format == "raw":
            # Raw PCM: (channels, samples) or (samples,) for mono
            if self.channels == 1:
                shape = (self.buffer_samples,)
            else:
                shape = (self.channels, self.buffer_samples)

            audio_space = gym.spaces.Box(
                low=-1.0 if self.normalize else -32768,
                high=1.0 if self.normalize else 32767,
                shape=shape,
                dtype=np.float32,
            )

        elif self.output_format == "spectrogram":
            # Mel spectrogram: (n_mels, time_frames)
            time_frames = self.buffer_samples // self.hop_length + 1
            audio_space = gym.spaces.Box(
                low=0.0, high=np.inf, shape=(self.n_mels, time_frames), dtype=np.float32
            )

        elif self.output_format == "mfcc":
            # MFCC: (n_mfcc, time_frames)
            time_frames = self.buffer_samples // self.hop_length + 1
            n_mfcc = 13  # Standard number of MFCCs
            audio_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_mfcc, time_frames), dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown output_format: {self.output_format}")

        # Create composite observation space
        spaces = {"waveform": audio_space}

        if self.include_subtitle:
            # Add subtitle info if requested
            # This would need to be coordinated with the existing sound wrapper
            pass

        self.observation_space = gym.spaces.Dict(spaces)

    def _decode_pcm(self, pcm_bytes: bytes) -> np.ndarray:
        """
        Decode raw PCM bytes to numpy array.

        Args:
            pcm_bytes: Raw PCM data (16-bit signed, little-endian)

        Returns:
            Audio samples as float32 array, shape (channels, samples) or (samples,)
        """
        if len(pcm_bytes) == 0:
            # Return silence if no audio data
            if self.channels == 1:
                return np.zeros(self.buffer_samples, dtype=np.float32)
            else:
                return np.zeros((self.channels, self.buffer_samples), dtype=np.float32)

        # Decode 16-bit signed PCM
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Convert to float32
        audio = audio.astype(np.float32)

        # Normalize to [-1, 1] if requested
        if self.normalize:
            audio = audio / 32768.0

        # Reshape for stereo
        if self.channels == 2:
            # Interleaved stereo: L R L R L R ...
            if len(audio) % 2 == 0:
                audio = audio.reshape(-1, 2).T  # Shape: (2, samples)
            else:
                # Handle odd length by padding
                audio = np.pad(audio, (0, 1))
                audio = audio.reshape(-1, 2).T

        return audio

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio."""
        if self._librosa is None:
            raise RuntimeError("librosa not available")

        # Use mono for spectrogram
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Compute mel spectrogram
        mel_spec = self._librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Convert to dB scale
        mel_spec_db = self._librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db.astype(np.float32)

    def _compute_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Compute MFCC features from audio."""
        if self._librosa is None:
            raise RuntimeError("librosa not available")

        # Use mono for MFCC
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Compute MFCCs
        mfcc = self._librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        return mfcc.astype(np.float32)

    def _extract_waveform(self, info: dict) -> np.ndarray:
        """Extract and process waveform from observation info."""
        obs_info = info.get("obs")
        if obs_info is None:
            return self._get_silence()

        # Check if audio waveform is available
        if not hasattr(obs_info, "audio_waveform") or obs_info.audio_waveform is None:
            return self._get_silence()

        audio_waveform = obs_info.audio_waveform

        # Update audio parameters from actual data
        if hasattr(audio_waveform, "sample_rate") and audio_waveform.sample_rate > 0:
            self.sample_rate = audio_waveform.sample_rate
        if hasattr(audio_waveform, "channels") and audio_waveform.channels > 0:
            self.channels = audio_waveform.channels
        if hasattr(audio_waveform, "num_samples") and audio_waveform.num_samples > 0:
            self.buffer_samples = audio_waveform.num_samples

        # Get PCM data
        pcm_data = (
            audio_waveform.pcm_data if hasattr(audio_waveform, "pcm_data") else b""
        )

        # Decode PCM
        audio = self._decode_pcm(pcm_data)

        # Process based on output format
        if self.output_format == "raw":
            return audio
        elif self.output_format == "spectrogram":
            return self._compute_spectrogram(audio)
        elif self.output_format == "mfcc":
            return self._compute_mfcc(audio)
        else:
            return audio

    def _get_silence(self) -> np.ndarray:
        """Return silence array matching the expected shape."""
        if self.output_format == "raw":
            if self.channels == 1:
                return np.zeros(self.buffer_samples, dtype=np.float32)
            else:
                return np.zeros((self.channels, self.buffer_samples), dtype=np.float32)
        elif self.output_format == "spectrogram":
            time_frames = self.buffer_samples // self.hop_length + 1
            return np.zeros((self.n_mels, time_frames), dtype=np.float32)
        elif self.output_format == "mfcc":
            time_frames = self.buffer_samples // self.hop_length + 1
            return np.zeros((13, time_frames), dtype=np.float32)
        else:
            return np.zeros(self.buffer_samples, dtype=np.float32)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute action and return observation with waveform."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        waveform = self._extract_waveform(info)

        return (
            {"waveform": waveform},
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Reset environment and return initial observation with waveform."""
        obs, info = self.env.reset(seed=seed, options=options)

        waveform = self._extract_waveform(info)

        return {"waveform": waveform}, info


class BimodalWaveformWrapper(gym.Wrapper):
    """
    Wrapper that combines vision and raw audio waveform observations.

    This is similar to BimodalWrapper but uses raw waveform instead of
    subtitle-based sound encoding.

    Observation space:
        {
            "vision": Box(0, 255, (height, width, 3), uint8),
            "audio": Box(-1, 1, (channels, samples), float32)
        }
    """

    def __init__(
        self,
        env: gym.Env,
        x_dim: int,
        y_dim: int,
        audio_format: Literal["raw", "spectrogram"] = "raw",
        n_mels: int = 64,
        **kwargs,
    ):
        """
        Initialize BimodalWaveformWrapper.

        Args:
            env: CraftGround environment
            x_dim: Image width
            y_dim: Image height
            audio_format: "raw" for waveform, "spectrogram" for mel spectrogram
            n_mels: Number of mel bands if using spectrogram
        """
        super().__init__(env)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.audio_format = audio_format
        self.n_mels = n_mels

        # Audio parameters
        self.sample_rate = 44100
        self.channels = 2
        self.buffer_samples = 2205

        # Setup observation space
        vision_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(y_dim, x_dim, 3),
            dtype=np.uint8,
        )

        if audio_format == "raw":
            audio_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.channels, self.buffer_samples),
                dtype=np.float32,
            )
        else:  # spectrogram
            time_frames = self.buffer_samples // 512 + 1
            audio_space = gym.spaces.Box(
                low=-80.0, high=0.0, shape=(n_mels, time_frames), dtype=np.float32
            )

        self.observation_space = gym.spaces.Dict(
            {
                "vision": vision_space,
                "audio": audio_space,
            }
        )

    def _decode_audio(self, info: dict) -> np.ndarray:
        """Decode audio from observation info."""
        obs_info = info.get("obs")
        if obs_info is None or not hasattr(obs_info, "audio_waveform"):
            return np.zeros((self.channels, self.buffer_samples), dtype=np.float32)

        audio_waveform = obs_info.audio_waveform
        if audio_waveform is None or not hasattr(audio_waveform, "pcm_data"):
            return np.zeros((self.channels, self.buffer_samples), dtype=np.float32)

        pcm_data = audio_waveform.pcm_data
        if len(pcm_data) == 0:
            return np.zeros((self.channels, self.buffer_samples), dtype=np.float32)

        # Decode 16-bit PCM
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Reshape for stereo
        if self.channels == 2 and len(audio) >= 2:
            if len(audio) % 2 != 0:
                audio = audio[:-1]
            audio = audio.reshape(-1, 2).T

        if self.audio_format == "spectrogram":
            try:
                import librosa

                mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio
                mel_spec = librosa.feature.melspectrogram(
                    y=mono, sr=self.sample_rate, n_mels=self.n_mels
                )
                return librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
            except ImportError:
                pass

        return audio.astype(np.float32)

    def step(self, action: WrapperActType):
        obs, reward, terminated, truncated, info = self.env.step(action)

        rgb = info.get("pov", np.zeros((self.y_dim, self.x_dim, 3), dtype=np.uint8))
        audio = self._decode_audio(info)

        return (
            {"vision": rgb, "audio": audio},
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ):
        obs, info = self.env.reset(seed=seed, options=options)

        rgb = info.get("pov", np.zeros((self.y_dim, self.x_dim, 3), dtype=np.uint8))
        audio = self._decode_audio(info)

        return {"vision": rgb, "audio": audio}, info
