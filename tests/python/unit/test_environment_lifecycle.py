"""Tests for environment lifecycle management."""

import pytest

from craftground import make
from craftground.exceptions import InvalidImageSizeError
from craftground.initial_environment_config import InitialEnvironmentConfig


class TestEnvironmentLifecycle:
    """Test environment creation and lifecycle."""

    def test_environment_creation(self):
        """Test that environment can be created."""
        config = InitialEnvironmentConfig(image_width=64, image_height=64)
        env = make(initial_env_config=config, verbose=False)
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_invalid_image_size(self):
        """Test that invalid image sizes raise appropriate errors."""
        # Too small
        with pytest.raises(InvalidImageSizeError):
            InitialEnvironmentConfig(image_width=0, image_height=64)

        # Too large
        with pytest.raises(InvalidImageSizeError):
            InitialEnvironmentConfig(image_width=10000, image_height=64)

        # Negative
        with pytest.raises(InvalidImageSizeError):
            InitialEnvironmentConfig(image_width=-1, image_height=64)

    def test_config_validation(self):
        """Test that configuration validation works."""
        # Valid config
        config = InitialEnvironmentConfig(image_width=640, image_height=360)
        assert config.imageSizeX == 640
        assert config.imageSizeY == 360

        # Invalid config should raise error
        with pytest.raises(InvalidImageSizeError):
            InitialEnvironmentConfig(image_width=0, image_height=360)
