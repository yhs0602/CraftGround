"""Custom exception classes for CraftGround."""


class CraftGroundError(Exception):
    """Base exception for all CraftGround errors."""

    pass


class ProcessTerminationError(CraftGroundError):
    """Raised when process termination fails."""

    pass


class ConnectionTimeoutError(CraftGroundError):
    """Raised when connection to server times out."""

    pass


class ConfigurationError(CraftGroundError):
    """Raised when configuration is invalid."""

    pass


class PortInUseError(CraftGroundError):
    """Raised when the requested port is already in use."""

    pass


class InvalidPortError(CraftGroundError):
    """Raised when port number is invalid."""

    pass


class InvalidImageSizeError(ConfigurationError):
    """Raised when image dimensions are invalid."""

    pass
