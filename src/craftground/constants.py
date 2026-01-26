"""Constants used throughout CraftGround."""

# Default environment values
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 360
DEFAULT_PORT = 8000
DEFAULT_FOV = 70

# Connection settings
MAX_CONNECTION_RETRIES = 1024
CONNECTION_RETRY_INTERVAL = 1.0
CONNECTION_TIMEOUT = 30.0

# Process management
PROCESS_TERMINATION_TIMEOUT = 10

# Port validation
MIN_PORT = 1
MAX_PORT = 65535

# Image validation
MIN_IMAGE_SIZE = 1
MAX_IMAGE_SIZE = 8192  # Reasonable upper limit
