"""Configuration constants and accessors for minirepo."""

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
LOG_LEVEL = "INFO"


def get_timeout():
    """Return the configured default timeout."""
    return DEFAULT_TIMEOUT


def get_retries():
    """Return the configured retry budget."""
    return MAX_RETRIES
