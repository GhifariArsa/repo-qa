"""String helpers forming a deliberate three-hop call chain."""


def strip_ws(value):
    return value.strip()


def sanitize(value):
    return strip_ws(value)


def normalize(value):
    return sanitize(value).lower()


def unused_helper():
    """Never called by anything -- fan-in of zero on purpose."""
    return 42
