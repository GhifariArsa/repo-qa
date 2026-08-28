"""Logging setup for the CLI.

Named `logconfig` rather than `logging` so there is no chance of shadowing the
standard library module for anything importing from this package.

Levels are used deliberately:
  INFO   phase boundaries, counts, timings — what you want on a normal run
  DEBUG  per-file and per-edge detail — `-v`, useful when a repo extracts wrong
  WARN   things silently dropped from the graph (unparseable files, skipped
         edges). These used to be invisible, which is the worst way for a
         benchmark's ground truth to be incomplete.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "cgx"


class _ElapsedFormatter(logging.Formatter):
    """`[  1.4s] message` — relative time matters more than wall clock here."""

    def format(self, record: logging.LogRecord) -> str:
        elapsed = record.relativeCreated / 1000.0
        prefix = f"[{elapsed:6.1f}s]"
        if record.levelno >= logging.WARNING:
            prefix += f" {record.levelname}"
        return f"{prefix} {record.getMessage()}"


def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO))
    handler = logging.StreamHandler(sys.stderr)   # stderr, so stdout stays pipeable
    handler.setFormatter(_ElapsedFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """`get_logger(__name__)` -> a child of the `cgx` logger."""
    suffix = name.split("cgx.", 1)[-1] if "cgx." in name else name
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}")
