"""Shared structured logging for Agent First IDE.

Provides a pre-configured logger factory that replaces raw print() calls
with proper leveled logging. All modules should use:

    from app.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently configured logger for the given module name.

    Uses a shared StreamHandler to stdout with a compact format including
    timestamp, level, and module name. Safe to call multiple times for the
    same name — returns the same logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
