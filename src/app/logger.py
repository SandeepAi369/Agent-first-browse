"""Shared structured logging for Agent First IDE.

Provides a pre-configured logger factory that replaces raw print() calls
with proper leveled logging. All modules should use:

    from app.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

_FMT = logging.Formatter(
    fmt="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)

# One timestamped file per process so every run is saved for later analysis
# (the run scripts don't redirect, so console output was previously lost).
# Disabled under pytest to avoid littering log files during tests.
_FILE_HANDLER: logging.Handler | None | bool = None


def _file_handler() -> logging.Handler | None:
    global _FILE_HANDLER
    if _FILE_HANDLER is None:
        if "pytest" in sys.modules or os.getenv("AGENT_NO_FILE_LOG"):
            _FILE_HANDLER = False
        else:
            try:
                log_dir = Path(os.getenv("AGENT_LOG_DIR", "logs"))
                log_dir.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(
                    log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log",
                    encoding="utf-8")
                fh.setFormatter(_FMT)
                _FILE_HANDLER = fh
            except Exception:
                _FILE_HANDLER = False
    return _FILE_HANDLER or None


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently configured logger for the given module name.

    Streams to stdout AND (for real runs) appends to a per-run file under logs/
    so sessions can be analyzed afterward. Safe to call multiple times.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(_FMT)
        logger.addHandler(handler)
        fh = _file_handler()
        if fh is not None:
            logger.addHandler(fh)
        logger.propagate = False

    return logger
