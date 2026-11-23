"""v2_engine.graph_policy — LangGraph RetryPolicy & Optional Stealth Driver Probe.

Provides:
  1. `v2_retry_policy()`: Native LangGraph `RetryPolicy` targeting only transient
     network/timeout/browser hiccups (exponential backoff + jitter), never
     retrying logic/validation bugs or user interrupts.
  2. `detect_stealth_driver()`: Probes whether optional 2026 driver-level stealth
     packages (`patchright` / `camoufox`) are installed so V2 logs exact stealth
     capabilities without forcing heavy optional dependencies.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from langgraph.types import RetryPolicy

_NON_TRANSIENT = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    AssertionError,
    KeyboardInterrupt,
    SystemExit,
)

_TRANSIENT_MSG_HINTS = (
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection reset",
    "rate limit",
    "429",
    "502",
    "503",
    "504",
    "execution context was destroyed",
)


def is_transient_graph_error(exc: Exception) -> bool:
    """Predicate for LangGraph RetryPolicy — retries only transient I/O errors."""
    if isinstance(exc, _NON_TRANSIENT):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    msg = str(exc).lower()
    return any(hint in msg for hint in _TRANSIENT_MSG_HINTS)


def v2_retry_policy(max_attempts: int = 3) -> RetryPolicy:
    """Build a bounded LangGraph RetryPolicy for transient browser/LLM nodes."""
    return RetryPolicy(
        initial_interval=0.4,
        backoff_factor=2.0,
        max_interval=4.0,
        max_attempts=max(1, max_attempts),
        jitter=True,
        retry_on=is_transient_graph_error,
    )


def detect_stealth_driver() -> dict[str, Any]:
    """Detect available Playwright-compatible stealth drivers without importing them."""
    has_patchright = importlib.util.find_spec("patchright") is not None
    has_camoufox = importlib.util.find_spec("camoufox") is not None
    return {
        "driver": "patchright" if has_patchright else "playwright",
        "patchright_installed": has_patchright,
        "camoufox_installed": has_camoufox,
        "cdp_runtime_leak_patched": has_patchright,
    }
