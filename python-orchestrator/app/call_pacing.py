"""Async call pacing helpers for provider/API rate smoothing."""

from __future__ import annotations

import asyncio
import time


class AsyncGapLimiter:
    """Enforce a minimum time gap between async calls across tasks."""

    def __init__(self, min_gap_seconds: float) -> None:
        self._min_gap_seconds = max(0.0, float(min_gap_seconds))
        self._lock = asyncio.Lock()
        self._last_call_at = 0.0

    async def wait_turn(self) -> float:
        """Wait if needed so consecutive calls keep at least the configured gap.

        Returns the wait time in seconds applied for the current turn.
        """
        if self._min_gap_seconds <= 0.0:
            return 0.0

        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_at
            wait_seconds = max(0.0, self._min_gap_seconds - elapsed)
            if wait_seconds > 0.0:
                await asyncio.sleep(wait_seconds)
            self._last_call_at = time.monotonic()
            return wait_seconds
