"""Unit tests for SessionGuard's Ctrl+C / SIGTERM shutdown path (no browser).

A signal must CANCEL the bound task so its `finally` awaits shutdown(), which
closes the context AND stops the Playwright driver — instead of the old
sys.exit() from inside the handler, which skipped both.

Run: .venv/bin/python -m pytest tests/test_session_guard.py -v
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from advanced_agent import SessionGuard


class _Fake:
    def __init__(self, log: list, name: str):
        self._log, self._name = log, name

    async def close(self):
        self._log.append(f"{self._name}.close")

    async def stop(self):
        self._log.append(f"{self._name}.stop")


@pytest.fixture
def guard():
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    SessionGuard._instance = None
    g = SessionGuard.get()
    yield g
    g.detach()  # keep its atexit hook quiet after pytest closes the streams
    SessionGuard._instance = None
    for s, h in saved.items():
        signal.signal(s, h)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_cancels_task_and_shuts_down_driver(guard, sig):
    log: list[str] = []

    async def runner():
        guard.bind_task()
        guard.attach(_Fake(log, "context"), playwright=_Fake(log, "playwright"))
        try:
            os.kill(os.getpid(), sig)
            await asyncio.sleep(5)  # cancelled long before this elapses
            log.append("not-cancelled")
        finally:
            await guard.shutdown()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner())
    assert log == ["context.close", "playwright.stop"]


def test_shutdown_is_idempotent(guard):
    log: list[str] = []

    async def runner():
        guard.attach(_Fake(log, "context"), playwright=_Fake(log, "playwright"))
        await guard.shutdown()
        await guard.shutdown()

    asyncio.run(runner())
    assert log == ["context.close", "playwright.stop"]


def test_signal_without_bound_task_raises_keyboard_interrupt(guard):
    guard.attach(_Fake([], "context"))
    with pytest.raises(KeyboardInterrupt):
        guard._signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
