"""Self-healing primitives: the model CircuitBreaker and the SessionGuard singleton.

The breaker moved from advanced_agent (v8 request-ceiling design) to
model_registry (V17 sliding-window, three-state design); these tests cover the
current CLOSED → OPEN → HALF_OPEN → CLOSED cycle.

Run: .venv/bin/python -m pytest test_selfhealing.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python-orchestrator"))

from model_registry import CircuitBreaker


def test_no_trip_below_min_calls():
    cb = CircuitBreaker(min_calls=5, failure_rate_threshold=0.5)
    for _ in range(4):
        cb.record_failure()
    assert cb.state == "closed" and not cb.tripped


def test_trips_open_when_failure_rate_crosses_threshold():
    cb = CircuitBreaker(min_calls=4, failure_rate_threshold=0.5, open_wait_secs=60)
    cb.record_success()
    cb.record_success()
    cb.record_failure()
    assert cb.state == "closed"
    cb.record_failure()                       # 2/4 = 50% ≥ threshold
    assert cb.state == "open" and cb.tripped
    assert cb.reason.startswith("OPEN")
    assert "2/4 fails" in cb.status_line()


def test_healthy_traffic_never_trips():
    cb = CircuitBreaker(min_calls=4, failure_rate_threshold=0.5)
    for _ in range(20):
        cb.record_success()
    cb.record_failure()
    assert cb.state == "closed" and cb.reason == ""


def test_open_blocks_then_recovers_through_half_open():
    async def scenario():
        cb = CircuitBreaker(min_calls=2, failure_rate_threshold=0.5,
                            open_wait_secs=0.05, half_open_probes=2)
        await cb.record(True)
        await cb.record(True)
        assert cb.state == "open"
        assert await cb.allow() is False     # still cooling down
        await asyncio.sleep(0.06)
        assert not cb.tripped                 # eligible for probing
        assert await cb.allow() is True and cb.state == "half_open"
        assert await cb.allow() is True       # second probe
        assert await cb.allow() is False      # probe budget spent
        await cb.record(False)
        await cb.record(False)
        assert cb.state == "closed"           # all probes passed → recovered

    asyncio.run(scenario())


def test_failed_probe_reopens():
    async def scenario():
        cb = CircuitBreaker(min_calls=2, failure_rate_threshold=0.5, open_wait_secs=0.01)
        await cb.record(True)
        await cb.record(True)
        await asyncio.sleep(0.02)
        assert await cb.allow() is True and cb.state == "half_open"
        await cb.record(True)
        assert cb.state == "open"

    asyncio.run(scenario())


def test_session_guard_is_singleton():
    from advanced_agent import SessionGuard
    assert SessionGuard.get() is SessionGuard.get()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
