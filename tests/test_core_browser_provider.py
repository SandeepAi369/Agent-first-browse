"""Unit, concurrency, and live-Playwright lifecycle tests for src/core/browser_provider.py."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

import core
from core import (
    DEFAULT_BROWSER_SCOPE,
    BrowserHandle,
    BrowserLaunchConfig,
    BrowserProvider,
    BrowserSessionGoneError,
    BrowserSessionManager,
    ConfigValidationError,
    LocalPlaywrightProvider,
    conversation_scope,
    get_current_context,
    require_current_context,
)


class _FakeProvider:
    """In-memory deterministic BrowserProvider for unit & concurrency testing."""

    def __init__(self) -> None:
        self.created: list[BrowserHandle] = []
        self.closed: list[str] = []
        self.alive_map: dict[int, bool | Exception] = {}
        self._seq = 0

    @property
    def name(self) -> str:
        return "fake-provider"

    async def create(self, session_id: str, config: BrowserLaunchConfig) -> BrowserHandle:
        self._seq += 1
        page_token = f"page-{self._seq}"
        handle = BrowserHandle(
            session_id=session_id,
            page=page_token,
            browser_context=f"ctx-{self._seq}",
            driver="fake",
            metadata={"seq": self._seq, "headless": config.headless},
        )
        self.alive_map[id(handle)] = True
        self.created.append(handle)
        return handle

    async def is_alive(self, handle: BrowserHandle) -> bool:
        state = self.alive_map.get(id(handle), False)
        if isinstance(state, Exception):
            raise state
        return bool(state)

    async def close(self, handle: BrowserHandle) -> None:
        self.alive_map[id(handle)] = False
        self.closed.append(f"{handle.session_id}:{handle.page}")

    def driver_info(self) -> dict[str, Any]:
        return {"provider": self.name, "driver": "fake"}


# ── 1. Barrel, Scope & Launch Config Validation ────────────────────────────────


def test_core_barrel_exports_chunk5_symbols_and_protocols():
    for name in (
        "DEFAULT_BROWSER_SCOPE",
        "DEFAULT_SWEEP_IDLE_S",
        "BrowserHandle",
        "BrowserLaunchConfig",
        "BrowserProvider",
        "BrowserSessionManager",
        "LocalPlaywrightProvider",
        "ResolvedSession",
        "SessionRecord",
        "SessionReport",
        "conversation_scope",
    ):
        assert name in core.__all__
        assert hasattr(core, name)

    assert isinstance(_FakeProvider(), BrowserProvider)
    assert isinstance(LocalPlaywrightProvider(), BrowserProvider)
    assert conversation_scope(" 42 ") == "conversation:42"

    with pytest.raises(ConfigValidationError):
        conversation_scope("")
    with pytest.raises(ConfigValidationError):
        conversation_scope(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"headless": "yes"},
        {"viewport": (1280,)},
        {"viewport": (0, 800)},
        {"viewport": (1280, -10)},
        {"viewport": (True, 800)},
        {"engine_mode": "v3"},
    ],
)
def test_browser_launch_config_validates_eagerly(kwargs):
    with pytest.raises(ConfigValidationError):
        BrowserLaunchConfig(**kwargs)


# ── 2. Lifecycle Reuse, Tombstones, Crash Recovery & Flaky Probe Safety ────────


async def test_session_reuse_close_tombstone_and_restart_reporting():
    provider = _FakeProvider()
    manager = BrowserSessionManager(provider)

    # First acquire -> fresh, not restarted
    first = await manager.acquire("s1", scope=conversation_scope("c1"))
    assert first.restarted is False
    assert first.restart_count == 0
    assert first.notices == ()
    assert len(provider.created) == 1

    # Second acquire while alive -> reuses exact handle & runtime context
    second = await manager.acquire("s1")
    assert second.handle is first.handle
    assert second.runtime_context is first.runtime_context
    assert second.restarted is False
    assert len(provider.created) == 1

    # Register cleanup on runtime_context to prove close_session disposes it
    cleaned = False
    first.runtime_context.register_cleanup(lambda: _mark())

    def _mark() -> None:
        nonlocal cleaned
        cleaned = True

    report = await manager.close_session("s1", reason="user reset")
    assert cleaned is True
    assert first.runtime_context.is_closed is True
    assert report.session_id == "s1"
    assert report.scope == "conversation:c1"
    assert any("user reset" in n for n in report.notices)
    assert report.to_dict()["session_id"] == "s1"

    tombstone = manager.get_tombstone("s1")
    assert tombstone is not None and tombstone.retired is True

    # Idempotent second close
    repeat_report = await manager.close_session("s1")
    assert repeat_report.notices == ()

    # Re-acquiring after close loudly reports restarted=True + notice
    reopened = await manager.acquire("s1", scope=conversation_scope("c1"))
    assert reopened.handle is not first.handle
    assert reopened.restarted is True
    assert reopened.restart_count == 1
    assert any("restarted" in n and "user reset" in n for n in reopened.notices)
    assert manager.get_tombstone("s1") is None


async def test_crash_recovery_and_auto_recover_false_behavior():
    provider = _FakeProvider()
    manager = BrowserSessionManager(provider)

    s1 = await manager.acquire("crash-sess")
    # Simulate upstream crash via is_alive returning False
    provider.alive_map[id(s1.handle)] = False

    recovered = await manager.acquire("crash-sess", auto_recover=True)
    assert recovered.restarted is True
    assert recovered.restart_count == 1
    assert recovered.handle is not s1.handle
    assert s1.runtime_context.is_closed is True
    assert any("crashed" in n for n in recovered.notices)

    # Simulate upstream crash via BrowserSessionGoneError with auto_recover=False
    provider.alive_map[id(recovered.handle)] = BrowserSessionGoneError("crash-sess")
    with pytest.raises(BrowserSessionGoneError):
        await manager.acquire("crash-sess", auto_recover=False)

    tombstone = manager.get_tombstone("crash-sess")
    assert tombstone is not None and tombstone.retired is True


async def test_flaky_liveness_probe_preserves_live_session():
    provider = _FakeProvider()
    manager = BrowserSessionManager(provider)

    live = await manager.acquire("healthy")
    # Uncertain transient probe error must NOT discard or tombstone the browser
    provider.alive_map[id(live.handle)] = TimeoutError("transient cdp lag")

    with pytest.raises(TimeoutError, match="transient cdp lag"):
        await manager.acquire("healthy")

    assert manager.get_tombstone("healthy") is None
    assert live.runtime_context.is_closed is False

    # Probe recovers -> exact same handle reused, restarted=False
    provider.alive_map[id(live.handle)] = True
    reused = await manager.acquire("healthy")
    assert reused.handle is live.handle
    assert reused.restarted is False


# ── 3. True Per-Session Parallelism & Same-Session FIFO Serialization ──────────


async def test_independent_sessions_run_concurrently_without_blocking_each_other():
    """Two different sessions must enter use_session() simultaneously (proves no global lock)."""
    provider = _FakeProvider()
    manager = BrowserSessionManager(provider)
    a_entered = asyncio.Event()
    b_entered = asyncio.Event()

    async def run_a() -> tuple[str, str]:
        cfg = BrowserLaunchConfig(engine_mode="legacy")
        async with manager.use_session("sess-a", config=cfg) as resolved:
            assert require_current_context() is resolved.runtime_context
            a_entered.set()
            # Would deadlock if a coarse manager lock were held across session I/O
            await asyncio.wait_for(b_entered.wait(), timeout=1.0)
            return resolved.session_id, require_current_context().engine_mode

    async def run_b() -> tuple[str, str]:
        cfg = BrowserLaunchConfig(engine_mode="v2")
        async with manager.use_session("sess-b", config=cfg) as resolved:
            assert require_current_context() is resolved.runtime_context
            b_entered.set()
            await asyncio.wait_for(a_entered.wait(), timeout=1.0)
            return resolved.session_id, require_current_context().engine_mode

    res_a, res_b = await asyncio.gather(run_a(), run_b())
    assert res_a == ("sess-a", "legacy")
    assert res_b == ("sess-b", "v2")
    assert get_current_context() is None


async def test_same_session_operations_serialize_without_interleaving():
    provider = _FakeProvider()
    manager = BrowserSessionManager(provider)
    trace: list[str] = []

    async def task(tag: str) -> None:
        async with manager.use_session("shared"):
            trace.append(f"enter-{tag}")
            await asyncio.sleep(0.02)
            trace.append(f"exit-{tag}")

    await asyncio.gather(task("1"), task("2"))
    # Strict non-interleaved execution (enter/exit pairs stay contiguous)
    assert trace in (
        ["enter-1", "exit-1", "enter-2", "exit-2"],
        ["enter-2", "exit-2", "enter-1", "exit-1"],
    )


# ── 4. Idle Sweep, Listing & Close All ─────────────────────────────────────────


async def test_sweep_idle_tombstones_stale_sessions_and_close_all_cleans_rest():
    now = [100.0]
    provider = _FakeProvider()
    manager = BrowserSessionManager(provider, clock=lambda: now[0])

    await manager.acquire("old-sess")
    now[0] = 550.0
    await manager.acquire("fresh-sess")

    swept = await manager.sweep_idle(max_idle_s=300.0, now=600.0)
    assert swept == ["old-sess"]

    active = manager.list_sessions()
    assert [s["session_id"] for s in active] == ["fresh-sess"]
    assert active[0]["scope"] == DEFAULT_BROWSER_SCOPE

    # Resuming the swept session loudly signals restart
    resumed = await manager.acquire("old-sess")
    assert resumed.restarted is True
    assert any("idle timeout" in n for n in resumed.notices)

    reports = await manager.close_all()
    assert {r.session_id for r in reports} == {"fresh-sess", "old-sess"}
    assert manager.list_sessions() == []

    with pytest.raises(ConfigValidationError):
        await manager.sweep_idle(max_idle_s=0)
    with pytest.raises(ConfigValidationError):
        await manager.sweep_idle(max_idle_s=math.nan)
    with pytest.raises(ConfigValidationError):
        await manager.sweep_idle(max_idle_s=True)  # type: ignore[arg-type]


# ── 5. Real Headless Chromium Smoke & Crash-Recovery Test ──────────────────────


async def test_local_playwright_provider_real_session_and_tab_crash_recovery():
    """Exercise LocalPlaywrightProvider + BrowserSessionManager against live Chromium."""
    provider = LocalPlaywrightProvider()
    manager = BrowserSessionManager(provider)
    try:
        info = provider.driver_info()
        assert info["provider"] == "local-playwright"

        async with manager.use_session(
            "live-pw",
            config=BrowserLaunchConfig(headless=True, viewport=(800, 600), engine_mode="v2"),
        ) as first:
            assert first.restarted is False
            await first.handle.page.set_content("<h1>Hello Chunk 5</h1>")
            assert await first.handle.page.text_content("h1") == "Hello Chunk 5"
            # Simulate unexpected tab close / crash
            await first.handle.page.close()

        # Next use detects closed page, tombstones it, and restarts cleanly with notice
        async with manager.use_session("live-pw") as recovered:
            assert recovered.restarted is True
            assert recovered.restart_count == 1
            assert len(recovered.notices) == 1
            assert recovered.handle.page is not first.handle.page
            await recovered.handle.page.set_content("<p>Recovered</p>")
            assert await recovered.handle.page.text_content("p") == "Recovered"
    finally:
        await manager.close_all()
        await provider.aclose()
