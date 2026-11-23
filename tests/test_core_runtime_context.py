"""Real-world unit & concurrency tests for disposable.py and runtime_context.py."""

from __future__ import annotations

import asyncio

import pytest

import core
from core import (
    AsyncDisposable,
    AsyncDisposableStore,
    BrowserRuntimeContext,
    BrowserSessionError,
    ConfigValidationError,
    Disposable,
    DisposableStore,
    Emitter,
    get_current_context,
    require_current_context,
    resolve_context_value,
    run_in_context,
    scoped_runtime_context,
    to_async_disposable,
    to_disposable,
    use_runtime_context,
)


# ── 1. Public Barrel Integrity ─────────────────────────────────────────────────


def test_core_barrel_exports_chunk4_symbols():
    for name in (
        "Disposable",
        "AsyncDisposable",
        "DisposableStore",
        "AsyncDisposableStore",
        "Emitter",
        "to_disposable",
        "to_async_disposable",
        "BrowserRuntimeContext",
        "get_current_context",
        "require_current_context",
        "use_runtime_context",
        "scoped_runtime_context",
        "run_in_context",
        "resolve_context_value",
    ):
        assert name in core.__all__
        assert hasattr(core, name)


# ── 2. Synchronous & Asynchronous Disposables ──────────────────────────────────


def test_to_disposable_is_idempotent_and_satisfies_protocol():
    calls = 0

    def cleanup() -> None:
        nonlocal calls
        calls += 1

    d = to_disposable(cleanup)
    assert isinstance(d, Disposable)
    d.dispose()
    d.dispose()
    assert calls == 1


def test_disposable_store_lifo_order_error_isolation_and_post_dispose_add():
    order: list[str] = []
    errors: list[str] = []

    def boom() -> None:
        order.append("boom")
        raise RuntimeError("cleanup failure")

    with DisposableStore(on_error=lambda exc, _item: errors.append(str(exc))) as store:
        store.add(lambda: order.append("first"))
        store.add(boom)
        store.add(to_disposable(lambda: order.append("third")))
        assert len(store) == 3

    assert store.is_disposed is True
    assert order == ["third", "boom", "first"]
    assert errors == ["cleanup failure"]

    # Adding after dispose cleans up immediately rather than leaking
    late_cleaned = False

    def late_fn() -> None:
        nonlocal late_cleaned
        late_cleaned = True

    store.add(late_fn)
    assert late_cleaned is True
    # Re-disposing is a safe no-op
    store.dispose()
    assert order == ["third", "boom", "first"]


async def test_async_disposable_store_lifo_mixed_sync_async_and_isolation():
    order: list[str] = []
    errors: list[str] = []

    async def async_step() -> None:
        await asyncio.sleep(0)
        order.append("async_third")

    async def async_boom() -> None:
        await asyncio.sleep(0)
        order.append("async_boom")
        raise RuntimeError("async teardown failed")

    class CustomAsyncResource:
        async def adispose(self) -> None:
            order.append("custom_fourth")

    res = CustomAsyncResource()
    assert isinstance(res, AsyncDisposable)

    # Idempotent wrapper check
    single_calls = 0

    async def once() -> None:
        nonlocal single_calls
        single_calls += 1

    wrapped = to_async_disposable(once)
    await wrapped.adispose()
    await wrapped.adispose()
    assert single_calls == 1

    async with AsyncDisposableStore(on_error=lambda exc, _i: errors.append(str(exc))) as store:
        store.add(to_disposable(lambda: order.append("sync_first")))
        store.add(async_boom)
        store.add(async_step)
        store.add(res)
        assert len(store) == 4

    assert store.is_disposed is True
    assert order == ["custom_fourth", "async_third", "async_boom", "sync_first"]
    assert errors == ["async teardown failed"]

    # Registering on an already-disposed AsyncDisposableStore fails loudly (no silent leak)
    with pytest.raises(RuntimeError, match="already-disposed"):
        store.add(lambda: None)


# ── 3. Listener-Isolated Typed Emitter ─────────────────────────────────────────


def test_emitter_isolation_unsubscribe_mid_fire_and_disposal():
    received: list[str] = []
    errors: list[str] = []
    emitter: Emitter[str] = Emitter(on_error=lambda exc, _l: errors.append(str(exc)))

    handle_self: list[Disposable] = []

    def self_unsubscribing(ev: str) -> None:
        received.append(f"one:{ev}")
        handle_self[0].dispose()

    def broken(ev: str) -> None:
        raise ValueError(f"bad listener on {ev}")

    def healthy(ev: str) -> None:
        received.append(f"three:{ev}")

    handle_self.append(emitter.subscribe(self_unsubscribing))
    emitter.subscribe(broken)
    h3 = emitter.subscribe(healthy)
    assert emitter.listener_count == 3

    emitter.fire("alpha")
    assert received == ["one:alpha", "three:alpha"]
    assert errors == ["bad listener on alpha"]
    assert emitter.listener_count == 2

    # Second fire: self_unsubscribing is gone, healthy still fires
    emitter.fire("beta")
    assert received == ["one:alpha", "three:alpha", "three:beta"]

    # Unsubscribing via DisposableStore works cleanly
    with DisposableStore() as ds:
        ds.add(h3)
    assert emitter.listener_count == 1

    emitter.dispose()
    assert emitter.is_disposed is True
    assert emitter.listener_count == 0
    # Firing or subscribing after dispose is a safe no-op
    emitter.subscribe(lambda ev: received.append(f"late:{ev}"))
    emitter.fire("gamma")
    assert received == ["one:alpha", "three:alpha", "three:beta"]


# ── 4. BrowserRuntimeContext Validation, Nesting & Lifecycle ───────────────────


def test_runtime_context_validation_mutators_and_snapshot():
    ctx = BrowserRuntimeContext(session_id=" eval-v2 ", scope=" tab:1 ", engine_mode=" V2 ")
    assert ctx.session_id == "eval-v2"
    assert ctx.scope == "tab:1"
    assert ctx.engine_mode == "v2"
    assert ctx.is_closed is False

    fake_page = object()
    ctx.set_page(fake_page)
    ctx.set_selector_map({"e1": {"x": 10, "y": 20}})
    assert ctx.set_engine_mode("LEGACY") == "legacy"

    snap = ctx.snapshot_info()
    assert snap == {
        "session_id": "eval-v2",
        "scope": "tab:1",
        "engine_mode": "legacy",
        "has_page": True,
        "has_browser_context": False,
        "selector_count": 1,
        "has_compressor": False,
        "has_metrics": False,
        "allowed_domains_count": 0,
        "closed": False,
    }

    with pytest.raises(ConfigValidationError):
        BrowserRuntimeContext(session_id="   ")
    with pytest.raises(ConfigValidationError):
        BrowserRuntimeContext(scope="")
    with pytest.raises(ConfigValidationError):
        BrowserRuntimeContext(engine_mode="v3")
    with pytest.raises(ConfigValidationError):
        ctx.set_engine_mode("invalid")


def test_use_runtime_context_nesting_exception_restore_and_fallback():
    assert get_current_context() is None
    with pytest.raises(BrowserSessionError):
        require_current_context()
    assert resolve_context_value("engine_mode", "legacy") == "legacy"

    outer = BrowserRuntimeContext(session_id="outer", engine_mode="legacy", page=None)
    inner = BrowserRuntimeContext(session_id="inner", engine_mode="v2", page="inner-page")

    with use_runtime_context(outer):
        assert require_current_context() is outer
        assert resolve_context_value("engine_mode", "fallback") == "legacy"
        # None field on active context gracefully falls back to legacy global default
        assert resolve_context_value("page", "legacy-global-page") == "legacy-global-page"

        with pytest.raises(RuntimeError, match="boom"):
            with use_runtime_context(inner):
                assert get_current_context() is inner
                assert resolve_context_value("page", "legacy-global-page") == "inner-page"
                raise RuntimeError("boom")

        # Outer context restored cleanly; inner stays open for subsequent steps
        assert get_current_context() is outer
        assert inner.is_closed is False

    assert get_current_context() is None


async def test_scoped_runtime_context_and_run_in_context():
    closed_steps: list[str] = []
    ctx = BrowserRuntimeContext(session_id="run-1", engine_mode="v2", page="p1")
    ctx.register_cleanup(lambda: closed_steps.append("sync-stop"))

    async def close_browser() -> None:
        await asyncio.sleep(0)
        closed_steps.append("async-browser-close")

    ctx.register_cleanup(close_browser)

    with pytest.raises(ValueError, match="task error"):
        async with scoped_runtime_context(ctx) as active:
            assert get_current_context() is active
            assert active.session_id == "run-1"
            raise ValueError("task error")

    # Both LIFO cleanup and ContextVar reset run even on exception
    assert get_current_context() is None
    assert ctx.is_closed is True
    assert closed_steps == ["async-browser-close", "sync-stop"]

    # run_in_context supports both sync and async callables
    ctx2 = BrowserRuntimeContext(session_id="run-2", engine_mode="v2", page="p2")
    assert await run_in_context(ctx2, lambda: require_current_context().session_id) == "run-2"

    async def async_reader(prefix: str) -> str:
        await asyncio.sleep(0)
        return f"{prefix}:{require_current_context().page}"

    assert await run_in_context(ctx2, async_reader, "seen") == "seen:p2"
    assert get_current_context() is None


# ── 5. True Concurrent Multi-Session Asyncio Isolation ─────────────────────────


async def test_concurrent_asyncio_tasks_have_zero_state_bleed():
    """Prove paired legacy + v2 sessions interleaved via asyncio.gather stay isolated."""
    sessions = [
        BrowserRuntimeContext(
            session_id=f"sess-{idx}",
            scope=f"scope-{idx}",
            engine_mode="legacy" if idx % 2 == 0 else "v2",
            page=f"page-{idx}",
            selector_map={f"e{idx}": {"x": idx * 10, "y": idx * 20}},
            compressor=f"compressor-{idx}",
            metrics=f"metrics-{idx}",
        )
        for idx in range(4)
    ]

    async def worker(ctx: BrowserRuntimeContext, idx: int) -> list[tuple[str, str, str]]:
        observed: list[tuple[str, str, str]] = []
        with use_runtime_context(ctx):
            for step in range(15):
                # Force repeated coroutine switches across all 4 tasks
                await asyncio.sleep(0)
                active = require_current_context()
                assert active is ctx
                assert active.session_id == f"sess-{idx}"
                assert active.page == f"page-{idx}"
                assert active.engine_mode == ("legacy" if idx % 2 == 0 else "v2")
                assert resolve_context_value("compressor", None) == f"compressor-{idx}"
                assert resolve_context_value("metrics", None) == f"metrics-{idx}"
                assert list(active.selector_map.keys()) == [f"e{idx}"]
                observed.append((active.session_id, active.engine_mode, f"step-{step}"))
        return observed

    results = await asyncio.gather(*(worker(s, i) for i, s in enumerate(sessions)))
    assert len(results) == 4
    for i, trace in enumerate(results):
        assert len(trace) == 15
        assert all(row[0] == f"sess-{i}" for row in trace)
    assert get_current_context() is None
