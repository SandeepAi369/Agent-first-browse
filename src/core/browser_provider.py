"""Browser provider seam, per-session scoped locks, and restart tombstone tracking.

Adapts Cloudflare's `browser/providers/provider-adapter.ts`, `browser/scope.ts`,
`browser/session-store.ts`, `browser/browser.ts`, and `browser/session.ts` to
Python 3.11+ `asyncio`:

  - Closed `BrowserProvider` Protocol seam decoupling browser launch/liveness/
    teardown from session orchestration, locks, and restart detection.
  - Per-session `asyncio.Lock` concurrency: the short `_registry_lock` only
    guards in-memory map lookups and is NEVER held across slow browser launch,
    liveness probes, or user session I/O — so independent sessions execute truly
    in parallel while same-session operations serialize cleanly.
  - Retired tombstone tracking: once a named session has owned a browser, closing,
    sweeping, or crashing moves its record to `_retired` so a subsequent acquire
    loudly reports `restarted=True` with explicit `notices` instead of silently
    resuming in a blank tab.
  - Flaky-probe protection: if `provider.is_alive(handle)` raises a transient
    error (other than `BrowserSessionGoneError`), the live session is preserved
    rather than discarded.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.errors import BrowserSessionGoneError, ConfigValidationError
from core.runtime_context import (
    DEFAULT_RUNTIME_SCOPE,
    VALID_RUNTIME_MODES,
    BrowserRuntimeContext,
    use_runtime_context,
)

DEFAULT_BROWSER_SCOPE: str = DEFAULT_RUNTIME_SCOPE
DEFAULT_SWEEP_IDLE_S: float = 600.0  # 10 minutes


def _validate_non_empty_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{field_name} must be a non-empty string, got {value!r}")
    return value.strip()


def conversation_scope(conversation_id: str) -> str:
    """Build a canonical conversation scope (`conversation:<id>`), rejecting empty IDs."""
    clean = _validate_non_empty_id(conversation_id, "conversation_id")
    return f"conversation:{clean}"


@dataclass(frozen=True)
class BrowserLaunchConfig:
    """Plain-data configuration passed to a `BrowserProvider.create()` call."""

    headless: bool = True
    viewport: tuple[int, int] = (1280, 800)
    engine_mode: str = "v2"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.headless, bool):
            raise ConfigValidationError(f"headless must be a bool, got {self.headless!r}")
        if not isinstance(self.viewport, tuple) or len(self.viewport) != 2:
            raise ConfigValidationError(
                f"viewport must be a (width, height) tuple, got {self.viewport!r}"
            )
        w, h = self.viewport
        if isinstance(w, bool) or isinstance(h, bool) or not isinstance(w, int) or not isinstance(h, int):
            raise ConfigValidationError(f"viewport dimensions must be ints, got {self.viewport!r}")
        if w <= 0 or h <= 0:
            raise ConfigValidationError(f"viewport dimensions must be > 0, got {self.viewport!r}")
        if not isinstance(self.engine_mode, str) or self.engine_mode.strip().lower() not in VALID_RUNTIME_MODES:
            raise ConfigValidationError(
                f"engine_mode must be one of {VALID_RUNTIME_MODES}, got {self.engine_mode!r}"
            )


@dataclass
class BrowserHandle:
    """Live provider-owned browser handle for one named session."""

    session_id: str
    page: Any
    browser_context: Any = None
    driver: str = "playwright"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionRecord:
    """Durable-style lifecycle metadata & retired tombstone state for a session."""

    session_id: str
    scope: str
    created_at: float
    updated_at: float
    closed_at: float | None = None
    retired: bool = False
    restart_count: int = 0
    close_reason: str = ""


@dataclass(frozen=True)
class SessionReport:
    """Lifecycle outcome returned when a session closes or reports health notices."""

    session_id: str
    scope: str
    restarted: bool = False
    restart_count: int = 0
    notices: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "scope": self.scope,
            "restarted": self.restarted,
            "restart_count": self.restart_count,
            "notices": list(self.notices),
        }


@dataclass(frozen=True)
class ResolvedSession:
    """Result of acquiring a live browser session via `BrowserSessionManager`."""

    session_id: str
    scope: str
    handle: BrowserHandle
    runtime_context: BrowserRuntimeContext
    restarted: bool = False
    restart_count: int = 0
    notices: tuple[str, ...] = ()

    def report(self) -> SessionReport:
        return SessionReport(
            session_id=self.session_id,
            scope=self.scope,
            restarted=self.restarted,
            restart_count=self.restart_count,
            notices=self.notices,
        )


@runtime_checkable
class BrowserProvider(Protocol):
    """Internal seam between `BrowserSessionManager` and the concrete browser driver."""

    @property
    def name(self) -> str: ...

    async def create(self, session_id: str, config: BrowserLaunchConfig) -> BrowserHandle: ...

    async def is_alive(self, handle: BrowserHandle) -> bool:
        """Return True if alive, False if definitely gone; raise if probe is uncertain."""
        ...

    async def close(self, handle: BrowserHandle) -> None:
        """Close the handle idempotently (an already-gone handle counts as closed)."""
        ...

    def driver_info(self) -> dict[str, Any]: ...


class LocalPlaywrightProvider:
    """Lazy-imported local Chromium `BrowserProvider` with zero import-time Playwright cost."""

    def __init__(self) -> None:
        self._pw: Any = None
        self._browsers: dict[bool, Any] = {}
        self._launch_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "local-playwright"

    async def _ensure_browser(self, headless: bool) -> Any:
        async with self._launch_lock:
            if self._pw is None:
                from playwright.async_api import async_playwright

                self._pw = await async_playwright().start()
            browser = self._browsers.get(headless)
            if browser is None or not browser.is_connected():
                browser = await self._pw.chromium.launch(headless=headless)
                self._browsers[headless] = browser
            return browser

    async def create(self, session_id: str, config: BrowserLaunchConfig) -> BrowserHandle:
        browser = await self._ensure_browser(config.headless)
        vw, vh = config.viewport
        context = await browser.new_context(viewport={"width": vw, "height": vh})
        page = await context.new_page()
        now = time.time()
        return BrowserHandle(
            session_id=session_id,
            page=page,
            browser_context=context,
            driver="playwright",
            created_at=now,
            updated_at=now,
            metadata={"headless": config.headless, "viewport": [vw, vh]},
        )

    async def is_alive(self, handle: BrowserHandle) -> bool:
        page = handle.page
        if page is None:
            return False
        is_closed = getattr(page, "is_closed", None)
        if callable(is_closed):
            return not bool(is_closed())
        return True

    async def close(self, handle: BrowserHandle) -> None:
        ctx = handle.browser_context
        if ctx is not None:
            with suppress(Exception):
                await ctx.close()

    def driver_info(self) -> dict[str, Any]:
        has_patchright = importlib.util.find_spec("patchright") is not None
        return {
            "provider": self.name,
            "driver": "patchright" if has_patchright else "playwright",
            "patchright_installed": has_patchright,
        }

    async def aclose(self) -> None:
        """Close all shared browser instances and stop the Playwright driver."""
        async with self._launch_lock:
            for browser in list(self._browsers.values()):
                with suppress(Exception):
                    await browser.close()
            self._browsers.clear()
            if self._pw is not None:
                with suppress(Exception):
                    await self._pw.stop()
                self._pw = None


class BrowserSessionManager:
    """Manages named browser sessions with per-session locks and restart tombstones.

    Concurrency invariant:
      - `_registry_lock` is held ONLY around short in-memory dictionary reads/
        writes (obtaining a session's lock, snapshotting metadata, updating
        records) and NEVER across provider `create()`, `is_alive()`, `close()`,
        or caller work inside `use_session()`.
      - Two operations targeting different `session_id`s run fully in parallel;
        two operations targeting the same `session_id` serialize cleanly.
    """

    def __init__(
        self,
        provider: BrowserProvider,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._provider = provider
        self._clock: Callable[[], float] = clock or time.time
        self._registry_lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._handles: dict[str, BrowserHandle] = {}
        self._contexts: dict[str, BrowserRuntimeContext] = {}
        self._records: dict[str, SessionRecord] = {}
        self._retired: dict[str, SessionRecord] = {}

    @property
    def provider(self) -> BrowserProvider:
        return self._provider

    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        async with self._registry_lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    async def _probe_alive(self, handle: BrowserHandle) -> bool:
        try:
            return bool(await self._provider.is_alive(handle))
        except BrowserSessionGoneError:
            return False

    async def _retire_unlocked(self, session_id: str, reason: str) -> SessionRecord | None:
        """Close handle & runtime context for `session_id` (caller holds session lock)."""
        async with self._registry_lock:
            handle = self._handles.pop(session_id, None)
            ctx = self._contexts.pop(session_id, None)
            record = self._records.pop(session_id, None)

        if ctx is not None:
            await ctx.aclose()
        if handle is not None:
            with suppress(Exception):
                await self._provider.close(handle)

        if record is None:
            return None

        now = self._clock()
        tombstone = SessionRecord(
            session_id=record.session_id,
            scope=record.scope,
            created_at=record.created_at,
            updated_at=now,
            closed_at=now,
            retired=True,
            restart_count=record.restart_count,
            close_reason=reason,
        )
        async with self._registry_lock:
            self._retired[session_id] = tombstone
        return tombstone

    async def _create_fresh_unlocked(
        self,
        session_id: str,
        scope: str,
        config: BrowserLaunchConfig,
        prior_tombstone: SessionRecord | None,
    ) -> ResolvedSession:
        handle = await self._provider.create(session_id, config)
        now = self._clock()
        handle.updated_at = now

        restarted = prior_tombstone is not None
        restart_count = (prior_tombstone.restart_count + 1) if prior_tombstone is not None else 0
        notices: tuple[str, ...] = ()
        if prior_tombstone is not None:
            reason = prior_tombstone.close_reason or "previous session ended"
            notices = (
                f"Browser session '{session_id}' was restarted ({reason}); "
                "prior tab, cookie, and DOM state from the earlier browser is gone.",
            )

        ctx = BrowserRuntimeContext(
            session_id=session_id,
            scope=scope,
            engine_mode=config.engine_mode,
            page=handle.page,
            browser_context=handle.browser_context,
        )
        record = SessionRecord(
            session_id=session_id,
            scope=scope,
            created_at=now,
            updated_at=now,
            retired=False,
            restart_count=restart_count,
        )

        async with self._registry_lock:
            self._handles[session_id] = handle
            self._contexts[session_id] = ctx
            self._records[session_id] = record
            self._retired.pop(session_id, None)

        return ResolvedSession(
            session_id=session_id,
            scope=scope,
            handle=handle,
            runtime_context=ctx,
            restarted=restarted,
            restart_count=restart_count,
            notices=notices,
        )

    async def _acquire_under_session_lock(
        self,
        clean_id: str,
        clean_scope: str,
        cfg: BrowserLaunchConfig,
        auto_recover: bool,
    ) -> ResolvedSession:
        async with self._registry_lock:
            existing_handle = self._handles.get(clean_id)
            existing_ctx = self._contexts.get(clean_id)
            existing_record = self._records.get(clean_id)
            prior_tombstone = self._retired.get(clean_id)

        if existing_handle is not None and existing_ctx is not None and existing_record is not None:
            if await self._probe_alive(existing_handle):
                now = self._clock()
                existing_handle.updated_at = now
                existing_record.updated_at = now
                return ResolvedSession(
                    session_id=clean_id,
                    scope=existing_record.scope,
                    handle=existing_handle,
                    runtime_context=existing_ctx,
                    restarted=False,
                    restart_count=existing_record.restart_count,
                    notices=(),
                )
            # Definitely gone -> record tombstone before deciding whether to recreate
            prior_tombstone = await self._retire_unlocked(
                clean_id,
                reason="browser closed or crashed upstream",
            )
            if not auto_recover:
                raise BrowserSessionGoneError(clean_id)

        return await self._create_fresh_unlocked(clean_id, clean_scope, cfg, prior_tombstone)

    async def acquire(
        self,
        session_id: str,
        *,
        config: BrowserLaunchConfig | None = None,
        scope: str = DEFAULT_BROWSER_SCOPE,
        auto_recover: bool = True,
    ) -> ResolvedSession:
        """Acquire or recover a named browser session, reporting restarts loudly."""
        clean_id = _validate_non_empty_id(session_id, "session_id")
        clean_scope = _validate_non_empty_id(scope, "scope")
        cfg = config if config is not None else BrowserLaunchConfig()

        session_lock = await self._get_session_lock(clean_id)
        async with session_lock:
            return await self._acquire_under_session_lock(
                clean_id, clean_scope, cfg, auto_recover
            )

    @asynccontextmanager
    async def use_session(
        self,
        session_id: str,
        *,
        config: BrowserLaunchConfig | None = None,
        scope: str = DEFAULT_BROWSER_SCOPE,
        auto_recover: bool = True,
    ) -> AsyncIterator[ResolvedSession]:
        """Hold `session_id`'s per-session lock and bind its `BrowserRuntimeContext`."""
        clean_id = _validate_non_empty_id(session_id, "session_id")
        clean_scope = _validate_non_empty_id(scope, "scope")
        cfg = config if config is not None else BrowserLaunchConfig()

        session_lock = await self._get_session_lock(clean_id)
        async with session_lock:
            resolved = await self._acquire_under_session_lock(
                clean_id, clean_scope, cfg, auto_recover
            )
            with use_runtime_context(resolved.runtime_context):
                yield resolved

    async def close_session(
        self,
        session_id: str,
        *,
        reason: str = "closed by caller",
    ) -> SessionReport:
        """Close `session_id`, persist a retired tombstone, and return its `SessionReport`."""
        clean_id = _validate_non_empty_id(session_id, "session_id")
        session_lock = await self._get_session_lock(clean_id)
        async with session_lock:
            tombstone = await self._retire_unlocked(clean_id, reason=reason)
            if tombstone is None:
                async with self._registry_lock:
                    existing = self._retired.get(clean_id)
                return SessionReport(
                    session_id=clean_id,
                    scope=existing.scope if existing is not None else DEFAULT_BROWSER_SCOPE,
                    restarted=False,
                    restart_count=existing.restart_count if existing is not None else 0,
                    notices=(),
                )
            return SessionReport(
                session_id=clean_id,
                scope=tombstone.scope,
                restarted=False,
                restart_count=tombstone.restart_count,
                notices=(f"Browser session '{clean_id}' closed ({reason}).",),
            )

    async def sweep_idle(
        self,
        max_idle_s: float = DEFAULT_SWEEP_IDLE_S,
        *,
        now: float | None = None,
    ) -> list[str]:
        """Close and tombstone sessions idle for at least `max_idle_s` seconds."""
        if isinstance(max_idle_s, bool) or not isinstance(max_idle_s, (int, float)):
            raise ConfigValidationError(f"max_idle_s must be a finite number > 0, got {max_idle_s!r}")
        if not math.isfinite(float(max_idle_s)) or float(max_idle_s) <= 0:
            raise ConfigValidationError(f"max_idle_s must be > 0, got {max_idle_s!r}")

        cutoff = (self._clock() if now is None else float(now)) - float(max_idle_s)
        async with self._registry_lock:
            candidates = [
                sid for sid, rec in self._records.items() if rec.updated_at <= cutoff
            ]

        swept: list[str] = []
        for sid in candidates:
            session_lock = await self._get_session_lock(sid)
            async with session_lock:
                async with self._registry_lock:
                    rec = self._records.get(sid)
                if rec is not None and rec.updated_at <= cutoff:
                    await self._retire_unlocked(sid, reason="swept after idle timeout")
                    swept.append(sid)
        return swept

    async def close_all(self) -> list[SessionReport]:
        """Close all currently active sessions cleanly."""
        async with self._registry_lock:
            active_ids = list(self._records.keys())
        reports: list[SessionReport] = []
        for sid in active_ids:
            reports.append(await self.close_session(sid, reason="manager shutdown"))
        return reports

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return a secret-free diagnostic list of currently active sessions."""
        items: list[dict[str, Any]] = []
        for sid, rec in sorted(self._records.items()):
            handle = self._handles.get(sid)
            ctx = self._contexts.get(sid)
            items.append(
                {
                    "session_id": sid,
                    "scope": rec.scope,
                    "engine_mode": ctx.engine_mode if ctx is not None else "unknown",
                    "driver": handle.driver if handle is not None else "unknown",
                    "created_at": rec.created_at,
                    "updated_at": rec.updated_at,
                    "restart_count": rec.restart_count,
                }
            )
        return items

    def get_tombstone(self, session_id: str) -> SessionRecord | None:
        """Inspect retired tombstone metadata for a closed/crashed session, if any."""
        return self._retired.get(session_id)


__all__ = [
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
]
