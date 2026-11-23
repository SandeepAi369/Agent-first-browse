"""Explicit `ContextVar`-backed runtime context for concurrency-safe session isolation.

Adapts Cloudflare's ambient host-context pattern (`lifecycle/current-agent.ts`)
to Python 3.11+ `contextvars.ContextVar` so concurrent `asyncio.Task` sessions
each carry their own isolated `page`, `selector_map`, `engine_mode`,
`compressor`, `metrics`, LLM chains, and `AsyncDisposableStore` without mutating
process-wide module globals (`_PAGE`, `_SELECTOR_MAP`, `AGENT_ENGINE_MODE`).

Key guarantees:
  - True per-task isolation across concurrent `asyncio.gather()` sessions.
  - Safe nesting (`use_runtime_context` restores the previous `ContextVar`
    token in `finally` on normal exit or exception without tearing down the
    session).
  - Scoped lifecycle helper (`scoped_runtime_context`) binds the context AND
    automatically awaits `ctx.aclose()` on exit.
  - `resolve_context_value(attr, fallback)` enables zero-risk incremental
    adoption in existing modules while keeping legacy global callers intact.
"""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator, TypeVar

from core.disposable import AsyncCleanup, AsyncDisposable, AsyncDisposableStore, Disposable
from core.errors import BrowserSessionError, ConfigValidationError

T = TypeVar("T")

DEFAULT_RUNTIME_SCOPE: str = "shared"
VALID_RUNTIME_MODES: tuple[str, ...] = ("legacy", "v2")

_CURRENT_RUNTIME_CONTEXT: ContextVar[BrowserRuntimeContext | None] = ContextVar(
    "afb_runtime_context",
    default=None,
)


def _normalize_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise ConfigValidationError(
            f"runtime_context.engine_mode must be 'legacy' or 'v2', got {mode!r}"
        )
    norm = mode.strip().lower()
    if norm not in VALID_RUNTIME_MODES:
        raise ConfigValidationError(
            f"runtime_context.engine_mode must be one of {VALID_RUNTIME_MODES}, got {mode!r}"
        )
    return norm


def _validate_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{field_name} must be a non-empty string, got {value!r}")
    return value.strip()


@dataclass
class BrowserRuntimeContext:
    """Per-session dependency container replacing process-wide mutable globals."""

    session_id: str = "default"
    scope: str = DEFAULT_RUNTIME_SCOPE
    engine_mode: str = "legacy"

    # Browser & perception runtime handles
    page: Any = None
    browser_context: Any = None
    selector_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    compressor: Any = None
    metrics: Any = None

    # Optional orchestration & LLM failover handles
    failover_chain: list[Any] = field(default_factory=list)
    worker_chain: list[Any] = field(default_factory=list)
    vision_chain: list[Any] = field(default_factory=list)
    breaker: Any = None
    health_tracker: Any = None
    invoke_fn: Any = None

    # Scoped safety & extensible metadata
    objective: str = ""
    allowed_domains: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Lifecycle cleanup store
    disposables: AsyncDisposableStore = field(default_factory=AsyncDisposableStore)

    def __post_init__(self) -> None:
        self.session_id = _validate_non_empty_str(self.session_id, "runtime_context.session_id")
        self.scope = _validate_non_empty_str(self.scope, "runtime_context.scope")
        self.engine_mode = _normalize_mode(self.engine_mode)

    @property
    def is_closed(self) -> bool:
        return self.disposables.is_disposed

    def set_page(self, page: Any) -> None:
        self.page = page

    def set_selector_map(self, selector_map: dict[str, dict[str, Any]]) -> None:
        self.selector_map = selector_map

    def set_engine_mode(self, mode: str) -> str:
        self.engine_mode = _normalize_mode(mode)
        return self.engine_mode

    def register_cleanup(
        self,
        item: AsyncDisposable | Disposable | AsyncCleanup,
    ) -> AsyncDisposable:
        """Attach a sync or async cleanup callback to this context's lifecycle."""
        return self.disposables.add(item)

    async def aclose(self) -> None:
        """Dispose all resources registered on this context in LIFO order."""
        await self.disposables.adispose()

    def snapshot_info(self) -> dict[str, Any]:
        """Return a safe, secret-free diagnostic summary of this context."""
        return {
            "session_id": self.session_id,
            "scope": self.scope,
            "engine_mode": self.engine_mode,
            "has_page": self.page is not None,
            "has_browser_context": self.browser_context is not None,
            "selector_count": len(self.selector_map),
            "has_compressor": self.compressor is not None,
            "has_metrics": self.metrics is not None,
            "allowed_domains_count": len(self.allowed_domains),
            "closed": self.is_closed,
        }


def get_current_context() -> BrowserRuntimeContext | None:
    """Return the `BrowserRuntimeContext` bound to the current task, or `None`."""
    return _CURRENT_RUNTIME_CONTEXT.get()


def require_current_context() -> BrowserRuntimeContext:
    """Return the active `BrowserRuntimeContext` or raise `BrowserSessionError`."""
    ctx = _CURRENT_RUNTIME_CONTEXT.get()
    if ctx is None:
        raise BrowserSessionError(
            "No active BrowserRuntimeContext bound in the current execution context."
        )
    return ctx


@contextmanager
def use_runtime_context(ctx: BrowserRuntimeContext) -> Iterator[BrowserRuntimeContext]:
    """Temporarily bind `ctx` to the current execution context without closing it on exit.

    Supports arbitrary nesting and always restores the previous context on exit
    or exception.
    """
    token = _CURRENT_RUNTIME_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT_RUNTIME_CONTEXT.reset(token)


@asynccontextmanager
async def scoped_runtime_context(
    ctx: BrowserRuntimeContext,
) -> AsyncIterator[BrowserRuntimeContext]:
    """Bind `ctx` for the duration of an `async with` block and `aclose()` it on exit."""
    token = _CURRENT_RUNTIME_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        try:
            await ctx.aclose()
        finally:
            _CURRENT_RUNTIME_CONTEXT.reset(token)


async def run_in_context(
    ctx: BrowserRuntimeContext,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke a sync or async callable with `ctx` bound as the active runtime context."""
    with use_runtime_context(ctx):
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def resolve_context_value(attr: str, fallback: T) -> T:
    """Read `attr` from the active `BrowserRuntimeContext` if set, else return `fallback`.

    Enables incremental migration away from module-level globals while remaining
    100% backward-compatible with legacy callers that have not bound a context.
    """
    ctx = _CURRENT_RUNTIME_CONTEXT.get()
    if ctx is None:
        return fallback
    value = getattr(ctx, attr, None)
    return fallback if value is None else value


__all__ = [
    "DEFAULT_RUNTIME_SCOPE",
    "VALID_RUNTIME_MODES",
    "BrowserRuntimeContext",
    "get_current_context",
    "require_current_context",
    "resolve_context_value",
    "run_in_context",
    "scoped_runtime_context",
    "use_runtime_context",
]
