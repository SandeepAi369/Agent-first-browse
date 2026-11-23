"""Composable resource disposal & listener-isolated typed event emission.

Adapts Cloudflare's `core/events.ts` (`Disposable`, `toDisposable`,
`DisposableStore`, `Emitter<T>`) to Python 3.11+ with first-class `asyncio`
support (`AsyncDisposable`, `AsyncDisposableStore`).

Key guarantees:
  - Idempotent wrappers (`to_disposable`, `to_async_disposable`) ensure each
    cleanup callback runs at most once even if disposed repeatedly.
  - LIFO disposal order (`DisposableStore`, `AsyncDisposableStore`) unwinds
    resources in reverse acquisition order and isolates exceptions per item so
    one failing teardown never leaks remaining resources.
  - `Emitter[T]` snapshots listeners before firing and isolates listener
    exceptions so one bad subscriber cannot break sibling listeners or mutate
    the set mid-dispatch.
"""

from __future__ import annotations

import inspect
import logging
from typing import Awaitable, Callable, Generic, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger("core.disposable")

T = TypeVar("T")

ErrorHandler = Callable[[Exception, object], None]
SyncCleanup = Callable[[], None]
AsyncCleanup = Callable[[], Awaitable[None] | None]


@runtime_checkable
class Disposable(Protocol):
    """Synchronous resource supporting deterministic cleanup."""

    def dispose(self) -> None: ...


@runtime_checkable
class AsyncDisposable(Protocol):
    """Asynchronous resource supporting deterministic cleanup."""

    async def adispose(self) -> None: ...


class _FnDisposable:
    __slots__ = ("_fn", "_disposed")

    def __init__(self, fn: SyncCleanup) -> None:
        self._fn: SyncCleanup | None = fn
        self._disposed = False

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        fn = self._fn
        self._fn = None
        if fn is not None:
            fn()


class _FnAsyncDisposable:
    __slots__ = ("_fn", "_disposed")

    def __init__(self, fn: AsyncCleanup) -> None:
        self._fn: AsyncCleanup | None = fn
        self._disposed = False

    async def adispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        fn = self._fn
        self._fn = None
        if fn is None:
            return
        maybe_coro = fn()
        if inspect.isawaitable(maybe_coro):
            await maybe_coro


def to_disposable(fn: SyncCleanup) -> Disposable:
    """Wrap a zero-arg synchronous cleanup callback in an idempotent `Disposable`."""
    return _FnDisposable(fn)


def to_async_disposable(fn: AsyncCleanup) -> AsyncDisposable:
    """Wrap a sync or async cleanup callback in an idempotent `AsyncDisposable`."""
    return _FnAsyncDisposable(fn)


def _handle_cleanup_error(
    exc: Exception,
    item: object,
    on_error: ErrorHandler | None,
    context: str,
) -> None:
    if on_error is not None:
        try:
            on_error(exc, item)
            return
        except Exception:
            logger.exception("Error handler failed during %s", context)
            return
    logger.warning("Cleanup error during %s (%s): %s", context, type(exc).__name__, exc)


class DisposableStore:
    """Collects synchronous `Disposable` items and tears them down in LIFO order."""

    def __init__(self, *, on_error: ErrorHandler | None = None) -> None:
        self._items: list[Disposable] = []
        self._disposed: bool = False
        self._on_error: ErrorHandler | None = on_error

    @property
    def is_disposed(self) -> bool:
        return self._disposed

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: Disposable | SyncCleanup) -> Disposable:
        """Register a `Disposable` or callback; disposes immediately if store is closed."""
        disposable = item if isinstance(item, Disposable) else to_disposable(item)
        if self._disposed:
            try:
                disposable.dispose()
            except Exception as exc:
                _handle_cleanup_error(exc, disposable, self._on_error, "post-dispose add")
            return disposable
        self._items.append(disposable)
        return disposable

    def dispose(self) -> None:
        """Dispose all registered items in LIFO order, isolating errors per item."""
        if self._disposed:
            return
        self._disposed = True
        while self._items:
            item = self._items.pop()
            try:
                item.dispose()
            except Exception as exc:
                _handle_cleanup_error(exc, item, self._on_error, "DisposableStore.dispose")

    def __enter__(self) -> DisposableStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.dispose()


class AsyncDisposableStore:
    """Collects sync and async disposables and tears them down in LIFO order."""

    def __init__(self, *, on_error: ErrorHandler | None = None) -> None:
        self._items: list[AsyncDisposable] = []
        self._disposed: bool = False
        self._on_error: ErrorHandler | None = on_error

    @property
    def is_disposed(self) -> bool:
        return self._disposed

    def __len__(self) -> int:
        return len(self._items)

    def _coerce(self, item: AsyncDisposable | Disposable | AsyncCleanup) -> AsyncDisposable:
        if isinstance(item, AsyncDisposable):
            return item
        if isinstance(item, Disposable):
            return to_async_disposable(item.dispose)
        return to_async_disposable(item)

    def add(self, item: AsyncDisposable | Disposable | AsyncCleanup) -> AsyncDisposable:
        """Register a sync/async disposable or callback.

        Raises `RuntimeError` if the store has already been disposed so async
        resources are never silently leaked.
        """
        if self._disposed:
            raise RuntimeError("Cannot register resource on an already-disposed AsyncDisposableStore")
        wrapped = self._coerce(item)
        self._items.append(wrapped)
        return wrapped

    async def adispose(self) -> None:
        """Await disposal of all registered items in LIFO order with error isolation."""
        if self._disposed:
            return
        self._disposed = True
        while self._items:
            item = self._items.pop()
            try:
                await item.adispose()
            except Exception as exc:
                _handle_cleanup_error(exc, item, self._on_error, "AsyncDisposableStore.adispose")

    async def __aenter__(self) -> AsyncDisposableStore:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.adispose()


class Emitter(Generic[T]):
    """Typed, listener-isolated synchronous event emitter returning `Disposable` handles."""

    def __init__(self, *, on_error: ErrorHandler | None = None) -> None:
        self._listeners: dict[int, Callable[[T], None]] = {}
        self._next_token: int = 1
        self._disposed: bool = False
        self._on_error: ErrorHandler | None = on_error

    @property
    def is_disposed(self) -> bool:
        return self._disposed

    @property
    def listener_count(self) -> int:
        return len(self._listeners)

    def subscribe(self, listener: Callable[[T], None]) -> Disposable:
        """Subscribe `listener` and return an idempotent `Disposable` unsubscribe handle."""
        if self._disposed:
            return to_disposable(lambda: None)
        token = self._next_token
        self._next_token += 1
        self._listeners[token] = listener
        return to_disposable(lambda: self._listeners.pop(token, None))

    def fire(self, data: T) -> None:
        """Dispatch `data` to a snapshot of active listeners, isolating exceptions."""
        if self._disposed or not self._listeners:
            return
        snapshot = list(self._listeners.values())
        for listener in snapshot:
            try:
                listener(data)
            except Exception as exc:
                _handle_cleanup_error(exc, listener, self._on_error, "Emitter.fire")

    def dispose(self) -> None:
        """Clear all listeners and mark the emitter disposed."""
        if self._disposed:
            return
        self._disposed = True
        self._listeners.clear()


__all__ = [
    "AsyncCleanup",
    "AsyncDisposable",
    "AsyncDisposableStore",
    "Disposable",
    "DisposableStore",
    "Emitter",
    "ErrorHandler",
    "SyncCleanup",
    "to_async_disposable",
    "to_disposable",
]
