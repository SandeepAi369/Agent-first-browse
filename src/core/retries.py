"""Centralized, eagerly-validated retry primitive with full-jitter backoff.

Adapts Cloudflare's `retries.ts` (`RetryOptions`, `validateRetryOptions`,
`resolveRetryConfig`, AWS full-jitter `jitterBackoff`, and attempt-aware `tryN`)
to Python 3.11+ `asyncio`, integrating cleanly with `core.errors`.

Key guarantees:
  - Eager configuration validation: invalid bounds or cross-field mismatches
    (`base_delay_ms > max_delay_ms`, `max_attempts < 1`, non-integer/bool
    attempts, NaN/Inf delays) raise `ConfigValidationError` BEFORE calling `fn`.
  - AWS Full Jitter backoff: `random() * min(2**attempt * base_delay_ms, max_delay_ms)`.
  - Never retries `asyncio.CancelledError`, `KeyboardInterrupt`, `SystemExit`,
    or `NonRetryableError` — even if a caller omits `should_retry`.
  - Injectable `rng` and `sleep_fn` enable fast, deterministic unit tests.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from core.errors import (
    AgentBrowseError,
    ConfigValidationError,
    NonRetryableError,
    is_non_retryable,
)

T = TypeVar("T")

DEFAULT_RETRY_MAX_ATTEMPTS: int = 3
DEFAULT_RETRY_BASE_DELAY_MS: float = 100.0
DEFAULT_RETRY_MAX_DELAY_MS: float = 3000.0

_TRANSIENT_MESSAGE_HINTS: tuple[str, ...] = (
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
    "detached",
    "intercepted",
    "stale",
    "not attached",
    "node is not",
)

ShouldRetryFn = Callable[[Exception, int], bool]
OnRetryFn = Callable[[int, Exception, float], None]
SleepFn = Callable[[float], Awaitable[None]]
RngFn = Callable[[], float]


@dataclass(frozen=True)
class RetryOptions:
    """Partial per-call retry overrides (unset fields inherit from defaults)."""

    max_attempts: int | None = None
    base_delay_ms: float | None = None
    max_delay_ms: float | None = None


@dataclass(frozen=True)
class ResolvedRetryConfig:
    """Fully resolved, eagerly-validated retry configuration."""

    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS
    base_delay_ms: float = DEFAULT_RETRY_BASE_DELAY_MS
    max_delay_ms: float = DEFAULT_RETRY_MAX_DELAY_MS

    def __post_init__(self) -> None:
        validate_retry_options(
            RetryOptions(
                max_attempts=self.max_attempts,
                base_delay_ms=self.base_delay_ms,
                max_delay_ms=self.max_delay_ms,
            )
        )


def _validate_int_attempts(value: object, field_name: str = "retry.max_attempts") -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"{field_name} must be an integer >= 1, got {value!r}")
    if value < 1:
        raise ConfigValidationError(f"{field_name} must be >= 1, got {value}")


def _validate_positive_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{field_name} must be a finite number > 0, got {value!r}")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0:
        raise ConfigValidationError(f"{field_name} must be a finite number > 0, got {value!r}")
    return fval


def validate_retry_options(
    options: RetryOptions | ResolvedRetryConfig,
    defaults: ResolvedRetryConfig | None = None,
) -> None:
    """Validate retry options eagerly, including cross-field constraints against defaults."""
    if options.max_attempts is not None:
        _validate_int_attempts(options.max_attempts)

    resolved_base: float | None = None
    if options.base_delay_ms is not None:
        resolved_base = _validate_positive_finite(options.base_delay_ms, "retry.base_delay_ms")
    elif defaults is not None:
        resolved_base = float(defaults.base_delay_ms)

    resolved_max: float | None = None
    if options.max_delay_ms is not None:
        resolved_max = _validate_positive_finite(options.max_delay_ms, "retry.max_delay_ms")
    elif defaults is not None:
        resolved_max = float(defaults.max_delay_ms)

    if resolved_base is not None and resolved_max is not None and resolved_base > resolved_max:
        raise ConfigValidationError(
            f"retry.base_delay_ms ({resolved_base}) must be <= retry.max_delay_ms ({resolved_max})"
        )


DEFAULT_RETRY_CONFIG = ResolvedRetryConfig()


def resolve_retry_config(
    options: RetryOptions | ResolvedRetryConfig | None = None,
    defaults: ResolvedRetryConfig = DEFAULT_RETRY_CONFIG,
) -> ResolvedRetryConfig:
    """Merge partial `RetryOptions` with `defaults` and validate eagerly."""
    if options is None:
        return defaults
    validate_retry_options(options, defaults=defaults)
    return ResolvedRetryConfig(
        max_attempts=(
            defaults.max_attempts if options.max_attempts is None else options.max_attempts
        ),
        base_delay_ms=(
            defaults.base_delay_ms
            if options.base_delay_ms is None
            else float(options.base_delay_ms)
        ),
        max_delay_ms=(
            defaults.max_delay_ms
            if options.max_delay_ms is None
            else float(options.max_delay_ms)
        ),
    )


def jitter_backoff_ms(
    attempt: int,
    base_delay_ms: float,
    max_delay_ms: float,
    *,
    rng: RngFn | None = None,
) -> float:
    """Compute AWS Full-Jitter exponential backoff delay in milliseconds.

    Args:
        attempt: 1-indexed failed attempt number (must be >= 1).
        base_delay_ms: Base backoff multiplier in ms (> 0).
        max_delay_ms: Maximum delay cap in ms (>= base_delay_ms).
        rng: Optional `[0.0, 1.0)` generator for deterministic tests.
    """
    _validate_int_attempts(attempt, field_name="attempt")
    base = _validate_positive_finite(base_delay_ms, "retry.base_delay_ms")
    cap = _validate_positive_finite(max_delay_ms, "retry.max_delay_ms")
    if base > cap:
        raise ConfigValidationError(
            f"retry.base_delay_ms ({base}) must be <= retry.max_delay_ms ({cap})"
        )

    # Cap exponent to avoid float overflow on very large attempt counts
    safe_exp = min(attempt, 60)
    upper_bound = min((2.0**safe_exp) * base, cap)
    sample = (rng or random.random)()
    clamped = max(0.0, min(1.0, float(sample)))
    return clamped * upper_bound


def jitter_backoff_s(
    attempt: int,
    base_delay_ms: float,
    max_delay_ms: float,
    *,
    rng: RngFn | None = None,
) -> float:
    """Compute AWS Full-Jitter exponential backoff delay in seconds."""
    return jitter_backoff_ms(attempt, base_delay_ms, max_delay_ms, rng=rng) / 1000.0


def _walk_causes(exc: BaseException, max_depth: int = 5):
    seen: set[int] = set()
    cur: BaseException | None = exc
    depth = 0
    while cur is not None and depth <= max_depth:
        oid = id(cur)
        if oid in seen:
            break
        seen.add(oid)
        yield cur
        cur = cur.__cause__ or cur.__context__
        depth += 1


def is_transient_error(exc: BaseException) -> bool:
    """Classify whether an exception represents a transient, retryable failure.

    Precedence:
      1. Explicit non-retryable / cancellation / logic errors -> False.
      2. Typed `AgentBrowseError` in chain -> honors `.retryable` directly.
      3. Built-in `TimeoutError` / `ConnectionError` -> True.
      4. Known transient Playwright/HTTP/network message hints -> True.
    """
    if is_non_retryable(exc):
        return False

    for item in _walk_causes(exc):
        if isinstance(item, AgentBrowseError):
            return item.retryable
        if isinstance(item, (TimeoutError, ConnectionError)):
            return True

    msg = str(exc).lower()
    return any(hint in msg for hint in _TRANSIENT_MESSAGE_HINTS)


def _resolve_try_n_config(
    max_attempts: int | None,
    base_delay_ms: float | None,
    max_delay_ms: float | None,
    config: RetryOptions | ResolvedRetryConfig | None,
) -> ResolvedRetryConfig:
    base_cfg = resolve_retry_config(config) if config is not None else DEFAULT_RETRY_CONFIG
    overrides = RetryOptions(
        max_attempts=max_attempts,
        base_delay_ms=base_delay_ms,
        max_delay_ms=max_delay_ms,
    )
    return resolve_retry_config(overrides, defaults=base_cfg)


def _is_hard_non_retryable(exc: BaseException) -> bool:
    """True when `exc` (or its cause chain) is an explicit `NonRetryableError`."""
    for item in _walk_causes(exc):
        if isinstance(item, NonRetryableError) or type(item).__name__ == "NonRetryableError":
            return True
    return False


def _default_allows_retry(exc: Exception) -> bool:
    if is_non_retryable(exc):
        return False
    for item in _walk_causes(exc):
        if isinstance(item, AgentBrowseError):
            return item.retryable
    return True


async def try_n(
    max_attempts: int | None,
    fn: Callable[[int], Awaitable[T]],
    *,
    base_delay_ms: float | None = None,
    max_delay_ms: float | None = None,
    config: RetryOptions | ResolvedRetryConfig | None = None,
    should_retry: ShouldRetryFn | None = None,
    on_retry: OnRetryFn | None = None,
    sleep_fn: SleepFn | None = None,
    rng: RngFn | None = None,
) -> T:
    """Retry async `fn(attempt)` up to `max_attempts` with full-jitter backoff.

    Args:
        max_attempts: Total attempts including the first (or None to inherit `config`/default).
        fn: Async callable receiving the 1-indexed attempt number.
        base_delay_ms: Optional override for base backoff delay in ms.
        max_delay_ms: Optional override for max backoff cap in ms.
        config: Optional `RetryOptions` or `ResolvedRetryConfig` base settings.
        should_retry: Optional predicate `(exc, next_attempt) -> bool`. When omitted,
            retries all non-fatal exceptions except non-retryable builtins,
            `NonRetryableError`, and `AgentBrowseError(retryable=False)`.
        on_retry: Optional observer `(failed_attempt, exc, delay_s) -> None`.
        sleep_fn: Optional async sleep replacement for deterministic testing.
        rng: Optional `[0.0, 1.0)` generator for deterministic backoff testing.
    """
    resolved = _resolve_try_n_config(max_attempts, base_delay_ms, max_delay_ms, config)
    sleeper = sleep_fn or asyncio.sleep

    attempt = 1
    while True:
        try:
            return await fn(attempt)
        except Exception as exc:
            next_attempt = attempt + 1
            if next_attempt > resolved.max_attempts:
                raise
            if _is_hard_non_retryable(exc):
                raise
            if should_retry is not None:
                if not should_retry(exc, next_attempt):
                    raise
            elif not _default_allows_retry(exc):
                raise

            delay_s = jitter_backoff_s(
                attempt,
                resolved.base_delay_ms,
                resolved.max_delay_ms,
                rng=rng,
            )
            if on_retry is not None:
                on_retry(attempt, exc, delay_s)
            await sleeper(delay_s)
            attempt = next_attempt


__all__ = [
    "DEFAULT_RETRY_BASE_DELAY_MS",
    "DEFAULT_RETRY_CONFIG",
    "DEFAULT_RETRY_MAX_ATTEMPTS",
    "DEFAULT_RETRY_MAX_DELAY_MS",
    "OnRetryFn",
    "ResolvedRetryConfig",
    "RetryOptions",
    "RngFn",
    "ShouldRetryFn",
    "SleepFn",
    "is_transient_error",
    "jitter_backoff_ms",
    "jitter_backoff_s",
    "resolve_retry_config",
    "try_n",
    "validate_retry_options",
]
