"""Comprehensive real-world unit tests for src/core/errors.py and src/core/retries.py."""

from __future__ import annotations

import asyncio
import json
import math

import pytest

import core
from core import (
    DEFAULT_RETRY_CONFIG,
    ActionExecutionError,
    AgentBrowseError,
    BrowserSessionError,
    BrowserSessionGoneError,
    ConfigValidationError,
    ErrorCode,
    ModelAttempt,
    ModelProviderError,
    NonRetryableError,
    PerceptionError,
    ResolvedRetryConfig,
    RetryOptions,
    SafetyPolicyError,
    classify_http_status,
    is_error_retryable,
    is_non_retryable,
    is_transient_error,
    jitter_backoff_ms,
    jitter_backoff_s,
    resolve_retry_config,
    to_error_message,
    try_n,
    validate_retry_options,
)


# ── 1. Public Barrel Integrity ─────────────────────────────────────────────────


def test_core_barrel_exports_match_all():
    assert len(core.__all__) > 0
    assert len(core.__all__) == len(set(core.__all__))
    for name in core.__all__:
        assert hasattr(core, name), f"Missing export {name} on core"


# ── 2. Typed Domain Error Hierarchy & Serialization ────────────────────────────


def test_base_error_cause_chaining_defensive_details_and_json_serialization():
    root_cause = TimeoutError("socket timed out")
    source_details = {"step": 4, "url": "https://example.test"}
    err = AgentBrowseError(
        "operation failed",
        code=ErrorCode.PERCEPTION_FAILED,
        retryable=True,
        cause=root_cause,
        details=source_details,
    )
    # Mutating caller dict must not mutate internal error state
    source_details["step"] = 999
    assert err.details["step"] == 4
    assert err.__cause__ is root_cause
    assert err.code is ErrorCode.PERCEPTION_FAILED
    assert err.retryable is True

    payload = err.to_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded == {
        "error_type": "AgentBrowseError",
        "code": "perception_failed",
        "message": "operation failed",
        "retryable": True,
        "details": {"step": 4, "url": "https://example.test"},
    }


def test_non_retryable_config_and_safety_errors():
    cfg_err = ConfigValidationError("max_attempts must be >= 1")
    assert isinstance(cfg_err, NonRetryableError)
    assert isinstance(cfg_err, ValueError)
    assert cfg_err.retryable is False
    assert cfg_err.code is ErrorCode.CONFIG_INVALID

    safety_err = SafetyPolicyError(
        "blocked private metadata host",
        policy="url_policy",
        violation="private_host",
        target="http://169.254.169.254/latest",
    )
    assert isinstance(safety_err, NonRetryableError)
    assert safety_err.retryable is False
    assert safety_err.code is ErrorCode.SAFETY_BLOCKED
    assert safety_err.policy == "url_policy"
    assert safety_err.violation == "private_host"
    assert safety_err.to_dict()["details"]["violation"] == "private_host"


def test_browser_session_perception_and_action_errors():
    gone = BrowserSessionGoneError("sess-v2")
    assert isinstance(gone, BrowserSessionError)
    assert gone.session_id == "sess-v2"
    assert gone.code is ErrorCode.BROWSER_SESSION_GONE
    assert gone.retryable is False
    assert "sess-v2" in str(gone)

    perc = PerceptionError("ax tree timeout", tier=2, url="https://app.test")
    assert perc.tier == 2
    assert perc.url == "https://app.test"
    assert perc.retryable is True
    assert perc.code is ErrorCode.PERCEPTION_FAILED

    stale = ActionExecutionError("detached node", action="click", element_id="e7", stale_ref=True)
    assert stale.code is ErrorCode.STALE_REF
    assert stale.retryable is True
    assert stale.stale_ref is True

    disabled = ActionExecutionError("element disabled", action="click", element_id="e8")
    assert disabled.code is ErrorCode.ACTION_FAILED
    assert disabled.retryable is False


@pytest.mark.parametrize(
    ("status", "expected_code", "expected_retryable"),
    [
        (401, ErrorCode.MODEL_AUTH, False),
        (403, ErrorCode.MODEL_AUTH, False),
        (404, ErrorCode.MODEL_NOT_FOUND, False),
        (429, ErrorCode.MODEL_RATE_LIMIT, True),
        (408, ErrorCode.MODEL_TIMEOUT, True),
        (504, ErrorCode.MODEL_TIMEOUT, True),
        (400, ErrorCode.MODEL_BAD_REQUEST, False),
        (422, ErrorCode.MODEL_BAD_REQUEST, False),
        (500, ErrorCode.MODEL_PROVIDER_ERROR, True),
        (502, ErrorCode.MODEL_PROVIDER_ERROR, True),
        (503, ErrorCode.MODEL_PROVIDER_ERROR, True),
        (None, ErrorCode.MODEL_UNKNOWN, False),
    ],
)
def test_classify_http_status_and_model_provider_error(status, expected_code, expected_retryable):
    code, retryable = classify_http_status(status)
    assert code is expected_code
    assert retryable is expected_retryable

    attempts = [ModelAttempt(model="primary-model", error="429 too many requests", status=429)]
    err = ModelProviderError(
        "model failed",
        model="fallback-model",
        provider="groq",
        status=status,
        attempts=attempts,
    )
    assert err.code is expected_code
    assert err.retryable is expected_retryable
    assert len(err.attempts) == 1
    serialized = err.to_dict()
    assert serialized["details"]["model"] == "fallback-model"
    assert serialized["details"]["attempts"][0]["model"] == "primary-model"


def test_cause_chain_classification_foreign_non_retryable_and_cycle_guard():
    class ForeignNonRetryable(Exception):
        pass

    ForeignNonRetryable.__name__ = "NonRetryableError"

    assert is_non_retryable(ForeignNonRetryable("stop")) is True
    assert is_error_retryable(ForeignNonRetryable("stop")) is False

    # Nested non-retryable cause overrides outer error
    inner = SafetyPolicyError("blocked", policy="url", violation="blocked_host")
    outer = RuntimeError("wrapper")
    outer.__cause__ = inner
    assert is_non_retryable(outer) is True
    assert is_error_retryable(outer) is False

    # Nested retryable AgentBrowseError is surfaced when no non-retryable error exists
    retryable_inner = ModelProviderError("429", model="m1", status=429)
    retryable_outer = RuntimeError("failed leg")
    retryable_outer.__cause__ = retryable_inner
    assert is_error_retryable(retryable_outer) is True

    # Self-referential / cyclic cause chain must terminate safely
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert is_non_retryable(a) is False
    assert is_error_retryable(a) is False

    # to_error_message handles empty messages and non-exception values cleanly
    assert to_error_message(ValueError()) == "ValueError"
    assert to_error_message(RuntimeError("boom")) == "boom"
    assert to_error_message(404) == "404"


# ── 3. Retry Validation, Resolution & Full-Jitter Backoff ──────────────────────


def test_resolve_retry_config_merges_and_validates_eagerly():
    resolved = resolve_retry_config(RetryOptions(max_attempts=5))
    assert resolved == ResolvedRetryConfig(max_attempts=5, base_delay_ms=100.0, max_delay_ms=3000.0)

    # Cross-field violation against default max_delay_ms (3000ms)
    with pytest.raises(ConfigValidationError, match="base_delay_ms"):
        validate_retry_options(RetryOptions(base_delay_ms=5000.0), defaults=DEFAULT_RETRY_CONFIG)

    # Cross-field violation against default base_delay_ms (100ms)
    with pytest.raises(ConfigValidationError, match="base_delay_ms"):
        resolve_retry_config(RetryOptions(max_delay_ms=50.0))


@pytest.mark.parametrize(
    "bad_options",
    [
        RetryOptions(max_attempts=0),
        RetryOptions(max_attempts=-2),
        RetryOptions(max_attempts=True),  # type: ignore[arg-type]
        RetryOptions(max_attempts=1.5),  # type: ignore[arg-type]
        RetryOptions(base_delay_ms=0),
        RetryOptions(base_delay_ms=-10),
        RetryOptions(base_delay_ms=math.nan),
        RetryOptions(base_delay_ms=math.inf),
        RetryOptions(base_delay_ms=True),  # type: ignore[arg-type]
        RetryOptions(max_delay_ms=0),
        RetryOptions(max_delay_ms=-5),
        RetryOptions(max_delay_ms=math.nan),
        RetryOptions(base_delay_ms=500.0, max_delay_ms=200.0),
    ],
)
def test_validate_retry_options_rejects_invalid_values(bad_options):
    with pytest.raises(ConfigValidationError):
        validate_retry_options(bad_options)


def test_jitter_backoff_bounds_cap_and_overflow_safety():
    # rng=1.0 hits the exact upper bound min(2**attempt * base, max)
    assert jitter_backoff_ms(1, 100.0, 3000.0, rng=lambda: 1.0) == pytest.approx(200.0)
    assert jitter_backoff_ms(2, 100.0, 3000.0, rng=lambda: 1.0) == pytest.approx(400.0)
    assert jitter_backoff_ms(3, 100.0, 3000.0, rng=lambda: 1.0) == pytest.approx(800.0)
    assert jitter_backoff_ms(5, 100.0, 3000.0, rng=lambda: 1.0) == pytest.approx(3000.0)
    # Huge attempt must not overflow float exponentiation
    assert jitter_backoff_ms(5000, 100.0, 3000.0, rng=lambda: 1.0) == pytest.approx(3000.0)
    # rng=0.5 yields half the upper bound; seconds helper scales by 1/1000
    assert jitter_backoff_ms(2, 100.0, 3000.0, rng=lambda: 0.5) == pytest.approx(200.0)
    assert jitter_backoff_s(2, 100.0, 3000.0, rng=lambda: 0.5) == pytest.approx(0.2)
    assert jitter_backoff_ms(1, 100.0, 3000.0, rng=lambda: 0.0) == pytest.approx(0.0)

    with pytest.raises(ConfigValidationError):
        jitter_backoff_ms(0, 100.0, 3000.0)
    with pytest.raises(ConfigValidationError):
        jitter_backoff_ms(True, 100.0, 3000.0)  # type: ignore[arg-type]


def test_is_transient_error_classification():
    assert is_transient_error(TimeoutError("timed out")) is True
    assert is_transient_error(ConnectionResetError("peer reset")) is True
    assert is_transient_error(RuntimeError("Execution context was destroyed")) is True
    assert is_transient_error(RuntimeError("HTTP 429 rate limit exceeded")) is True
    assert is_transient_error(PerceptionError("stale dom", retryable=True)) is True
    assert is_transient_error(ActionExecutionError("detached", stale_ref=True)) is True
    assert is_transient_error(ModelProviderError("503", model="m", status=503)) is True

    assert is_transient_error(ValueError("bad arg")) is False
    assert is_transient_error(KeyError("missing")) is False
    assert is_transient_error(asyncio.CancelledError()) is False
    assert is_transient_error(BrowserSessionGoneError("s1")) is False
    assert is_transient_error(SafetyPolicyError("blocked", policy="url", violation="blocked_host")) is False
    assert is_transient_error(ModelProviderError("bad key", model="m", status=401)) is False
    assert is_transient_error(RuntimeError("completely unrelated logic error")) is False

    # Wrapped transient cause is transient; wrapped NonRetryableError is not
    wrapped_timeout = RuntimeError("action failed")
    wrapped_timeout.__cause__ = TimeoutError("cdp")
    assert is_transient_error(wrapped_timeout) is True

    wrapped_stop = RuntimeError("timeout while stopped")
    wrapped_stop.__cause__ = NonRetryableError("do not retry")
    assert is_transient_error(wrapped_stop) is False


# ── 4. Async try_n Behavior ────────────────────────────────────────────────────


async def test_try_n_succeeds_after_transient_failures_with_deterministic_backoff():
    attempts_seen: list[int] = []
    sleeps_seen: list[float] = []
    retry_events: list[tuple[int, str, float]] = []

    async def fake_sleep(delay_s: float) -> None:
        sleeps_seen.append(delay_s)

    async def flaky(attempt: int) -> str:
        attempts_seen.append(attempt)
        if attempt < 3:
            raise TimeoutError(f"hiccup #{attempt}")
        return f"ok-on-{attempt}"

    result = await try_n(
        4,
        flaky,
        base_delay_ms=100.0,
        max_delay_ms=1000.0,
        should_retry=lambda exc, _next: is_transient_error(exc),
        on_retry=lambda att, exc, d: retry_events.append((att, str(exc), d)),
        sleep_fn=fake_sleep,
        rng=lambda: 0.5,
    )

    assert result == "ok-on-3"
    assert attempts_seen == [1, 2, 3]
    # Attempt 1: 0.5 * 200ms = 0.1s; Attempt 2: 0.5 * 400ms = 0.2s
    assert sleeps_seen == [pytest.approx(0.1), pytest.approx(0.2)]
    assert len(retry_events) == 2
    assert retry_events[0][0] == 1
    assert retry_events[1][0] == 2


async def test_try_n_validates_config_eagerly_before_calling_fn():
    called = 0

    async def should_never_run(attempt: int) -> str:
        nonlocal called
        called += 1
        return "unreachable"

    with pytest.raises(ConfigValidationError):
        await try_n(0, should_never_run)
    with pytest.raises(ConfigValidationError):
        await try_n(3, should_never_run, base_delay_ms=5000.0)
    assert called == 0


async def test_try_n_exhausts_attempts_and_reraises_last_error():
    seen: list[int] = []

    async def always_fails(attempt: int) -> None:
        seen.append(attempt)
        raise ModelProviderError(f"503 on {attempt}", model="m", status=503)

    async def noop_sleep(_: float) -> None:
        return None

    with pytest.raises(ModelProviderError, match="503 on 3"):
        await try_n(3, always_fails, sleep_fn=noop_sleep, rng=lambda: 0.0)
    assert seen == [1, 2, 3]


async def test_try_n_short_circuits_non_retryable_domain_and_cancellation_errors():
    async def noop_sleep(_: float) -> None:
        raise AssertionError("sleep must not be called for non-retryable errors")

    # 1. NonRetryableError short-circuits even when should_retry returns True
    calls = 0

    async def raises_non_retryable(attempt: int) -> None:
        nonlocal calls
        calls += 1
        raise NonRetryableError("fatal business rejection")

    with pytest.raises(NonRetryableError):
        await try_n(
            5,
            raises_non_retryable,
            should_retry=lambda _e, _n: True,
            sleep_fn=noop_sleep,
        )
    assert calls == 1

    # 2. BrowserSessionGoneError & ModelProviderError(401) short-circuit by default
    for domain_err in (
        BrowserSessionGoneError("sess-dead"),
        ModelProviderError("invalid key", model="m", status=401),
        SafetyPolicyError("blocked", policy="url", violation="blocked_host"),
        ValueError("caller bug"),
    ):
        count = 0

        async def _fn(attempt: int, err=domain_err) -> None:
            nonlocal count
            count += 1
            raise err

        with pytest.raises(type(domain_err)):
            await try_n(4, _fn, sleep_fn=noop_sleep)
        assert count == 1

    # 3. asyncio.CancelledError propagates immediately without catching or sleeping
    cancel_calls = 0

    async def cancels(attempt: int) -> None:
        nonlocal cancel_calls
        cancel_calls += 1
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await try_n(3, cancels, sleep_fn=noop_sleep)
    assert cancel_calls == 1


async def test_try_n_attempt_aware_predicate_and_config_inheritance():
    seen_next_attempts: list[int] = []
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    async def fails(attempt: int) -> None:
        raise RuntimeError(f"err-{attempt}")

    def predicate(_exc: Exception, next_attempt: int) -> bool:
        seen_next_attempts.append(next_attempt)
        # Allow retry for attempt 2, stop before attempt 3
        return next_attempt < 3

    cfg = ResolvedRetryConfig(max_attempts=5, base_delay_ms=50.0, max_delay_ms=200.0)
    with pytest.raises(RuntimeError, match="err-2"):
        await try_n(
            None,
            fails,
            config=cfg,
            should_retry=predicate,
            sleep_fn=fake_sleep,
            rng=lambda: 1.0,
        )
    assert seen_next_attempts == [2, 3]
    assert sleeps == [pytest.approx(0.1)]


async def test_try_n_real_asyncio_sleep_smoke():
    """Verify real asyncio.sleep integration (tiny 1ms–5ms window, no mocks)."""
    calls = 0

    async def recover_on_second(attempt: int) -> str:
        nonlocal calls
        calls += 1
        if attempt == 1:
            raise ConnectionError("transient reset")
        return "recovered"

    out = await try_n(
        2,
        recover_on_second,
        base_delay_ms=1.0,
        max_delay_ms=5.0,
        should_retry=lambda exc, _n: is_transient_error(exc),
    )
    assert out == "recovered"
    assert calls == 2
