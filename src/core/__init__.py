"""core — Shared architectural primitives & typed contracts for Agent-first-browse.

Provides lightweight, zero-dependency foundations shared across the browser engine,
evaluation connector, and orchestration layers:
  - `core.errors`: Typed domain exception hierarchy with stable `ErrorCode` values,
    explicit `.retryable` metadata, and JSON-safe serialization.
  - `core.retries`: Eagerly-validated full-jitter exponential backoff & async `try_n`.
"""

from __future__ import annotations

from core.errors import (
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
    SafetyPolicyError,
    classify_http_status,
    is_error_retryable,
    is_non_retryable,
    to_error_message,
)
from core.retries import (
    DEFAULT_RETRY_BASE_DELAY_MS,
    DEFAULT_RETRY_CONFIG,
    DEFAULT_RETRY_MAX_ATTEMPTS,
    DEFAULT_RETRY_MAX_DELAY_MS,
    OnRetryFn,
    ResolvedRetryConfig,
    RetryOptions,
    RngFn,
    ShouldRetryFn,
    SleepFn,
    is_transient_error,
    jitter_backoff_ms,
    jitter_backoff_s,
    resolve_retry_config,
    try_n,
    validate_retry_options,
)

__all__ = [
    "DEFAULT_RETRY_BASE_DELAY_MS",
    "DEFAULT_RETRY_CONFIG",
    "DEFAULT_RETRY_MAX_ATTEMPTS",
    "DEFAULT_RETRY_MAX_DELAY_MS",
    "ActionExecutionError",
    "AgentBrowseError",
    "BrowserSessionError",
    "BrowserSessionGoneError",
    "ConfigValidationError",
    "ErrorCode",
    "ModelAttempt",
    "ModelProviderError",
    "NonRetryableError",
    "OnRetryFn",
    "PerceptionError",
    "ResolvedRetryConfig",
    "RetryOptions",
    "RngFn",
    "SafetyPolicyError",
    "ShouldRetryFn",
    "SleepFn",
    "classify_http_status",
    "is_error_retryable",
    "is_non_retryable",
    "is_transient_error",
    "jitter_backoff_ms",
    "jitter_backoff_s",
    "resolve_retry_config",
    "to_error_message",
    "try_n",
    "validate_retry_options",
]
