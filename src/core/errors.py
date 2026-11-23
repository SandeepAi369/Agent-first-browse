"""Typed domain error hierarchy for Agent-first-browse.

Adapts structured error-classification patterns (stable error codes, explicit
retryability metadata, cause-chain inspection, and non-retryable short-circuits)
to Python 3.11+ without third-party dependencies.

Design principles:
  - Every domain error carries a stable `ErrorCode` (`code`) and explicit
    `retryable: bool` flag so retry loops, failover chains, and MCP/CLI
    transports classify failures deterministically without brittle string checks.
  - `NonRetryableError` (and subclasses like `ConfigValidationError` and
    `SafetyPolicyError`) unconditionally force `retryable = False`.
  - `to_dict()` produces a clean, JSON-serializable payload suitable for MCP
    responses, daemon IPC, and structured observability sinks.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

_MAX_CAUSE_WALK_DEPTH = 5

_NON_RETRYABLE_BUILTINS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    AssertionError,
    SyntaxError,
    KeyboardInterrupt,
    SystemExit,
    asyncio.CancelledError,
)


class ErrorCode(StrEnum):
    """Stable machine-readable error classifications across Agent-first-browse."""

    INTERNAL = "internal"
    NON_RETRYABLE = "non_retryable"
    CONFIG_INVALID = "config_invalid"
    BROWSER_SESSION = "browser_session"
    BROWSER_SESSION_GONE = "browser_session_gone"
    PERCEPTION_FAILED = "perception_failed"
    ACTION_FAILED = "action_failed"
    STALE_REF = "stale_ref"
    SAFETY_BLOCKED = "safety_blocked"
    MODEL_AUTH = "model_auth"
    MODEL_RATE_LIMIT = "model_rate_limit"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_BAD_REQUEST = "model_bad_request"
    MODEL_PROVIDER_ERROR = "model_provider_error"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_UNKNOWN = "model_unknown"


def to_error_message(error: object) -> str:
    """Return a clean human-readable message for any thrown or non-Exception value."""
    if isinstance(error, BaseException):
        msg = str(error).strip()
        return msg if msg else type(error).__name__
    return str(error)


def classify_http_status(status: int | None) -> tuple[ErrorCode, bool]:
    """Map an HTTP status code to `(ErrorCode, retryable)` for model/API calls."""
    if status is None:
        return ErrorCode.MODEL_UNKNOWN, False
    if status in (401, 403):
        return ErrorCode.MODEL_AUTH, False
    if status == 404:
        return ErrorCode.MODEL_NOT_FOUND, False
    if status == 429:
        return ErrorCode.MODEL_RATE_LIMIT, True
    if status in (408, 504):
        return ErrorCode.MODEL_TIMEOUT, True
    if 400 <= status < 500:
        return ErrorCode.MODEL_BAD_REQUEST, False
    if 500 <= status < 600:
        return ErrorCode.MODEL_PROVIDER_ERROR, True
    return ErrorCode.MODEL_UNKNOWN, False


class AgentBrowseError(Exception):
    """Base exception for all typed domain errors in Agent-first-browse."""

    default_code: ErrorCode = ErrorCode.INTERNAL
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        retryable: bool | None = None,
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code: ErrorCode = code if code is not None else self.default_code
        self.retryable: bool = (
            self.default_retryable if retryable is None else bool(retryable)
        )
        self.details: dict[str, Any] = dict(details) if details else {}
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Serialize error metadata for MCP/daemon responses and telemetry."""
        payload: dict[str, Any] = {
            "error_type": type(self).__name__,
            "code": self.code.value,
            "message": str(self),
            "retryable": self.retryable,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


class NonRetryableError(AgentBrowseError):
    """Explicitly fails an operation immediately, skipping all remaining retries."""

    default_code: ErrorCode = ErrorCode.NON_RETRYABLE
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code or self.default_code,
            retryable=False,
            cause=cause,
            details=details,
        )


class ConfigValidationError(NonRetryableError, ValueError):
    """Raised eagerly when retry, engine, or policy configuration is invalid."""

    default_code: ErrorCode = ErrorCode.CONFIG_INVALID


class BrowserSessionError(AgentBrowseError):
    """Base error for browser session lifecycle or transport failures."""

    default_code: ErrorCode = ErrorCode.BROWSER_SESSION
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        session_id: str | None = None,
        code: ErrorCode | None = None,
        retryable: bool | None = None,
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(details) if details else {}
        if session_id is not None:
            merged["session_id"] = session_id
        super().__init__(
            message,
            code=code,
            retryable=retryable,
            cause=cause,
            details=merged,
        )
        self.session_id: str | None = session_id


class BrowserSessionGoneError(BrowserSessionError):
    """Raised when a browser session/page closed, crashed, or expired upstream.

    Callers should recreate the session cleanly and report `restarted=True`
    rather than retrying on the dead handle.
    """

    default_code: ErrorCode = ErrorCode.BROWSER_SESSION_GONE
    default_retryable: bool = False

    def __init__(
        self,
        session_id: str,
        message: str | None = None,
        *,
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = message or f"Browser session gone: {session_id}"
        super().__init__(
            msg,
            session_id=session_id,
            code=ErrorCode.BROWSER_SESSION_GONE,
            retryable=False,
            cause=cause,
            details=details,
        )


class PerceptionError(AgentBrowseError):
    """Raised when DOM or accessibility-tree perception fails."""

    default_code: ErrorCode = ErrorCode.PERCEPTION_FAILED
    default_retryable: bool = True

    def __init__(
        self,
        message: str,
        *,
        tier: int | None = None,
        url: str | None = None,
        retryable: bool = True,
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(details) if details else {}
        if tier is not None:
            merged["tier"] = tier
        if url is not None:
            merged["url"] = url
        super().__init__(
            message,
            code=ErrorCode.PERCEPTION_FAILED,
            retryable=retryable,
            cause=cause,
            details=merged,
        )
        self.tier: int | None = tier
        self.url: str | None = url


class ActionExecutionError(AgentBrowseError):
    """Raised when a browser action (click, type, select, navigate) fails."""

    default_code: ErrorCode = ErrorCode.ACTION_FAILED
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        action: str | None = None,
        element_id: str | None = None,
        failure_class: str | None = None,
        stale_ref: bool = False,
        retryable: bool | None = None,
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(details) if details else {}
        if action is not None:
            merged["action"] = action
        if element_id is not None:
            merged["element_id"] = element_id
        if failure_class is not None:
            merged["failure_class"] = failure_class
        if stale_ref:
            merged["stale_ref"] = True
        resolved_code = ErrorCode.STALE_REF if stale_ref else ErrorCode.ACTION_FAILED
        resolved_retryable = stale_ref if retryable is None else retryable
        super().__init__(
            message,
            code=resolved_code,
            retryable=resolved_retryable,
            cause=cause,
            details=merged,
        )
        self.action: str | None = action
        self.element_id: str | None = element_id
        self.failure_class: str | None = failure_class
        self.stale_ref: bool = stale_ref


class SafetyPolicyError(NonRetryableError):
    """Raised when execution safety blocks a URL, domain, or file upload."""

    default_code: ErrorCode = ErrorCode.SAFETY_BLOCKED

    def __init__(
        self,
        message: str,
        *,
        policy: str,
        violation: str,
        target: str | None = None,
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(details) if details else {}
        merged["policy"] = policy
        merged["violation"] = violation
        if target is not None:
            merged["target"] = target
        super().__init__(
            message,
            code=ErrorCode.SAFETY_BLOCKED,
            cause=cause,
            details=merged,
        )
        self.policy: str = policy
        self.violation: str = violation
        self.target: str | None = target


@dataclass(frozen=True)
class ModelAttempt:
    """Single leg recorded inside a `ModelProviderError` failover chain."""

    model: str
    error: str
    status: int | None = None
    code: str | None = None


class ModelProviderError(AgentBrowseError):
    """Structured LLM provider/failover error with HTTP status & attempt chain."""

    default_code: ErrorCode = ErrorCode.MODEL_UNKNOWN
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        model: str,
        status: int | None = None,
        code: ErrorCode | None = None,
        retryable: bool | None = None,
        provider: str | None = None,
        attempts: tuple[ModelAttempt, ...] | list[ModelAttempt] = (),
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        inferred_code, inferred_retryable = classify_http_status(status)
        resolved_code = code if code is not None else inferred_code
        if retryable is None:
            resolved_retryable = inferred_retryable or resolved_code in (
                ErrorCode.MODEL_RATE_LIMIT,
                ErrorCode.MODEL_TIMEOUT,
                ErrorCode.MODEL_PROVIDER_ERROR,
            )
        else:
            resolved_retryable = bool(retryable)

        attempt_tuple = tuple(attempts)
        merged = dict(details) if details else {}
        merged["model"] = model
        if provider is not None:
            merged["provider"] = provider
        if status is not None:
            merged["status"] = status
        if attempt_tuple:
            merged["attempts"] = [asdict(a) for a in attempt_tuple]

        super().__init__(
            message,
            code=resolved_code,
            retryable=resolved_retryable,
            cause=cause,
            details=merged,
        )
        self.model: str = model
        self.provider: str | None = provider
        self.status: int | None = status
        self.attempts: tuple[ModelAttempt, ...] = attempt_tuple


def _iter_cause_chain(exc: BaseException, max_depth: int = _MAX_CAUSE_WALK_DEPTH):
    """Yield `exc` and its `__cause__`/`__context__` chain with cycle & depth guards."""
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth <= max_depth:
        obj_id = id(current)
        if obj_id in seen:
            break
        seen.add(obj_id)
        yield current
        current = current.__cause__ or current.__context__
        depth += 1


def is_non_retryable(exc: BaseException) -> bool:
    """Return True if `exc` (or an explicit cause) must never be retried."""
    for item in _iter_cause_chain(exc):
        if isinstance(item, NonRetryableError):
            return True
        if type(item).__name__ == "NonRetryableError":
            return True
        if isinstance(item, _NON_RETRYABLE_BUILTINS):
            return True
    return False


def is_error_retryable(exc: BaseException) -> bool:
    """Inspect typed `AgentBrowseError` flags across the exception cause chain.

    Returns False immediately if any error in the chain is explicitly non-retryable.
    Returns True if an `AgentBrowseError` marks `retryable=True`.
    """
    if is_non_retryable(exc):
        return False
    for item in _iter_cause_chain(exc):
        if isinstance(item, AgentBrowseError):
            return item.retryable
    return False


__all__ = [
    "ActionExecutionError",
    "AgentBrowseError",
    "BrowserSessionError",
    "BrowserSessionGoneError",
    "ConfigValidationError",
    "ErrorCode",
    "ModelAttempt",
    "ModelProviderError",
    "NonRetryableError",
    "PerceptionError",
    "SafetyPolicyError",
    "classify_http_status",
    "is_error_retryable",
    "is_non_retryable",
    "to_error_message",
]
