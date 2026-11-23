"""Decoupled, typed observability event bus and pluggable telemetry sinks.

Built for Python 3.11+ while remaining completely decoupled from legacy
`app/browser_promoter/state.py`:

  - Independent `OBSERVABILITY_SCHEMA_VERSION = "1"` (never tied to package version).
  - Silent-by-default `ObservabilityBus` backed by `core.disposable.Emitter` so
    emitting with zero subscribers costs virtually nothing and failing sinks
    never break browser execution or sibling sinks.
  - Automatic ambient enrichment (`session_id`, `scope`, `mode`) from
    `core.runtime_context.get_current_context()`.
  - Built-in sinks:
      * `MemoryEventSink` (bounded in-memory ring for tests & diagnostics)
      * `JsonlFileSink` (redacted, length-bounded NDJSON event log)
      * `RunMetricsSink` (bridges events cleanly to `v2_engine.metrics.RunMetrics`)
      * `LangSmithSink` + `build_run_config()` (optional general-purpose tracing
        bridge with zero promoter/campaign coupling)
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.disposable import Disposable, Emitter, ErrorHandler
from core.errors import ConfigValidationError
from core.redaction import redact_payload
from core.runtime_context import get_current_context
from core.truncate import truncate_result

logger = logging.getLogger("core.observability")

OBSERVABILITY_SCHEMA_VERSION: str = "1"


class EventType(StrEnum):
    """Canonical observability event names across perception, actions, MCP, and sessions."""

    PERCEPTION_COMPLETED = "perception.completed"
    ACTION_EXECUTED = "action.executed"
    ACTION_RETRIED = "action.retried"
    STALE_REF_RECOVERED = "action.stale_ref_recovered"
    STATE_SAFETY_OVERRIDE = "action.state_safety_override"
    SAFETY_BLOCKED = "safety.blocked"
    SESSION_STARTED = "session.started"
    SESSION_RESTARTED = "session.restarted"
    SESSION_CLOSED = "session.closed"
    MCP_TOOL_CALLED = "mcp.tool_called"
    MODEL_FAILOVER = "model.failover"
    CUSTOM = "custom"


def _normalize_event_type(raw: EventType | str) -> str:
    if isinstance(raw, EventType):
        return raw.value
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigValidationError(f"event type must be a non-empty string, got {raw!r}")
    return raw.strip()


def _epoch_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class ObservabilityEvent:
    """Typed, schema-versioned telemetry event envelope."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    scope: str | None = None
    mode: str | None = None
    timestamp_ms: int = field(default_factory=_epoch_ms)
    schema_version: str = OBSERVABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized = _normalize_event_type(self.type)
        object.__setattr__(self, "type", normalized)
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(
        self,
        *,
        redact: bool = True,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        """Serialize to a clean JSON-ready dict with optional redaction & truncation."""
        body: Any = redact_payload(self.payload) if redact else dict(self.payload)
        if max_chars is not None:
            body = truncate_result(body, max_chars=max_chars)
        return {
            "schema_version": self.schema_version,
            "type": self.type,
            "timestamp_ms": self.timestamp_ms,
            "session_id": self.session_id,
            "scope": self.scope,
            "mode": self.mode,
            "payload": body,
        }


@runtime_checkable
class EventSink(Protocol):
    """Pluggable consumer for `ObservabilityEvent` records."""

    def handle(self, event: ObservabilityEvent) -> None: ...


EventHandler = EventSink | Callable[[ObservabilityEvent], None]


class ObservabilityBus:
    """Silent-by-default, error-isolated pub/sub bus for structured runtime telemetry."""

    def __init__(self, *, on_error: ErrorHandler | None = None) -> None:
        self._emitter: Emitter[ObservabilityEvent] = Emitter(on_error=on_error)

    @property
    def subscriber_count(self) -> int:
        return self._emitter.listener_count

    @property
    def is_disposed(self) -> bool:
        return self._emitter.is_disposed

    def subscribe(
        self,
        sink: EventHandler,
        *,
        event_types: Iterable[EventType | str] | None = None,
    ) -> Disposable:
        """Register a sink or callback with optional event-type filtering."""
        handler = sink.handle if isinstance(sink, EventSink) else sink
        allowed = (
            frozenset(_normalize_event_type(t) for t in event_types)
            if event_types is not None
            else None
        )

        if allowed is None:
            return self._emitter.subscribe(handler)

        def _filtered(event: ObservabilityEvent) -> None:
            if event.type in allowed:
                handler(event)

        return self._emitter.subscribe(_filtered)

    def emit(
        self,
        event_or_type: ObservabilityEvent | EventType | str,
        payload: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        scope: str | None = None,
        mode: str | None = None,
    ) -> ObservabilityEvent:
        """Emit an event, enriching unset context metadata from `BrowserRuntimeContext`."""
        ctx = get_current_context()
        if isinstance(event_or_type, ObservabilityEvent):
            event = ObservabilityEvent(
                type=event_or_type.type,
                payload=event_or_type.payload,
                session_id=(
                    event_or_type.session_id
                    if event_or_type.session_id is not None
                    else (ctx.session_id if ctx is not None else None)
                ),
                scope=(
                    event_or_type.scope
                    if event_or_type.scope is not None
                    else (ctx.scope if ctx is not None else None)
                ),
                mode=(
                    event_or_type.mode
                    if event_or_type.mode is not None
                    else (ctx.engine_mode if ctx is not None else None)
                ),
                timestamp_ms=event_or_type.timestamp_ms,
                schema_version=event_or_type.schema_version,
            )
        else:
            event = ObservabilityEvent(
                type=_normalize_event_type(event_or_type),
                payload=payload or {},
                session_id=(
                    session_id
                    if session_id is not None
                    else (ctx.session_id if ctx is not None else None)
                ),
                scope=scope if scope is not None else (ctx.scope if ctx is not None else None),
                mode=mode if mode is not None else (ctx.engine_mode if ctx is not None else None),
            )

        self._emitter.fire(event)
        return event

    def dispose(self) -> None:
        self._emitter.dispose()


DEFAULT_OBSERVABILITY_BUS = ObservabilityBus()


def get_default_bus() -> ObservabilityBus:
    """Return the process-wide silent-by-default `ObservabilityBus`."""
    return DEFAULT_OBSERVABILITY_BUS


# ── Built-In Sinks ─────────────────────────────────────────────────────────────


class MemoryEventSink:
    """In-memory bounded event buffer useful for tests and MCP/CLI diagnostics."""

    def __init__(self, *, max_events: int | None = None) -> None:
        if max_events is not None and (
            isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0
        ):
            raise ConfigValidationError(
                f"memory_sink.max_events must be an integer > 0, got {max_events!r}"
            )
        self._events: deque[ObservabilityEvent] = deque(maxlen=max_events)

    @property
    def events(self) -> list[ObservabilityEvent]:
        return list(self._events)

    def handle(self, event: ObservabilityEvent) -> None:
        self._events.append(event)

    def by_type(self, event_type: EventType | str) -> list[ObservabilityEvent]:
        target = _normalize_event_type(event_type)
        return [ev for ev in self._events if ev.type == target]

    def clear(self) -> None:
        self._events.clear()


class JsonlFileSink:
    """Writes redacted, optionally bounded JSONL telemetry records to disk."""

    def __init__(
        self,
        path: Path | str,
        *,
        redact: bool = True,
        max_payload_chars: int | None = 8000,
    ) -> None:
        self.path = Path(path)
        self.redact = redact
        self.max_payload_chars = max_payload_chars
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    @property
    def is_closed(self) -> bool:
        return self._closed

    def handle(self, event: ObservabilityEvent) -> None:
        if self._closed:
            return
        record = event.to_dict(redact=self.redact, max_chars=self.max_payload_chars)
        self._fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._fh.flush()

    def dispose(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            self._fh.close()


@runtime_checkable
class _RunMetricsLike(Protocol):
    def record_step(self, metric: Any) -> None: ...
    def record_action_retry(self) -> None: ...
    def record_stale_ref_recovery(self) -> None: ...
    def record_state_safety_override(self) -> None: ...


def _default_step_metric_builder(event: ObservabilityEvent) -> Any:
    """Build a `v2_engine.metrics.StepMetric` lazily without hard import coupling."""
    from v2_engine.metrics import StepMetric

    p = event.payload
    return StepMetric(
        step=int(p.get("step", 0)),
        url=str(p.get("url", "")),
        mode=str(p.get("mode") or event.mode or "legacy"),
        settle_ms=float(p.get("settle_ms", 0.0)),
        extract_ms=float(p.get("extract_ms", 0.0)),
        total_perceive_ms=float(p.get("total_perceive_ms", 0.0)),
        raw_elements=int(p.get("raw_elements", 0)),
        emitted_elements=int(p.get("emitted_elements", 0)),
        occluded_filtered=int(p.get("occluded_filtered", 0)),
        offscreen_dropped=int(p.get("offscreen_dropped", 0)),
        markdown_chars=int(p.get("markdown_chars", 0)),
        est_tokens=int(p.get("est_tokens", 0)),
        snapshot_kind=str(p.get("snapshot_kind", "full")),
        tier_used=int(p.get("tier_used", 1)),
        action_retries=int(p.get("action_retries", 0)),
        stale_ref_recoveries=int(p.get("stale_ref_recoveries", 0)),
    )


class RunMetricsSink:
    """Bridges observability events into a `RunMetrics`-compatible accumulator."""

    def __init__(
        self,
        metrics: _RunMetricsLike,
        *,
        step_builder: Callable[[ObservabilityEvent], Any] | None = None,
    ) -> None:
        self._metrics = metrics
        self._step_builder = step_builder or _default_step_metric_builder

    def handle(self, event: ObservabilityEvent) -> None:
        if event.type == EventType.PERCEPTION_COMPLETED.value:
            self._metrics.record_step(self._step_builder(event))
        elif event.type == EventType.ACTION_RETRIED.value:
            self._metrics.record_action_retry()
        elif event.type == EventType.STALE_REF_RECOVERED.value:
            self._metrics.record_stale_ref_recovery()
        elif event.type == EventType.STATE_SAFETY_OVERRIDE.value:
            self._metrics.record_state_safety_override()


def build_langsmith_run_payload(
    event: ObservabilityEvent,
    *,
    project: str = "agent-first-browse",
    tags: Iterable[str] = (),
    max_payload_chars: int | None = 8000,
) -> dict[str, Any]:
    """Build a sanitized, general-purpose LangSmith `create_run` payload from an event."""
    safe = event.to_dict(redact=True, max_chars=max_payload_chars)
    merged_tags = ["agent-first-browse", event.type]
    if event.mode:
        merged_tags.append(f"mode:{event.mode}")
    merged_tags.extend(str(t) for t in tags if str(t).strip())

    return {
        "name": event.type,
        "run_type": "tool" if event.type.startswith(("action.", "mcp.")) else "chain",
        "project_name": project,
        "inputs": {"payload": safe["payload"]},
        "extra": {
            "metadata": {
                "schema_version": safe["schema_version"],
                "session_id": safe["session_id"],
                "scope": safe["scope"],
                "mode": safe["mode"],
                "timestamp_ms": safe["timestamp_ms"],
            }
        },
        "tags": merged_tags,
    }


def build_run_config(
    run_name: str,
    *,
    session_id: str | None = None,
    scope: str | None = None,
    mode: str | None = None,
    thread_id: str | None = None,
    tags: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
    recursion_limit: int = 1000,
) -> dict[str, Any]:
    """Build a LangChain/LangGraph run config decoupled from legacy promoter state."""
    ctx = get_current_context()
    resolved_session = session_id if session_id is not None else (ctx.session_id if ctx else None)
    resolved_scope = scope if scope is not None else (ctx.scope if ctx else None)
    resolved_mode = mode if mode is not None else (ctx.engine_mode if ctx else None)

    merged_tags = ["agent-first-browse", run_name]
    if resolved_mode:
        merged_tags.append(f"mode:{resolved_mode}")
    merged_tags.extend(str(t) for t in tags if str(t).strip())

    merged_meta: dict[str, Any] = {
        "component": run_name,
        "schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "session_id": resolved_session,
        "scope": resolved_scope,
        "mode": resolved_mode,
    }
    if metadata:
        merged_meta.update(redact_payload(metadata))

    configurable: dict[str, Any] = {"recursion_limit": recursion_limit}
    if thread_id:
        configurable["thread_id"] = thread_id

    return {
        "run_name": run_name,
        "tags": merged_tags,
        "metadata": merged_meta,
        "configurable": configurable,
    }


class LangSmithSink:
    """Optional LangSmith tracing sink with redacted payloads and safe failure isolation."""

    def __init__(
        self,
        *,
        client: Any = None,
        project: str = "agent-first-browse",
        tags: Iterable[str] = (),
        enabled: bool = True,
        max_payload_chars: int | None = 8000,
    ) -> None:
        self._client = client
        self.project = project
        self.tags = tuple(tags)
        self.enabled = enabled
        self.max_payload_chars = max_payload_chars
        self.dropped_count: int = 0

    def handle(self, event: ObservabilityEvent) -> None:
        if not self.enabled or self._client is None:
            return
        payload = build_langsmith_run_payload(
            event,
            project=self.project,
            tags=self.tags,
            max_payload_chars=self.max_payload_chars,
        )
        create_run = getattr(self._client, "create_run", None)
        if not callable(create_run):
            self.dropped_count += 1
            return
        create_run(**payload)


__all__ = [
    "DEFAULT_OBSERVABILITY_BUS",
    "OBSERVABILITY_SCHEMA_VERSION",
    "EventHandler",
    "EventSink",
    "EventType",
    "JsonlFileSink",
    "LangSmithSink",
    "MemoryEventSink",
    "ObservabilityBus",
    "ObservabilityEvent",
    "RunMetricsSink",
    "build_langsmith_run_payload",
    "build_run_config",
    "get_default_bus",
]
