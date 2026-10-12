"""Append-Only Event Log — Single Source of Truth.

Inspired by OpenHands' immutable event stream architecture.
Every action, observation, plan, and error is captured as an
immutable Event and appended to the log. The full execution
history can be replayed, serialized for LLM context, or
persisted to disk as JSON.
"""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
#  Event Types
# ═══════════════════════════════════════════════════════════════════════════════

class EventType(str, Enum):
    """Classification of events in the stream."""
    PLAN       = "plan"         # CEO produced or revised a plan
    ACTION     = "action"       # An action was dispatched to the executor
    OBSERVE    = "observe"      # The result of an action (DOM state, output, etc.)
    CRITIQUE   = "critique"     # The Critic's evaluation of an action's result
    REPLAN     = "replan"       # CEO revised the plan after a failure
    ERROR      = "error"        # An unrecoverable error occurred
    HUMAN      = "human"        # Human intervention was requested or received
    SYSTEM     = "system"       # Internal system events (startup, shutdown)


# ═══════════════════════════════════════════════════════════════════════════════
#  Event — The fundamental unit of the log
# ═══════════════════════════════════════════════════════════════════════════════

class Event(BaseModel):
    """An immutable event in the execution stream.

    Once appended, events are NEVER modified. This guarantees
    deterministic replay and reliable LLM context construction.
    """
    timestamp: float = Field(default_factory=time.time)
    event_type: EventType
    node_id: str | None = Field(default=None, description="Which DAG node this event relates to")
    summary: str = Field(..., description="Human-readable one-line summary")
    data: dict[str, Any] = Field(default_factory=dict, description="Structured payload")

    class Config:
        use_enum_values = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Event Log — The append-only stream
# ═══════════════════════════════════════════════════════════════════════════════

class EventLog:
    """Append-only event stream. The single source of truth for execution state.

    Design principles (from OpenHands):
      - Immutable: Events are never modified after append.
      - Serializable: The entire log can be dumped to JSON for persistence.
      - Context-Ready: Can produce a compressed summary for LLM context windows.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []

    def append(self, event: Event) -> None:
        """Append an event to the log. Once appended, it is immutable."""
        self._events.append(event)

    def action(self, node_id: str, summary: str, **data: Any) -> None:
        """Shortcut: append an ACTION event."""
        self.append(Event(
            event_type=EventType.ACTION,
            node_id=node_id,
            summary=summary,
            data=data,
        ))

    def observe(self, node_id: str, summary: str, **data: Any) -> None:
        """Shortcut: append an OBSERVE event."""
        self.append(Event(
            event_type=EventType.OBSERVE,
            node_id=node_id,
            summary=summary,
            data=data,
        ))

    def critique(self, node_id: str, summary: str, **data: Any) -> None:
        """Shortcut: append a CRITIQUE event."""
        self.append(Event(
            event_type=EventType.CRITIQUE,
            node_id=node_id,
            summary=summary,
            data=data,
        ))

    def error(self, summary: str, node_id: str | None = None, **data: Any) -> None:
        """Shortcut: append an ERROR event."""
        self.append(Event(
            event_type=EventType.ERROR,
            node_id=node_id,
            summary=summary,
            data=data,
        ))

    def system(self, summary: str, **data: Any) -> None:
        """Shortcut: append a SYSTEM event."""
        self.append(Event(
            event_type=EventType.SYSTEM,
            summary=summary,
            data=data,
        ))

    # ── Query ─────────────────────────────────────────────────────────────

    @property
    def events(self) -> list[Event]:
        """Read-only access to the full event list."""
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def last(self, n: int = 1) -> list[Event]:
        """Return the last N events."""
        return self._events[-n:]

    def for_node(self, node_id: str) -> list[Event]:
        """Return all events related to a specific DAG node."""
        return [e for e in self._events if e.node_id == node_id]

    def of_type(self, event_type: EventType) -> list[Event]:
        """Return all events of a specific type."""
        return [e for e in self._events if e.event_type == event_type]

    # ── LLM Context Generation ────────────────────────────────────────────

    def to_context(self, last_n: int = 15, max_chars: int = 4000) -> str:
        """Produce a compressed summary of recent events for LLM context.

        This is the critical interface between the event log and the LLM.
        It must be concise (to fit context windows) but informative
        (so the LLM can reason about what happened).
        """
        recent = self._events[-last_n:]
        lines: list[str] = []
        total_chars = 0

        for evt in recent:
            icon = {
                "plan": "📋", "action": "⚡", "observe": "👁️",
                "critique": "🔍", "replan": "🔄", "error": "❌",
                "human": "👤", "system": "⚙️",
            }.get(evt.event_type, "•")

            node_tag = f"[{evt.node_id}] " if evt.node_id else ""
            line = f"{icon} {node_tag}{evt.summary}"

            # Truncate if we're approaching the char limit
            if total_chars + len(line) > max_chars:
                lines.append("... (earlier events truncated)")
                break

            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    # ── Persistence (JSON) ────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist the full event log to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.model_dump() for e in self._events]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EventLog":
        """Load an event log from a JSON file."""
        log = cls()
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            for item in raw:
                log.append(Event(**item))
        return log

    def to_dict(self) -> list[dict]:
        """Serialize the entire log to a list of dicts."""
        return [e.model_dump() for e in self._events]
