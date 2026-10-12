"""Orchestrator State — Typed State + Dual Ledgers.

Inspired by Magentic-One's Task Ledger + Progress Ledger pattern.
The state is the single mutable object that flows through the system.
All components read and write to it; the Event Log captures
the immutable history of those mutations.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from orchestrator.actions import TaskDAG, NodeStatus


# ═══════════════════════════════════════════════════════════════════════════════
#  Task Ledger — Facts, hypotheses, and the current plan
# ═══════════════════════════════════════════════════════════════════════════════

class TaskLedger(BaseModel):
    """Stores the CEO's understanding of the task.

    Updated when: CEO plans or replans.
    Read by: Executor (to understand context), Critic (to evaluate).
    """
    objective: str = ""
    task_type: str = ""                           # browsing, scraping, posting, etc.
    facts: list[str] = Field(default_factory=list)  # Known truths about the environment
    assumptions: list[str] = Field(default_factory=list)  # Educated guesses
    constraints: list[str] = Field(default_factory=list)  # Limitations (rate limits, auth, etc.)


# ═══════════════════════════════════════════════════════════════════════════════
#  Progress Ledger — Self-reflection after each step
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressEntry(BaseModel):
    """A single reflection entry in the Progress Ledger."""
    node_id: str
    timestamp: float = Field(default_factory=time.time)
    action_taken: str
    observation: str
    success: bool
    reasoning: str = ""


class ProgressLedger(BaseModel):
    """Tracks execution progress and self-reflection.

    Updated when: Executor completes a node, Critic evaluates.
    Read by: CEO (for replanning context), Executor (for loop detection).
    """
    entries: list[ProgressEntry] = Field(default_factory=list)
    total_actions: int = 0
    total_successes: int = 0
    total_failures: int = 0
    replan_count: int = 0

    def record(self, entry: ProgressEntry) -> None:
        """Record a progress entry."""
        self.entries.append(entry)
        self.total_actions += 1
        if entry.success:
            self.total_successes += 1
        else:
            self.total_failures += 1

    def recent(self, n: int = 5) -> list[ProgressEntry]:
        """Get the N most recent entries."""
        return self.entries[-n:]

    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_actions == 0:
            return 1.0
        return self.total_successes / self.total_actions

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"Progress: {self.total_successes}/{self.total_actions} succeeded "
            f"({self.success_rate():.0%}), "
            f"{self.replan_count} replans"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestrator State — The mutable state flowing through the system
# ═══════════════════════════════════════════════════════════════════════════════

class OrchestratorState(BaseModel):
    """The central state object for the entire orchestration pipeline.

    This is the ONLY mutable object in the system. Everything else
    (events, plans, observations) is derived from or recorded alongside it.
    """
    # ── Core ──
    objective: str = ""
    dag: TaskDAG | None = None
    current_node_id: str | None = None
    current_url: str = ""

    # ── Ledgers (from Magentic-One) ──
    task_ledger: TaskLedger = Field(default_factory=TaskLedger)
    progress_ledger: ProgressLedger = Field(default_factory=ProgressLedger)

    # ── Execution Metadata ──
    started_at: float = Field(default_factory=time.time)
    max_replan_attempts: int = 3
    is_paused: bool = False
    pause_reason: str = ""
    is_complete: bool = False
    final_result: Any = None

    # ── DOM Context (refreshed each step) ──
    dom_tree: str = ""
    dom_elements: list[dict] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def elapsed_seconds(self) -> float:
        """Time since orchestration started."""
        return time.time() - self.started_at

    def request_human_intervention(self, reason: str) -> None:
        """Pause execution and request human help."""
        self.is_paused = True
        self.pause_reason = reason

    def mark_complete(self, result: Any = None) -> None:
        """Mark the entire orchestration as complete."""
        self.is_complete = True
        self.final_result = result

    def get_context_for_llm(self) -> dict[str, Any]:
        """Build a compact context dict for LLM prompts."""
        dag_summary = self.dag.summary() if self.dag else "No plan yet"
        progress = self.progress_ledger.summary()
        recent = self.progress_ledger.recent(3)

        return {
            "objective": self.objective,
            "current_url": self.current_url,
            "plan": dag_summary,
            "progress": progress,
            "recent_actions": [
                {"node": e.node_id, "action": e.action_taken,
                 "result": e.observation[:200], "ok": e.success}
                for e in recent
            ],
            "dom_tree": self.dom_tree[:3000] if self.dom_tree else "(no DOM loaded)",
            "elapsed_sec": round(self.elapsed_seconds(), 1),
        }
