"""BrainState — Typed Global State for the True Brain v16.0 Architecture.

Consolidates all 15+ scattered variables from advanced_agent.py into a
single Pydantic model that flows through the LangGraph StateGraph.

Design principles:
  - Every node reads and writes ONE typed state object
  - Annotated[list, add] fields use LangGraph reducer semantics (append, not overwrite)
  - State is kept lean (<10KB) for fast checkpointing (<15ms with SqliteSaver)
  - Page FSM makes browser lifecycle states explicit

References:
  - MAST (arXiv 2503.13657): typed state defeats "context loss" (44.2% of failures)
  - Six Sigma Agent (arXiv 2601.22290): checkpoint-every-k-steps strategy
  - LangGraph docs: reducer semantics for accumulator fields
"""

from __future__ import annotations

from typing import Annotated, Literal, Any
from operator import add

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
#  Action Types — Proposed and Recorded
# ═══════════════════════════════════════════════════════════════════════════════

class ProposedAction(BaseModel):
    """An action proposed by a worker node. Never committed until Overwatch validates."""

    verb: Literal["goto", "click", "type", "scroll", "press_enter", "wait", "done"] = "wait"
    element_id: str | None = None
    target_name: str = ""
    text: str | None = None
    url: str | None = None
    x: float | None = None
    y: float | None = None
    rationale: str = ""
    reversible: bool = True
    risk_level: str = "REVERSIBLE"  # REVERSIBLE | CAUTIOUS | IRREVERSIBLE

    # OTA chain-of-thought fields (from AutonomousAction)
    screen_state: str = ""
    previous_action_result: str = ""
    goal_progress: str = ""
    reasoning: str = ""

    def describe(self) -> str:
        """Human-readable description for logging."""
        parts = [self.verb]
        if self.element_id:
            parts.append(f"[{self.element_id}]")
        if self.target_name:
            parts.append(f"'{self.target_name[:30]}'")
        if self.text:
            parts.append(f"text='{self.text[:40]}'")
        if self.url:
            parts.append(f"url='{self.url[:50]}'")
        return " ".join(parts)


class StepRecord(BaseModel):
    """Immutable record of a completed step (appended to history via reducer)."""

    step_id: int = 0
    action_verb: str = ""
    action_element_id: str | None = None
    action_text: str | None = None
    action_target_name: str = ""
    pre_state_hash: int = 0
    post_state_hash: int = 0
    verifier_verdict: Literal["pass", "fail", "retry", "rollback"] = "pass"
    outcome: str = ""
    screen_hint: str = ""
    url: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  BrainState — The single source of truth
# ═══════════════════════════════════════════════════════════════════════════════

class BrainState(BaseModel):
    """Typed global state for the LangGraph orchestration spine.

    Every node in the graph reads from and writes to this object.
    LangGraph's checkpointer serializes it at each super-step.

    Fields marked with `Annotated[list, add]` use LangGraph reducer
    semantics: updates APPEND to the list rather than overwriting it.
    """

    # ── Identity & Goal ──
    objective: str = ""
    task_domain: str = ""  # auto-extracted domain (e.g., "amazon.in")

    # ── Plan (from PlanState) ──
    plan_steps: list[dict] = Field(default_factory=list)
    plan_cursor: int = 0
    plan_progress_pct: int = 0
    reflections: Annotated[list[str], add] = Field(default_factory=list)

    # ── Working Memory ──
    history: Annotated[list[dict], add] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)
    failures: list[dict] = Field(default_factory=list)
    extracted_data: Annotated[list[dict], add] = Field(default_factory=list)

    # ── Page State / FSM ──
    page_fsm: Literal[
        "LOADING", "READY", "ACTION_PENDING", "VALIDATED", "ERROR"
    ] = "READY"
    current_url: str = ""
    dom_markdown: str = ""
    selector_map: dict[str, Any] = Field(default_factory=dict)
    elements_list: list[dict] = Field(default_factory=list)
    element_count: int = 0
    login_detected: bool = False

    # ── Control Flow ──
    step_number: int = 0
    max_steps: int = 25
    error_count: int = 0
    correction_failures: int = 0
    consecutive_identical_actions: int = 0
    last_action_signature: str = ""
    same_url_streak: int = 0
    last_url_for_streak: str = ""

    # ── Loop Detection ──
    loop_signatures: Annotated[list[str], add] = Field(default_factory=list)

    # ── Routing & Action ──
    next_node: str = ""
    proposed_action: dict | None = None  # Serialized ProposedAction
    overwatch_verdict: str = ""  # "pass" | "retry" | "rollback" | "escalate"
    action_outcome: str = ""  # "→ OK" | "→ FAILED: ..." | etc.
    retry_count: int = 0

    # ── Session Results ──
    mission_success: bool = False
    session_checkpoint_id: str | None = None
    skill_context: str = ""  # Injected SkillMemory context

    # ── PRM Checklist ──
    prm_checklist: list[dict] = Field(default_factory=list)

    # ── Recovery ──
    recovery_advice: str = ""
    correction_context: str = ""

    # ── Metrics ──
    total_actions: int = 0
    grounding_rejects: int = 0
    critic_progress: int = 0
    critic_no_progress: int = 0
    reflexion_triggers: int = 0
    done_blocked: int = 0

    # ── Budget extension tracking ──
    budget_extended: bool = False

    # ── Pre-computed renders (updated by perceive_node each cycle) ──
    plan_render: str = ""
    facts_render: str = ""
    history_compressed: str = ""

    def get_current_plan_step(self) -> dict | None:
        """Return the current active plan step."""
        for s in self.plan_steps:
            if s.get("status") in ("active", "in_progress"):
                return s
        return None

    def get_plan_render(self) -> str:
        """Render plan for injection into system prompt (compact)."""
        done = [s for s in self.plan_steps if s.get("status") == "done"]
        active = self.get_current_plan_step()
        pending = [s for s in self.plan_steps if s.get("status") == "pending"]

        lines = [f"MISSION: {self.objective[:200]}"]
        lines.append(f"PROGRESS: {self.plan_progress_pct}% ({len(done)}/{len(self.plan_steps)} steps)")

        if done:
            done_str = "; ".join(s["desc"][:50] for s in done[-3:])
            if len(done) > 3:
                done_str = f"...({len(done)-3} earlier); " + done_str
            lines.append(f"DONE: {done_str}")

        if active:
            lines.append(f"CURRENT: ▶ {active['desc']}")
        elif not pending:
            lines.append("CURRENT: All steps completed — verify and finish")

        if pending:
            remaining = "; ".join(s["desc"][:40] for s in pending[:3])
            if len(pending) > 3:
                remaining += f"; ...+{len(pending)-3} more"
            lines.append(f"REMAINING: {remaining}")

        if self.reflections:
            lines.append(f"REFLECTION: {self.reflections[-1][:150]}")

        return "\n".join(lines)

    def compress_history(self) -> str:
        """SWE-agent style history compression."""
        if not self.history:
            return "[]"
        entries = list(self.history)
        result_lines = []
        if len(entries) > 5:
            for e in entries[:-5]:
                result_lines.append(
                    f"Step {e.get('step', '?')}: {e.get('action', '?')} {e.get('outcome', '')}"
                )
        for e in entries[-5:]:
            line = f"Step {e.get('step', '?')}: {e.get('action', '?')} {e.get('outcome', '')}"
            if e.get("screen"):
                line += f" | Screen: {e['screen']}"
            result_lines.append(line)
        return "\n".join(result_lines)

    def render_facts(self) -> str:
        """Render facts for system prompt."""
        if not self.facts:
            return ""
        return "\n".join(f"• {k}: {v}" for k, v in list(self.facts.items())[-10:])
