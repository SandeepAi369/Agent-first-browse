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

    verb: Literal[
        "goto", "click", "type", "scroll", "press_enter", "wait", "done",
        "hover", "select_option", "press_key",
    ] = "wait"
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

    # ── Cognition Core (V18) — the agent's working theory of the task ──
    # NOTE: `beliefs` deliberately uses overwrite (not the `add` reducer) so the
    # updating node can keep it capped/de-duplicated — bounded memory prevents
    # prompt overload / hallucination. See cognition.merge_beliefs().
    strategy: str = ""                  # overall approach the agent is testing
    success_criteria: str = ""          # what "done" looks like (anti done-confusion)
    strategy_confidence: float = 1.0    # decays on no-progress, reinforces on progress
    beliefs: list[str] = Field(default_factory=list)   # bounded learned facts
    current_obstacle: str = ""          # "<url>|<plan_step>" identity for the ladder
    ladder_rung: int = 0                # escalation position for current obstacle
    tried_tactics: list[str] = Field(default_factory=list)  # tactics used on it
    goal_score_window: list[float] = Field(default_factory=list)  # PRM goal-scores
    restrategize_count: int = 0         # bounded by cognition.MAX_RESTRATEGIZE
    goal_complete_hint: str = ""        # set when PRM says the goal looks achieved

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

    # ── V21 Vision-on-demand (a11y DOM is default; vision is a per-step toggle) ──
    vision_consults: int = 0       # per-task budget counter
    force_vision: bool = False     # set by the ladder's vision rung; consumed next step
    ineffective_streak: int = 0    # consecutive ineffective click/type on same target

    # ── V29 Reality Monitor (Phase 1) — screen-reality reconciliation ──
    # Written ONLY by Overwatch after executing an action. `reality_status` is
    # observational; `reality_note` carries a CONTRADICTED discrepancy back to the
    # worker via the guidance bus so the agent re-evaluates the REAL screen instead
    # of blindly proceeding. Cleared on a successful commit (clear_transient).
    reality_status: str = ""   # "" | CONFIRMED | CONTRADICTED | UNCLEAR | NULL
    reality_note: str = ""     # discrepancy fed to the guidance bus on contradiction

    # ── V29 Phase A — unified DOM-diff state-change signal (written by Overwatch
    # from CriticV12). 0..1: how much the page state changed after the last action.
    state_change_score: float = 0.0

    # ── V29 Phase B — WebDreamer (predictive simulation) diagnostics ──
    webdreamer_runs: int = 0       # times the dreamer simulated before acting
    webdreamer_overrides: int = 0  # times its imagined best replaced the worker pick

    # ── V29 Phase 2/3 — Clarity Gate, Target Lock, Stagnation, Smart Scroll ──
    # bound_target: the semantic identity of the current sub-task's target (written
    # by perceive_node). stagnation_*: written by perceive_node. scroll_stuck_streak:
    # written by overwatch when a scroll yields no new content / no movement.
    bound_target: str = ""             # human-readable focus (target_lock phrase)
    stagnation_level: int = 0          # 0..3 progress-signals fired
    stagnation_note: str = ""          # guidance-bus directive when stuck
    scroll_stuck_streak: int = 0       # consecutive unproductive scrolls

    # ── V29 Atomic Intent Journal — write-ahead record of a side-effecting action.
    # Written by Overwatch BEFORE execution; cleared on a confirmed success. While
    # set (action timed out / unconfirmed), the next worker decision is warned not
    # to blindly repeat it — fixes handoff-amnesia double-toggling.
    last_attempted_action: dict | None = None

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
    # P2: consensus diagnostics — how many IRREVERSIBLE actions were put to a
    # multi-model vote, and how many were abstained (held back as too ambiguous).
    consensus_votes: int = 0
    abstentions: int = 0
    # V20: the outcome judge's cited on-page proof (or the honest gap on an
    # unverified shutdown) — surfaced in the finalize report.
    done_evidence: str = ""

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
        """The agent's SINGLE source of cognitive state (injected into the system
        prompt). This is the ONE authoritative answer to: what is the master goal,
        what is the one sub-task to do right now, what's already done, and what to
        ignore. There is intentionally NO second/competing task list — a prior
        design that injected a separate checklist caused the agent to lose track
        of its goal and tunnel onto abstract sub-goals.
        """
        done = [s for s in self.plan_steps if s.get("status") == "done"]
        active = self.get_current_plan_step()
        pending = [s for s in self.plan_steps if s.get("status") == "pending"]

        lines = [f"🎯 MASTER GOAL: {self.objective[:240]}"]
        lines.append(f"📊 PROGRESS: {self.plan_progress_pct}% ({len(done)}/{len(self.plan_steps)} steps done)")

        if done:
            done_str = "; ".join(s["desc"][:50] for s in done[-3:])
            if len(done) > 3:
                done_str = f"...({len(done)-3} earlier); " + done_str
            lines.append(f"✓ DONE (do NOT repeat): {done_str}")

        if active:
            lines.append(
                f"▶ CURRENT SUB-TASK (your ONLY focus right now — accomplish THIS "
                f"using what is actually on the screen): {active['desc']}")
            nxt = pending[0]["desc"][:50] if pending else None
            if nxt:
                lines.append(f"   (do NOT start the next step until current is done): {nxt}")
        elif not pending:
            lines.append("▶ CURRENT SUB-TASK: all steps complete — verify the goal on screen and finish.")

        # Distraction guard — the #1 real-world failure on dense e-commerce pages.
        lines.append(
            "⚠️ STAY ON TASK: act only on the element(s) that advance your CURRENT "
            "SUB-TASK. Ignore distractions — ads, 'sponsored', 'recommended', "
            "'related products', 'people also bought', newsletter/cookie popups, "
            "and any widget unrelated to the current sub-task.")

        if self.reflections:
            lines.append(f"💡 NOTE: {self.reflections[-1][:150]}")

        return "\n".join(lines)

    def compress_history(self) -> str:
        """SWE-agent style history compression.
        
        V17: Deliberately excludes 'expected_change' from history entries.
        The EXPECT field caused cognitive confusion — when reality differed
        from the prediction, the next step's OBSERVE tried to reconcile them,
        creating loops. The agent should react to what IS, not what it predicted.
        """
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
