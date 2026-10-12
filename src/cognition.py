"""Cognition Core — strategy, belief, escalation, and stall detection (V18).

This is the agent's "how it thinks" layer. It turns the brain from a stateless
step-reactor into a goal-directed reasoner that:

  1. forms an explicit STRATEGY (approach + success criteria) up front,
  2. reasons WITH that strategy every step (injected into worker prompts),
  3. updates CONFIDENCE in the strategy from evidence (reinforce / decay),
  4. escalates through a deterministic LADDER of DISTINCT tactics when stuck
     (provably never repeating a tactic for the same obstacle), and
  5. detects goal-progress STALL ("busy but not progressing") via the revived
     PRM goal-score, and re-strategizes.

Design constraints honored (from user feedback):
  - LEAN memory: beliefs are capped, de-duplicated, and truncated so the prompt
    never floods the LLM (overload → hallucination).
  - CLEAN handoff: this module is pure data/logic — all task state lives in
    BrainState (per-task), and `clear_transient()` / `clear_all()` give callers
    explicit reset points so a finished task never bleeds into the next.

Pure logic — no browser, no LLM-client construction. The single LLM call
(`restrategize`) is delegated through the injected `invoke_fn`, matching the
rest of the codebase.

References:
  - Reflexion (arXiv 2303.11366): verbal self-feedback as persistent memory.
  - Web-Shepherd PRM (arXiv 2505.15277): goal-aware step reward (revived here).
  - Tree-of-Thought / strategy-as-prior: conditioning the policy on a stable
    plan reduces decision entropy and prevents per-step oscillation.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("cognition")


# ═══════════════════════════════════════════════════════════════════════════════
#  Tunable constants (with the reasoning baked into the plan)
# ═══════════════════════════════════════════════════════════════════════════════

GAMMA = 0.3          # confidence EWMA rate: 3 consecutive no-progress → below TAU
TAU = 0.4            # confidence below this triggers a strategy review
STALL_EPS = 0.05     # goal-score spread below this over the window = stalled
STALL_WINDOW = 4     # number of recent goal-scores to inspect for a stall
PRM_AUDIT_EVERY = 2  # V17: tightened from 3 — catches sub-goal regression faster
MAX_RESTRATEGIZE = 3 # hard cap on full re-strategy calls per task
MAX_BELIEFS = 6      # keep injected memory LEAN — never flood the prompt
BELIEF_MAXLEN = 120  # truncate each belief so one bad line can't dominate
DONE_CEILING = 0.9   # PRM goal-score above which we nudge the agent to finish


# ═══════════════════════════════════════════════════════════════════════════════
#  Structured-output schemas (strict-safe — extra='forbid' per Plan 1)
# ═══════════════════════════════════════════════════════════════════════════════

class StrategicPlan(BaseModel):
    """Output of the strategic planner: approach + finish-line + steps."""
    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(
        description=(
            "ONE or two sentences describing your overall APPROACH to achieve the "
            "objective (e.g., 'Use the top-nav search to find the product, open the "
            "cheapest non-sponsored result, then add it to the cart')."
        )
    )
    success_criteria: str = Field(
        description=(
            "ONE sentence describing exactly what the page will show when the task "
            "is DONE (e.g., 'The cart counter shows 1 and a confirmation message "
            "appears'). This is the finish line."
        )
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 short, testable assumptions about the site/task you are starting "
            "with (e.g., 'Login is not required to search')."
        ),
    )
    steps: list[str] = Field(
        description=(
            "3-5 short sequential checkpoints, each ONE browser action "
            "(navigate, click, type, verify)."
        )
    )


class Restrategy(BaseModel):
    """Output of a re-strategize call after the agent gets stuck."""
    model_config = ConfigDict(extra="forbid")

    new_strategy: str = Field(
        description="A DIFFERENT overall approach that avoids the failed path. 1-2 sentences."
    )
    lesson: str = Field(
        description="ONE short sentence: what specifically was NOT working and why."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Escalation ladder — distinct tactics, ordered cheap → expensive
# ═══════════════════════════════════════════════════════════════════════════════
#  Each rung is (tactic_name, directive_text). The directive is injected into the
#  worker prompt via the existing `correction_context` slot. The final rung,
#  "restrategize", is special: it triggers an LLM re-strategy instead of a hint.

LADDER: list[tuple[str, str]] = [
    ("reperceive",
     "Re-read the page carefully — your previous snapshot may be stale. "
     "Look again at the element list before acting."),
    ("dismiss_overlay",
     "A popup, modal, cookie banner, or overlay is likely blocking you. "
     "Find and close/dismiss it FIRST, then continue."),
    ("scroll",
     "The target may be below the fold. Scroll down to reveal more elements, "
     "then choose your action."),
    ("wait",
     "The page may still be loading dynamic content. Use a short wait "
     "(1000-2000ms), then re-evaluate."),
    ("alternate_element",
     "The element you tried is not responding. Choose a DIFFERENT element that "
     "achieves the same sub-goal (a different button, link, or input)."),
    ("renavigate",
     "This route appears blocked. Back out or navigate to a different URL and "
     "approach the goal a different way."),
    ("vision",
     "The DOM is ambiguous here. Rely on what is visually present and pick the "
     "most likely target by its position and label."),
    ("restrategize", ""),  # special — handled by the rollback node (LLM call)
]

_LADDER_TACTICS = [t for t, _ in LADDER]
RESTRATEGIZE_TACTIC = "restrategize"


def obstacle_key(url: str, plan_step: str) -> str:
    """Identity of the current obstacle: same URL + same plan step = same obstacle.

    When this key changes, the situation genuinely changed, so the ladder resets.
    """
    return f"{(url or '').strip()}|{(plan_step or '').strip()[:60]}"


def advance_ladder(
    current_obstacle: str,
    new_obstacle: str,
    rung: int,
    tried: list[str],
) -> dict[str, Any]:
    """Pick the next escalation step for an obstacle.

    Guarantees (the math): for a FIXED obstacle the rung index advances
    monotonically through a finite list of DISTINCT tactics, so a tactic is
    never repeated for the same obstacle (no "scroll, fail, scroll again"
    loops) and the ladder terminates after at most len(LADDER) steps at
    `restrategize`. A changed obstacle key resets the ladder to rung 0.

    Returns a dict with: tactic, directive, rung (next), tried (updated),
    restrategize (bool), obstacle (the resolved key).
    """
    if new_obstacle != current_obstacle:
        rung = 0
        tried = []

    tried = list(tried)

    # Skip any rung whose tactic was already tried for this obstacle
    while rung < len(LADDER) and LADDER[rung][0] in tried:
        rung += 1

    if rung >= len(LADDER):
        return {
            "tactic": RESTRATEGIZE_TACTIC, "directive": "",
            "rung": len(LADDER), "tried": tried,
            "restrategize": True, "obstacle": new_obstacle,
        }

    tactic, directive = LADDER[rung]
    tried.append(tactic)
    return {
        "tactic": tactic,
        "directive": directive,
        "rung": rung + 1,
        "tried": tried,
        "restrategize": tactic == RESTRATEGIZE_TACTIC,
        "obstacle": new_obstacle,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Confidence — evidence accumulation
# ═══════════════════════════════════════════════════════════════════════════════

def update_confidence(c: float, progress: bool, gamma: float = GAMMA) -> float:
    """Evidence-weighted update of strategy confidence in [0, 1].

        progress:    c ← c + γ(1 − c)     (reinforce toward 1)
        no-progress: c ← (1 − γ)·c        (geometric decay)

    With γ=0.3, three consecutive no-progress steps drive c from 1.0 to ~0.34
    (< TAU=0.4), while one failure after successes barely dents it — so the
    agent revises a failing hypothesis but does not thrash on a single miss.
    """
    c = max(0.0, min(1.0, c))
    if progress:
        return c + gamma * (1.0 - c)
    return (1.0 - gamma) * c


def needs_restrategy(confidence: float, restrategize_count: int) -> bool:
    """True if confidence has collapsed and we still have re-strategy budget."""
    return confidence < TAU and restrategize_count < MAX_RESTRATEGIZE


# ═══════════════════════════════════════════════════════════════════════════════
#  Stall detection — goal-aware ("busy but not progressing")
# ═══════════════════════════════════════════════════════════════════════════════

def detect_stall(goal_score_window: list[float],
                 eps: float = STALL_EPS,
                 window: int = STALL_WINDOW) -> bool:
    """True if recent goal-scores are flat (no real goal progress).

    Complements CriticV12: CriticV12 answers "did anything change?" (necessary
    for progress); this answers "did we get CLOSER to the goal?" (sufficient).
    Only fires once the window is full, so early exploration isn't punished.
    """
    if len(goal_score_window) < window:
        return False
    recent = goal_score_window[-window:]
    return (max(recent) - min(recent)) < eps


def push_goal_score(window: list[float], score: float) -> list[float]:
    """Append a goal-score, keeping the window bounded."""
    out = list(window)
    out.append(round(float(score), 4))
    return out[-STALL_WINDOW:]


def prm_should_audit(step_number: int, critic_made_progress: bool) -> bool:
    """Throttle the (LLM-costing) PRM goal-audit.

    Run on a fixed cadence, plus opportunistically right after CriticV12 reports
    a change (the moment a goal-score is most likely to have moved).
    """
    return critic_made_progress or (step_number > 0 and step_number % PRM_AUDIT_EVERY == 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Beliefs — bounded, deduplicated, truncated (LEAN memory)
# ═══════════════════════════════════════════════════════════════════════════════

def merge_beliefs(existing: list[str], new_items: list[str]) -> list[str]:
    """Add new beliefs while keeping memory LEAN.

    - truncate each to BELIEF_MAXLEN,
    - drop case-insensitive duplicates / near-duplicates (substring containment),
    - cap to MAX_BELIEFS, dropping the OLDEST first.

    Keeping this list short and non-redundant is what prevents the prompt from
    flooding the model and inducing hallucination.
    """
    out = [b for b in existing]

    def _norm(s: str) -> str:
        return " ".join(s.lower().split())

    for raw in new_items:
        item = (raw or "").strip()
        if not item:
            continue
        item = item[:BELIEF_MAXLEN].rstrip()
        ni = _norm(item)
        # Skip if a near-duplicate already present (either direction of containment)
        if any(ni == _norm(b) or ni in _norm(b) or _norm(b) in ni for b in out):
            continue
        out.append(item)

    if len(out) > MAX_BELIEFS:
        out = out[-MAX_BELIEFS:]
    return out


def render_strategy_block(strategy: str,
                          confidence: float,
                          beliefs: list[str],
                          success_criteria: str = "",
                          goal_complete_hint: str = "") -> str:
    """Render the PERSISTENT cognitive context for the worker prompt.

    Deliberately compact (~6 short lines): a one-line approach, the finish line,
    a confidence percentage, and at most a few short learned facts.

    V27: `goal_complete_hint` is NO LONGER embedded here — transient "finish now"
    nudges are owned by the Guidance Bus (`build_guidance`) so the worker never
    sees more than ONE transient directive per step. The param is kept for
    backward compatibility but ignored.
    """
    if not (strategy or beliefs or success_criteria):
        return ""

    lines = [f"═══ YOUR STRATEGY (confidence: {int(round(confidence * 100))}%) ═══"]
    if strategy:
        lines.append(f"APPROACH: {strategy.strip()[:300]}")
    if success_criteria:
        lines.append(f"DONE WHEN: {success_criteria.strip()[:200]}")

    if beliefs:
        lines.append("WHAT YOU'VE LEARNED:")
        for b in beliefs[-MAX_BELIEFS:]:
            lines.append(f"• {b.strip()[:BELIEF_MAXLEN]}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  V27 Guidance Bus — exactly ONE arbitrated transient directive per step
# ═══════════════════════════════════════════════════════════════════════════════
#  Fixes F2 (audit): the worker prompt used to stack up to five transient,
#  possibly-conflicting directives — including a re-injection of the abstract PRM
#  checklist (`critical_action_hint`), the exact pattern that caused the V26
#  goal-amnesia regression. The single source of truth for sub-goals stays
#  `plan_steps` (rendered in the system prompt). This bus arbitrates only the
#  TRANSIENT nudges, emitting the single highest-priority one.

GUIDANCE_MAXLEN = 400  # keep the directive lean — never crowd out the DOM


def build_guidance(state: dict[str, Any]) -> str:
    """Return the single highest-priority transient directive (or "").

    Priority (highest first), each mapped to an existing state field set by an
    existing node — we centralize only the READER, never the setters:
      0. REALITY    — `reality_note` (V29 Overwatch Reality Monitor): the live
                      screen CONTRADICTED the worker's prediction. This is the most
                      urgent signal — acting on the assumed (wrong) state is the
                      "blind execution" failure — so it outranks everything.
      1. WIN/DONE  — `goal_complete_hint` (commit_node win-state / PRM ceiling):
                     acting further is actively harmful (post-action wandering).
      1.5 STAGNATION — `stagnation_note` (V29 Phase 3): busy but not progressing —
                     break the loop before it deepens.
      2. REPETITION — `consecutive_identical_actions >= 2`: break loops first.
      3. ESCALATION — `correction_context` (overwatch `_escalate` ladder tactic).
      4. RECOVERY   — `recovery_advice` (rollback advisor).

    The abstract PRM-checklist re-injection is deliberately NOT a source here.
    """
    reality = (state.get("reality_note") or "").strip()
    win = (state.get("goal_complete_hint") or "").strip()
    if reality:
        directive = (
            f"🚨 SCREEN-REALITY MISMATCH: {reality} "
            "Do NOT repeat or assume your last action worked. Re-read what is "
            "ACTUALLY on the screen right now and respond to that real state "
            "(handle the error/redirect/popup, or choose a different element)."
        )

    elif win:
        directive = win

    elif (state.get("stagnation_note") or "").strip():
        directive = f"🔁 STAGNATION: {state['stagnation_note'].strip()}"

    elif state.get("consecutive_identical_actions", 0) >= 2:
        n = state.get("consecutive_identical_actions", 0) + 1
        directive = (
            f"🛑 REPETITION BLOCK: you chose the EXACT SAME action {n} times. "
            "It is not working. Choose a COMPLETELY DIFFERENT action now."
        )

    elif (state.get("correction_context") or "").strip():
        directive = state["correction_context"].strip()

    elif (state.get("recovery_advice") or "").strip():
        directive = f"🛠️ RECOVERY ADVICE: {state['recovery_advice'].strip()}"

    else:
        return ""

    directive = directive[:GUIDANCE_MAXLEN].rstrip()
    return f"═══ PRIORITY GUIDANCE (do this before anything else) ═══\n{directive}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Re-strategize — the single LLM call (last rung of the ladder)
# ═══════════════════════════════════════════════════════════════════════════════

async def restrategize(
    invoke_fn,
    chain: list,
    breaker,
    health_tracker,
    *,
    objective: str,
    current_strategy: str,
    beliefs: list[str],
    current_url: str,
    recent_failure: str,
    plan_render: str = "",
) -> tuple[str, str]:
    """Ask the LLM for a genuinely DIFFERENT strategy after the ladder is spent.

    Returns (new_strategy, lesson). On any failure, returns ("", "") so callers
    can keep the old strategy rather than crash.
    """
    system = SystemMessage(content=(
        "You are a senior web-automation strategist. The agent's current approach "
        "is NOT working and it is stuck. Propose a genuinely DIFFERENT overall "
        "strategy — not a tweak of the same path. Be concrete and brief."
    ))
    user = HumanMessage(content=(
        f"OBJECTIVE: {objective}\n"
        f"CURRENT URL: {current_url}\n"
        f"CURRENT (failing) STRATEGY: {current_strategy or '(none)'}\n"
        f"PROGRESS SO FAR:\n{plan_render or '(unknown)'}\n"
        f"WHAT WE KNOW: {'; '.join(beliefs[-MAX_BELIEFS:]) or '(nothing yet)'}\n"
        f"MOST RECENT FAILURE: {recent_failure or '(unknown)'}\n\n"
        "Give a new strategy and the lesson learned."
    ))
    try:
        result, model = await invoke_fn(
            chain, [system, user], Restrategy,
            breaker, health_tracker=health_tracker,
        )
        new_strategy = (getattr(result, "new_strategy", "") or "").strip()
        lesson = (getattr(result, "lesson", "") or "").strip()
        logger.info("🧭 Re-strategized (by %s): %s", model, new_strategy[:150])
        return new_strategy, lesson
    except Exception as e:  # noqa: BLE001
        logger.warning("Re-strategize failed (non-fatal): %s", e)
        return "", ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Clean handoff — explicit reset points (no task bleeds into the next)
# ═══════════════════════════════════════════════════════════════════════════════

def clear_transient() -> dict[str, Any]:
    """State to clear the moment a step SUCCEEDS — the obstacle is resolved, so
    no stale 'you are stuck' directive should carry forward."""
    return {
        "current_obstacle": "",
        "ladder_rung": 0,
        "tried_tactics": [],
        "correction_context": "",
        "recovery_advice": "",
        "consecutive_identical_actions": 0,
        # V29: a resolved step clears the reality discrepancy so a stale
        # "screen mismatch" directive never bleeds into the next sub-task.
        "reality_status": "",
        "reality_note": "",
        # V29 Phase 3: progress was made → wipe the stagnation/loop signals.
        "stagnation_note": "",
        "stagnation_level": 0,
        "scroll_stuck_streak": 0,
        # V29 Intent Journal: a verified-success step resolves any pending intent.
        "last_attempted_action": None,
    }


def clear_all() -> dict[str, Any]:
    """Full cognitive reset at task end (finalize), so a fresh task starts clean."""
    base = clear_transient()
    base.update({
        "strategy": "",
        "success_criteria": "",
        "beliefs": [],
        "strategy_confidence": 1.0,
        "goal_score_window": [],
        "restrategize_count": 0,
        "goal_complete_hint": "",
    })
    return base
