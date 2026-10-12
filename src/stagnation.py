"""Stagnation Detector — progress-aware loop breaking (V29 / Phase 3, Mandate 3).

THE LOOP CURE
═════════════
The agent gets stuck "busy but not progressing": it keeps clicking/scrolling and
the DOM keeps changing (so the per-step critic sees "progress"), yet it makes no
real headway toward the goal — same URL forever, a flat goal-score, or a short
cycle of repeating actions (A,B,A,B…). The existing guards catch EXACT repeats
([overwatch] loop_signatures) and identical consecutive actions; they miss this
*productive-looking churn*.

Following PABU (Progress-Aware Belief Update, arXiv 2602.09138), we model progress
as a task-dependent but environment-agnostic abstraction and combine three cheap,
already-computed signals:

  1. URL-stuck     — `same_url_streak` (computed every step, previously UNUSED in
                     the live graph — revived here).
  2. Goal-flat     — the PRM goal-score window is flat (reuses cognition.detect_stall).
  3. Action-cycle  — the recent action signatures form a short repeating cycle.

When ≥2 fire, we declare stagnation and emit a directive that tells the agent it is
NOT making real progress toward its goal and must change approach (different
element, scroll, or re-navigate) — and, at the higher level, opens the eyes to
re-orient. Universal: no site/task rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

URL_STUCK_THRESHOLD = 4      # steps on the same URL before it counts as a signal
CYCLE_MIN_REPEATS = 2        # a 2-step cycle repeated ≥2× (A,B,A,B) = churn


@dataclass
class StagnationSignal:
    stuck: bool = False
    level: int = 0                       # number of signals that fired (0..3)
    reasons: list[str] = field(default_factory=list)
    note: str = ""                       # directive for the guidance bus


def _has_action_cycle(signatures: list[str]) -> bool:
    """Detect a short repeating cycle in the recent action signatures.

    Catches A,A,A (length-1 cycle) and A,B,A,B (length-2 cycle) — the classic
    'alternate between two useless actions' churn that exact-repeat guards miss.
    """
    sigs = [s for s in (signatures or []) if s]
    if len(sigs) < 4:
        return False
    tail = sigs[-6:]
    # length-1: last 3 identical
    if len(set(tail[-3:])) == 1:
        return True
    # length-2: last 4 form X,Y,X,Y with X != Y
    last4 = tail[-4:]
    if len(last4) == 4 and last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
        return True
    return False


def detect_stagnation(state: dict) -> StagnationSignal:
    """Combine the progress signals into a single stagnation verdict (pure)."""
    from cognition import detect_stall

    reasons: list[str] = []

    same_url = int(state.get("same_url_streak", 0) or 0)
    url_stuck = same_url >= URL_STUCK_THRESHOLD
    if url_stuck:
        reasons.append(f"on the same URL for {same_url} steps")

    goal_flat = detect_stall(state.get("goal_score_window", []) or [])
    if goal_flat:
        reasons.append("goal-progress score is flat")

    cycle = _has_action_cycle(state.get("loop_signatures", []) or [])
    if cycle:
        reasons.append("repeating a short action cycle")

    level = sum((url_stuck, goal_flat, cycle))
    stuck = level >= 2
    note = ""
    if stuck:
        note = (
            "You are NOT making real progress toward the goal ("
            + "; ".join(reasons)
            + "). Stop repeating variations of the same approach. Re-read the page, "
            "and either act on a clearly DIFFERENT element that advances the goal, "
            "scroll to reveal something new, or re-navigate — do not loop."
        )
    return StagnationSignal(stuck=stuck, level=level, reasons=reasons, note=note)
