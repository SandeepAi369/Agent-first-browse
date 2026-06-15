"""Unit tests for the V18 Adaptive Cognition Core.

Covers the guarantees the design (and the user's constraints) depend on:
  - Escalation ladder advances monotonically, never repeats a tactic for the
    same obstacle, resets on obstacle change, and terminates at `restrategize`.
  - Confidence is evidence-weighted (3 no-progress → below τ; reinforcement
    recovers; a single miss after successes does not collapse it).
  - Stall detection fires on a flat goal-score window, not a rising one.
  - Beliefs stay LEAN: capped, de-duplicated, truncated (no prompt overload).
  - The strategy block renders compactly; clear_* gives a clean task handoff.
  - StrategicPlan / Restrategy schemas are Groq-strict-safe.

Run: .venv/bin/python -m pytest test_cognition_v18.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import cognition as cog
from cognition import (
    LADDER, advance_ladder, obstacle_key, update_confidence, needs_restrategy,
    detect_stall, push_goal_score, prm_should_audit, merge_beliefs,
    render_strategy_block, clear_transient, clear_all,
    GAMMA, TAU, MAX_BELIEFS, MAX_RESTRATEGIZE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Escalation ladder
# ═══════════════════════════════════════════════════════════════════════════════

def _drive_ladder(obstacle: str, n: int):
    """Drive the ladder n times for one obstacle, returning the tactic sequence."""
    cur, rung, tried = "", 0, []
    tactics = []
    for _ in range(n):
        lad = advance_ladder(cur, obstacle, rung, tried)
        tactics.append(lad["tactic"])
        cur, rung, tried = lad["obstacle"], lad["rung"], lad["tried"]
    return tactics, cur, rung, tried


def test_ladder_never_repeats_a_tactic_for_same_obstacle():
    tactics, *_ = _drive_ladder("urlA|stepA", len(LADDER))
    # Every non-restrategize tactic appears at most once
    non_restrat = [t for t in tactics if t != "restrategize"]
    assert len(non_restrat) == len(set(non_restrat)), tactics


def test_ladder_is_monotonic_and_matches_order():
    tactics, *_ = _drive_ladder("urlA|stepA", len(LADDER))
    expected = [t for t, _ in LADDER]
    assert tactics == expected, tactics


def test_ladder_terminates_at_restrategize():
    # Drive well past the ladder length — it must keep returning restrategize
    tactics, _, _, _ = _drive_ladder("urlA|stepA", len(LADDER) + 3)
    assert tactics[-1] == "restrategize"
    assert tactics[-2] == "restrategize"  # stays terminal, never wraps around


def test_ladder_resets_on_obstacle_change():
    # Advance a few rungs on obstacle A
    cur, rung, tried = "", 0, []
    for _ in range(3):
        lad = advance_ladder(cur, "urlA|stepA", rung, tried)
        cur, rung, tried = lad["obstacle"], lad["rung"], lad["tried"]
    # Now the obstacle changes → ladder resets to rung 0 (reperceive)
    lad = advance_ladder(cur, "urlB|stepB", rung, tried)
    assert lad["tactic"] == LADDER[0][0]
    assert lad["rung"] == 1
    assert lad["tried"] == [LADDER[0][0]]


def test_obstacle_key_distinguishes_url_and_step():
    assert obstacle_key("u", "s1") != obstacle_key("u", "s2")
    assert obstacle_key("u1", "s") != obstacle_key("u2", "s")
    assert obstacle_key("u", "s") == obstacle_key("u", "s")


# ═══════════════════════════════════════════════════════════════════════════════
#  Confidence
# ═══════════════════════════════════════════════════════════════════════════════

def test_confidence_three_no_progress_crosses_tau():
    c = 1.0
    for _ in range(3):
        c = update_confidence(c, progress=False)
    assert c < TAU, c                      # ~0.343
    # Two failures alone should NOT cross τ
    c2 = update_confidence(update_confidence(1.0, False), False)
    assert c2 >= TAU, c2                    # ~0.49


def test_confidence_single_miss_after_successes_is_resilient():
    c = 1.0
    for _ in range(3):
        c = update_confidence(c, progress=True)   # stays at 1.0
    c = update_confidence(c, progress=False)
    assert c >= TAU                          # one miss doesn't collapse a strong prior


def test_confidence_reinforcement_recovers():
    c = 0.3
    for _ in range(5):
        c = update_confidence(c, progress=True)
    assert c > 0.8


def test_needs_restrategy_respects_budget():
    assert needs_restrategy(0.2, 0) is True
    assert needs_restrategy(0.2, MAX_RESTRATEGIZE) is False   # budget spent
    assert needs_restrategy(0.9, 0) is False                  # confident, no need


# ═══════════════════════════════════════════════════════════════════════════════
#  Stall detection
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_stall_on_flat_window():
    window = []
    for s in [0.5, 0.51, 0.5, 0.52]:
        window = push_goal_score(window, s)
    assert detect_stall(window) is True


def test_no_stall_on_rising_window():
    window = []
    for s in [0.2, 0.4, 0.6, 0.85]:
        window = push_goal_score(window, s)
    assert detect_stall(window) is False


def test_no_stall_before_window_full():
    window = push_goal_score(push_goal_score([], 0.5), 0.5)
    assert detect_stall(window) is False     # only 2 samples


def test_push_goal_score_bounds_window():
    window = []
    for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        window = push_goal_score(window, s)
    assert len(window) == cog.STALL_WINDOW


def test_prm_should_audit_cadence_and_progress():
    from cognition import PRM_AUDIT_EVERY            # tracks the tuned cadence (V-parallel: 2)
    assert prm_should_audit(0, True) is True         # critic progress always audits
    assert prm_should_audit(PRM_AUDIT_EVERY, False) is True       # on-cadence
    assert prm_should_audit(PRM_AUDIT_EVERY + 1, False) is False  # off-cadence, no progress


# ═══════════════════════════════════════════════════════════════════════════════
#  Beliefs — LEAN memory (no overload)
# ═══════════════════════════════════════════════════════════════════════════════

def test_beliefs_dedup_case_and_substring():
    beliefs = merge_beliefs([], ["Login is not required to search"])
    beliefs = merge_beliefs(beliefs, ["login is not required to search"])  # case dup
    beliefs = merge_beliefs(beliefs, ["Login is not required"])            # substring
    assert len(beliefs) == 1


def test_beliefs_capped_drop_oldest():
    beliefs = []
    for i in range(MAX_BELIEFS + 4):
        beliefs = merge_beliefs(beliefs, [f"distinct belief number {i}"])
    assert len(beliefs) == MAX_BELIEFS
    # Oldest dropped, newest kept
    assert any(f"number {MAX_BELIEFS + 3}" in b for b in beliefs)
    assert not any("number 0" in b for b in beliefs)


def test_beliefs_truncated():
    long = "x" * 500
    beliefs = merge_beliefs([], [long])
    assert len(beliefs[0]) <= cog.BELIEF_MAXLEN


def test_strategy_block_is_compact_and_includes_done():
    block = render_strategy_block(
        strategy="Use top-nav search then add to cart",
        confidence=0.72,
        beliefs=["Login not required", "Results load on scroll"],
        success_criteria="Cart shows 1 item",
    )
    assert "72%" in block
    assert "DONE WHEN" in block
    assert block.count("•") == 2
    # Stays compact (a handful of short lines)
    assert len(block.splitlines()) <= 8


def test_strategy_block_empty_when_no_cognition():
    assert render_strategy_block("", 1.0, [], "") == ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Clean handoff
# ═══════════════════════════════════════════════════════════════════════════════

def test_clear_transient_resets_obstacle_not_strategy():
    t = clear_transient()
    assert t["current_obstacle"] == "" and t["ladder_rung"] == 0
    assert t["tried_tactics"] == [] and t["correction_context"] == ""
    assert "strategy" not in t          # strategy survives a successful step


def test_clear_all_resets_everything():
    a = clear_all()
    assert a["strategy"] == "" and a["beliefs"] == []
    assert a["strategy_confidence"] == 1.0 and a["restrategize_count"] == 0
    assert a["goal_score_window"] == [] and a["current_obstacle"] == ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Schemas strict-safe (Groq strict mode)
# ═══════════════════════════════════════════════════════════════════════════════

def test_schemas_strict_safe():
    assert cog.StrategicPlan.model_json_schema()["additionalProperties"] is False
    assert cog.Restrategy.model_json_schema()["additionalProperties"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
