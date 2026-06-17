"""Unit tests for V29 Phase 2 + 3 + the Contextual-Focus mandate.

Covers (pure logic — no browser/network):
  • target_lock: semantic target binding, look-alike counting, off-target risk,
    and the temporal self-questioning prompt block.
  • clarity: the uncertainty signal + the broadened PRE-action consensus gate +
    the vision-for-ambiguity gate.
  • stagnation: progress-aware loop detection (revived same_url_streak, goal-flat,
    action-cycle) and its guidance-bus priority.
  • flags + clear_transient + wiring guards.

Run: .venv/bin/python -m pytest test_phase23_v29.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import feature_flags as ff
import target_lock as tl
from clarity import (compute_clarity, needs_consensus, needs_vision_for_clarity,
                     ClaritySignal)
from stagnation import detect_stagnation, _has_action_cycle
from cognition import build_guidance, clear_transient


# ═══════════════════════════════════════════════════════════════════════════════
#  Target Lock
# ═══════════════════════════════════════════════════════════════════════════════

def test_extract_target_pulls_item_identity():
    t = tl.extract_target("Add the 'Sauce Labs Fleece Jacket' to the cart",
                          "Click Add to Cart for the Fleece Jacket")
    assert "fleece" in t.tokens and "jacket" in t.tokens
    assert "cart" not in t.tokens   # action/chrome word is stripped
    assert t.specified


def test_is_primary_action_click():
    assert tl.is_primary_action_click("click", "Add to cart", "")
    assert tl.is_primary_action_click("click", "", "Buy Now")
    assert not tl.is_primary_action_click("click", "Home", "Nav link")
    assert not tl.is_primary_action_click("type", "Add to cart", "")  # not a click


def test_count_lookalikes():
    smap = {
        "e1": {"name": "Add to cart", "text": "Onesie"},
        "e2": {"name": "Add to cart", "text": "Fleece Jacket"},
        "e3": {"name": "Add to cart", "text": "Bike Light"},
        "e9": {"name": "Home", "text": ""},
    }
    assert tl.count_lookalikes(smap, "Add to cart") == 3
    assert tl.count_lookalikes(smap, "Home") == 0  # not a primary action


def test_off_target_risk_detects_wrong_item():
    target = tl.extract_target("Add the 'Fleece Jacket' to cart", "Add the Fleece Jacket to cart")
    smap = {
        "e1": {"name": "Add to cart", "text": "Onesie"},
        "e2": {"name": "Add to cart", "text": "Fleece Jacket"},
    }
    # chosen = the Onesie button (wrong item) → risk
    assert tl.off_target_risk(target, "Add to cart Onesie", smap, "Add to cart") is True
    # chosen = the Fleece Jacket button (right item) → no risk
    assert tl.off_target_risk(target, "Add to cart Fleece Jacket", smap, "Add to cart") is False


def test_target_lock_block_has_temporal_self_check():
    t = tl.extract_target("Add the 'Fleece Jacket'", "Add the Fleece Jacket to cart")
    block = tl.render_target_lock_block(t, "Add the Fleece Jacket")
    assert "TARGET LOCK" in block
    assert "TEMPORAL SELF-CHECK" in block
    assert "neighboring" in block.lower() or "neighbor" in block.lower()
    assert tl.render_target_lock_block(tl.TargetDescriptor(), "") == ""  # no target → no block


# ═══════════════════════════════════════════════════════════════════════════════
#  Clarity Gate — uncertainty-triggered PRE-action consensus
# ═══════════════════════════════════════════════════════════════════════════════

def test_low_confidence_is_low_clarity():
    sig = compute_clarity({"verb": "click", "confidence": 0.3}, {})
    assert sig.low_clarity and sig.uncertain


def test_needs_vision_flag_is_low_clarity():
    sig = compute_clarity({"verb": "click", "confidence": 0.9, "needs_vision": True}, {})
    assert sig.low_clarity


def test_reality_contradiction_is_low_clarity():
    sig = compute_clarity({"verb": "click", "confidence": 0.9},
                          {"reality_status": "CONTRADICTED"})
    assert sig.low_clarity


def test_target_ambiguity_triggers_even_when_confident():
    smap = {"e1": {"name": "Add to cart", "text": "A"},
            "e2": {"name": "Add to cart", "text": "B"}}
    sig = compute_clarity(
        {"verb": "click", "target_name": "Add to cart", "confidence": 0.95,
         "element_id": "e1"},
        {"selector_map": smap}, selector_map=smap)
    assert sig.target_ambiguity and sig.uncertain


def test_consensus_gate_irreversible_always():
    clear = ClaritySignal()  # perfectly clear
    do, _ = needs_consensus(clear, is_irreversible=True, broaden=False)
    assert do is True


def test_consensus_gate_broaden_off_only_irreversible():
    uncertain = ClaritySignal(low_clarity=True, reasons=["low confidence"])
    do, _ = needs_consensus(uncertain, is_irreversible=False, broaden=False)
    assert do is False  # broadening disabled → reversible low-clarity does NOT vote


def test_consensus_gate_broaden_on_votes_on_uncertainty():
    uncertain = ClaritySignal(target_ambiguity=True)
    do, why = needs_consensus(uncertain, is_irreversible=False, broaden=True)
    assert do is True and why


def test_vision_for_clarity_on_ambiguity():
    v, _ = needs_vision_for_clarity(ClaritySignal(off_target=True))
    assert v is True
    v2, _ = needs_vision_for_clarity(ClaritySignal())
    assert v2 is False


# ═══════════════════════════════════════════════════════════════════════════════
#  Stagnation — progress-aware loop breaking
# ═══════════════════════════════════════════════════════════════════════════════

def test_action_cycle_detection():
    assert _has_action_cycle(["click:a", "scroll:", "click:a", "scroll:"])  # A,B,A,B
    assert _has_action_cycle(["x", "click:a", "click:a", "click:a"])         # A,A,A
    assert not _has_action_cycle(["a", "b", "c", "d"])                       # all distinct


def test_stagnation_needs_two_signals():
    # only URL-stuck (1 signal) → not stuck
    s1 = detect_stagnation({"same_url_streak": 6, "goal_score_window": [], "loop_signatures": []})
    assert not s1.stuck and s1.level == 1
    # URL-stuck + flat goal score (2 signals) → stuck
    s2 = detect_stagnation({"same_url_streak": 6,
                            "goal_score_window": [0.5, 0.5, 0.5, 0.5],
                            "loop_signatures": []})
    assert s2.stuck and s2.level >= 2 and s2.note


def test_stagnation_guidance_priority():
    # stagnation outranks repetition …
    out = build_guidance({"stagnation_note": "no real progress",
                          "consecutive_identical_actions": 5})
    assert "STAGNATION" in out and "REPETITION" not in out
    # … but win and reality outrank stagnation.
    assert "STAGNATION" not in build_guidance(
        {"goal_complete_hint": "finish now", "stagnation_note": "x"})
    assert "SCREEN-REALITY" in build_guidance(
        {"reality_note": "mismatch", "stagnation_note": "x"})


def test_clear_transient_wipes_phase3_fields():
    d = clear_transient()
    assert d["stagnation_note"] == "" and d["stagnation_level"] == 0
    assert d["scroll_stuck_streak"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Flags + wiring guards
# ═══════════════════════════════════════════════════════════════════════════════

def test_target_lock_flag(monkeypatch):
    monkeypatch.delenv("V29_ENABLED", raising=False)
    monkeypatch.delenv("V29_TARGET_LOCK", raising=False)
    assert ff.target_lock_enabled() is True
    monkeypatch.setenv("V29_TARGET_LOCK", "0")
    assert ff.target_lock_enabled() is False
    monkeypatch.setenv("V29_TARGET_LOCK", "1")
    monkeypatch.setenv("V29_ENABLED", "0")
    assert ff.target_lock_enabled() is False  # master kill-switch wins


def test_wiring_guards():
    bw = (Path(__file__).parent / "workers" / "base_worker.py").read_text()
    assert "render_target_lock_block" in bw and "needs_consensus" in bw
    assert "PRE-ACTION CONSENSUS" in bw
    ow = (Path(__file__).parent / "overwatch.py").read_text()
    assert "scroll_stuck_streak" in ow and "bound_target" in ow
    bg = (Path(__file__).parent / "brain_graph.py").read_text()
    assert "detect_stagnation" in bg


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
