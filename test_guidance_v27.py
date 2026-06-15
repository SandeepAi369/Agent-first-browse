"""Unit tests for V27/P1 — the Guidance Bus (single arbitrated directive).

Guarantees:
  - build_guidance returns EXACTLY one block (or empty), honoring priority
    win > repetition > escalation > recovery.
  - The abstract PRM-checklist re-injection (critical_action_hint) is GONE.
  - render_strategy_block no longer embeds the transient goal_complete_hint.

Run: .venv/bin/python -m pytest test_guidance_v27.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import cognition as cog
from cognition import build_guidance, render_strategy_block


def _count_blocks(s: str) -> int:
    return s.count("═══ PRIORITY GUIDANCE")


def test_empty_when_no_transient_directive():
    assert build_guidance({}) == ""
    assert build_guidance({"strategy": "x", "plan_render": "y"}) == ""


def test_win_has_top_priority():
    state = {
        "goal_complete_hint": "GOAL COMPLETE — verify and finish.",
        "consecutive_identical_actions": 5,
        "correction_context": "scroll down",
        "recovery_advice": "dismiss popup",
    }
    out = build_guidance(state)
    assert _count_blocks(out) == 1
    assert "verify and finish" in out.lower()
    assert "scroll down" not in out and "dismiss popup" not in out


def test_repetition_beats_escalation_and_recovery():
    state = {
        "consecutive_identical_actions": 2,
        "correction_context": "scroll down",
        "recovery_advice": "dismiss popup",
    }
    out = build_guidance(state)
    assert _count_blocks(out) == 1
    assert "REPETITION BLOCK" in out
    assert "scroll down" not in out


def test_escalation_beats_recovery():
    state = {"correction_context": "🧗 ADAPTIVE TACTIC [scroll]: scroll down",
             "recovery_advice": "dismiss popup"}
    out = build_guidance(state)
    assert _count_blocks(out) == 1
    assert "ADAPTIVE TACTIC" in out
    assert "dismiss popup" not in out


def test_recovery_is_lowest():
    out = build_guidance({"recovery_advice": "dismiss the cookie banner"})
    assert _count_blocks(out) == 1
    assert "RECOVERY ADVICE" in out
    assert "dismiss the cookie banner" in out


def test_guidance_is_length_bounded():
    out = build_guidance({"correction_context": "x" * 5000})
    body = out.split("\n", 1)[1]
    assert len(body) <= cog.GUIDANCE_MAXLEN


def test_repetition_only_at_threshold():
    # 1 identical action is below threshold (no directive)
    assert build_guidance({"consecutive_identical_actions": 1}) == ""
    assert "REPETITION" in build_guidance({"consecutive_identical_actions": 2})


def test_strategy_block_no_longer_embeds_done_hint():
    block = render_strategy_block(
        strategy="approach", confidence=0.8, beliefs=["b1"],
        success_criteria="cart shows 1",
        goal_complete_hint="GOAL COMPLETE — finish now",  # must be IGNORED
    )
    assert "approach" in block
    assert "GOAL COMPLETE" not in block          # transient hint not embedded
    assert "PRIORITY GUIDANCE" not in block       # that's the bus's job


def test_worker_prompt_has_no_critical_action_hint():
    # The PRM-checklist re-injection string must be gone from the source.
    src = (Path(__file__).parent / "workers" / "base_worker.py").read_text()
    assert "CRITICAL REMAINING ACTION" not in src
    assert "build_guidance" in src


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
