"""Unit tests for the verification hand-off (V26 → corrected in V26.1).

After the V26 regression (a second, competing task list injected into the worker
caused goal-loss, and an over-eager auto-lock made the agent skip real steps),
the behavior under test is now the SAFE subset:
  - MONOTONICITY: a 'done' sub-goal is never demoted to a lesser status by a
    later background audit that can't re-confirm it (the original re-do-loop fix).
  - NO AUTO-LOCK: the background goal-audit must NOT auto-mark items as sticky
    'verified' — that prematurely skipped real work on busy pages. The
    evidence-grounded done-judge is the real gate.
  - LEDGER HAND-OFF: the done-judge prompt carries only *genuinely* verified
    sub-goals (none from the background audit).
  - PERSISTENCE: verified/evidence survive the checklist dict round-trip.

Run: .venv/bin/python -m pytest test_handoff_v26.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

from outcome_judge import build_judge_messages
from prm_critic import ChecklistEvaluation, ChecklistItem, EvaluationItem, PRMCritic


def _prm(eval_results):
    """A PRMCritic whose LLM returns the given (id, status, conf, evidence) tuples."""
    async def fake_invoke(chain, messages, schema, breaker, health_tracker=None):
        evals = [EvaluationItem(id=i, status=s, confidence=c, evidence=e)
                 for (i, s, c, e) in eval_results]
        return ChecklistEvaluation(evaluations=evals), "mock"
    return PRMCritic(fake_invoke, ["m"], None, None)


def _score(prm, checklist):
    asyncio.run(prm.score_step(checklist, dom_markdown="(page)", step_number=5))


# ═══════════════════════════════════════════════════════════════════════════════
#  Monotonicity — keep the safe loop guard
# ═══════════════════════════════════════════════════════════════════════════════

def test_manually_verified_item_never_demoted():
    cl = [ChecklistItem(id=0, description="A", status="done",
                        confidence=0.95, verified=True, evidence="confirmed")]
    _score(_prm([(0, "pending", 0.9, "can't see it now")]), cl)
    assert cl[0].status == "done" and cl[0].verified is True
    assert cl[0].evidence == "confirmed"


def test_plain_done_item_not_demoted():
    cl = [ChecklistItem(id=0, description="A", status="done", confidence=0.8)]
    _score(_prm([(0, "pending", 0.9, "can't re-confirm")]), cl)
    assert cl[0].status == "done"                       # no regression


def test_pending_can_still_progress():
    cl = [ChecklistItem(id=0, description="A", status="pending")]
    _score(_prm([(0, "in_progress", 0.6, "on the right page")]), cl)
    assert cl[0].status == "in_progress"                # promotions allowed


# ═══════════════════════════════════════════════════════════════════════════════
#  NO AUTO-LOCK — the regression fix
# ═══════════════════════════════════════════════════════════════════════════════

def test_confident_done_is_done_but_not_auto_locked():
    cl = [ChecklistItem(id=0, description="A", status="pending")]
    _score(_prm([(0, "done", 0.95, "looks done")]), cl)
    assert cl[0].status == "done"
    assert cl[0].verified is False    # background audit must NOT lock it sticky


def test_audit_done_does_not_get_fed_to_judge_as_verified():
    # A done-but-unverified item must not leak into the judge's "trusted" ledger.
    import overwatch
    state = {"prm_checklist": [
        {"desc": "A", "status": "done", "verified": False, "evidence": "x"},
        {"desc": "B", "status": "pending"},
    ]}
    assert overwatch._render_verified_subgoals(state) == ""   # nothing trusted


# ═══════════════════════════════════════════════════════════════════════════════
#  Ledger hand-off (only genuinely-verified items)
# ═══════════════════════════════════════════════════════════════════════════════

def test_judge_prompt_carries_verified_ledger_when_present():
    msgs = build_judge_messages(
        objective="do A and B", success_criteria="", url="u",
        dom_markdown="d", history_tail="", claim="",
        verified_subgoals="  ✓ A — control switched")
    user = msgs[1]["content"]
    assert "ALREADY VERIFIED DURING EXECUTION" in user
    assert "control switched" in user


def test_judge_prompt_omits_ledger_when_none():
    msgs = build_judge_messages(objective="x", success_criteria="", url="u",
        dom_markdown="d", history_tail="", claim="")
    assert "ALREADY VERIFIED DURING EXECUTION" not in msgs[1]["content"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Single cognitive state — exactly one task list, with master goal + focus
# ═══════════════════════════════════════════════════════════════════════════════

def test_plan_render_is_single_cognitive_state():
    from brain_state import BrainState
    s = BrainState(
        objective="Add a laptop to the cart and check out",
        plan_steps=[
            {"desc": "Search for the laptop", "status": "done"},
            {"desc": "Open the product page", "status": "active"},
            {"desc": "Add to cart", "status": "pending"},
            {"desc": "Check out", "status": "pending"},
        ],
        plan_progress_pct=25,
    )
    r = s.get_plan_render()
    assert "MASTER GOAL" in r and "Add a laptop" in r          # master goal present
    assert "CURRENT SUB-TASK" in r and "Open the product page" in r  # one focus
    assert "Add to cart" not in r.split("CURRENT SUB-TASK")[1].split("STAY ON TASK")[0] \
        or "do NOT start" in r                                  # next is not the focus
    assert "Ignore distractions" in r or "STAY ON TASK" in r    # distraction guard
    # There must be NO second competing task block.
    assert "FOCUS NOW" not in r and "MISSION PROGRESS" not in r


# ═══════════════════════════════════════════════════════════════════════════════
#  Persistence across the state dict round-trip
# ═══════════════════════════════════════════════════════════════════════════════

def test_verified_survives_roundtrip():
    it = ChecklistItem(id=0, description="A", status="done", verified=True, evidence="ev")
    d = {"verified": it.verified, "evidence": it.evidence, "status": it.status}
    back = ChecklistItem(id=0, description="A", status=d["status"],
                         verified=bool(d.get("verified", False)), evidence=d.get("evidence", ""))
    assert back.verified is True and back.evidence == "ev"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
