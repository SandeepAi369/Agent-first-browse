"""Unit tests for V26 — Critical-Thinker verification + hand-off (Star/Fork loop fix).

Guarantees under test (all task-agnostic / generalized — no UI keywords):
  - MONOTONICITY (the smoking-gun fix): a VERIFIED sub-goal is never demoted; a
    plain 'done' sub-goal is never demoted to a lesser status by a later audit
    that can't re-confirm it; a confident new 'done' becomes sticky-verified with
    its captured evidence; a low-confidence 'done' is done but NOT locked.
  - SERIALIZATION: the mission block exposes exactly ONE focus sub-goal, lists
    verified ones as done-do-not-repeat, and the rest as later.
  - LEDGER HAND-OFF: the done-judge prompt carries the verified-sub-goals ledger.
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
from workers.base_worker import _render_mission_progress


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
#  Monotonicity — the core loop fix
# ═══════════════════════════════════════════════════════════════════════════════

def test_verified_item_never_demoted():
    cl = [ChecklistItem(id=0, description="sub-goal A", status="done",
                        confidence=0.95, verified=True, evidence="confirmed earlier")]
    _score(_prm([(0, "pending", 0.9, "can't see it now")]), cl)   # audit tries to demote
    assert cl[0].status == "done" and cl[0].verified is True
    assert cl[0].evidence == "confirmed earlier"                  # untouched


def test_plain_done_item_not_demoted():
    cl = [ChecklistItem(id=0, description="sub-goal A", status="done", confidence=0.8)]
    _score(_prm([(0, "pending", 0.9, "can't re-confirm")]), cl)
    assert cl[0].status == "done"                                 # no regression


def test_confident_done_becomes_sticky_verified():
    cl = [ChecklistItem(id=0, description="sub-goal A", status="pending")]
    _score(_prm([(0, "done", 0.9, "control switched to its active state")]), cl)
    assert cl[0].status == "done" and cl[0].verified is True
    assert "active state" in cl[0].evidence
    assert cl[0].step_completed == 5


def test_low_confidence_done_not_locked():
    cl = [ChecklistItem(id=0, description="sub-goal A", status="pending")]
    _score(_prm([(0, "done", 0.5, "maybe")]), cl)
    assert cl[0].status == "done" and cl[0].verified is False     # done but not sticky


def test_pending_can_still_progress():
    cl = [ChecklistItem(id=0, description="sub-goal A", status="pending")]
    _score(_prm([(0, "in_progress", 0.6, "on the right page")]), cl)
    assert cl[0].status == "in_progress"                          # promotions allowed


def test_two_subgoals_independent():
    cl = [ChecklistItem(id=0, description="A", status="done", verified=True, evidence="e"),
          ChecklistItem(id=1, description="B", status="pending")]
    # Audit demotes A (ignored) and completes B.
    _score(_prm([(0, "pending", 0.9, "x"), (1, "done", 0.9, "B confirmed")]), cl)
    assert cl[0].status == "done" and cl[0].verified is True      # A protected
    assert cl[1].status == "done" and cl[1].verified is True      # B newly verified


# ═══════════════════════════════════════════════════════════════════════════════
#  Serialization — one sub-goal at a time, never repeat a verified one
# ═══════════════════════════════════════════════════════════════════════════════

def test_mission_progress_one_focus():
    cl = [{"desc": "A", "status": "done", "verified": True, "evidence": "x"},
          {"desc": "B", "status": "pending"},
          {"desc": "C", "status": "pending"}]
    r = _render_mission_progress(cl)
    focus = r.split("FOCUS NOW")[1].split("LATER")[0]
    assert "B" in focus and "C" not in focus                      # exactly one focus
    assert "A" in r.split("ALREADY DONE")[1].split("FOCUS")[0]    # A is done
    assert "C" in r.split("LATER")[1]                             # C deferred
    assert "never over-verify" in r.lower()                       # critical-thinker guidance


def test_mission_progress_all_done_no_focus():
    cl = [{"desc": "A", "status": "done", "verified": True},
          {"desc": "B", "status": "done", "verified": True}]
    r = _render_mission_progress(cl)
    assert "FOCUS NOW" not in r                                   # nothing left to do


def test_mission_progress_empty():
    assert _render_mission_progress([]) == ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Ledger hand-off to the done-judge
# ═══════════════════════════════════════════════════════════════════════════════

def test_judge_prompt_carries_verified_ledger():
    msgs = build_judge_messages(
        objective="do A and B", success_criteria="", url="u",
        dom_markdown="d", history_tail="", claim="",
        verified_subgoals="  ✓ A — control switched\n  ✓ B — redirected")
    user = msgs[1]["content"]
    assert "ALREADY VERIFIED DURING EXECUTION" in user
    assert "control switched" in user and "redirected" in user
    assert "do not mark the task incomplete merely because" in user


def test_judge_prompt_omits_ledger_when_none():
    msgs = build_judge_messages(objective="x", success_criteria="", url="u",
        dom_markdown="d", history_tail="", claim="")
    assert "ALREADY VERIFIED DURING EXECUTION" not in msgs[1]["content"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Persistence across the state dict round-trip
# ═══════════════════════════════════════════════════════════════════════════════

def test_verified_survives_roundtrip():
    it = ChecklistItem(id=0, description="A", status="done", verified=True, evidence="ev")
    d = {"desc": it.description, "status": it.status, "confidence": it.confidence,
         "step_completed": it.step_completed, "verified": it.verified, "evidence": it.evidence}
    back = ChecklistItem(
        id=0, description=d["desc"], status=d["status"], confidence=d["confidence"],
        step_completed=d["step_completed"], verified=bool(d.get("verified", False)),
        evidence=d.get("evidence", ""))
    assert back.verified is True and back.evidence == "ev"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
