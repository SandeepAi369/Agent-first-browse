"""Sub-Goal Lock tests — the universal "amnesia loop" fix.

Scenario: objective "Do X and Do Y"; Y is verified-complete (locked); a premature
'done' is globally rejected for missing X. The agent must NOT re-do Y.

Pure logic — no browser/network.
Run: .venv/bin/python -m pytest test_subgoal_lock_v29.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import feature_flags as ff
from subgoal_lock import (locked_subgoals, remaining_subgoals, render_lock_list,
                          compose_rejection, reconcile_plan_with_ledger,
                          targets_locked_subgoal)

# Y = "add the blue widget" (DONE & verified-locked); X = "apply discount" (pending)
CHECKLIST = [
    {"desc": "Add the blue widget to the cart", "status": "done",
     "verified": True, "evidence": "cart badge shows 1"},
    {"desc": "Apply the discount code SAVE10", "status": "pending", "verified": False},
]


def test_locked_and_remaining_split():
    assert [d["desc"] for d in locked_subgoals(CHECKLIST)] == ["Add the blue widget to the cart"]
    assert [d["desc"] for d in remaining_subgoals(CHECKLIST)] == ["Apply the discount code SAVE10"]
    assert locked_subgoals([]) == [] and remaining_subgoals(None) == []


def test_lock_list_is_a_forbid_list():
    block = render_lock_list(CHECKLIST)
    assert "ALREADY DONE" in block and "never repeat" in block.lower()
    assert "blue widget" in block
    # a verified item stays locked regardless of button visibility (it's the evidence that locks it)
    assert render_lock_list([{"desc": "Star the repo", "verified": True}]) != ""
    assert render_lock_list([]) == ""


def test_rejection_is_partial_success_not_global_false():
    msg = compose_rejection("discount not applied", "enter SAVE10 and apply", CHECKLIST)
    assert "DONE & LOCKED" in msg and "blue widget" in msg          # re-affirms Y
    assert "STILL REMAINING" in msg and "discount" in msg.lower()    # names X
    assert "do this now" in msg.lower()
    # the message must NOT read as a flat global failure that erases progress
    assert "safe" in msg.lower() and "locked" in msg.lower()


def test_plan_reconciles_to_the_ledger():
    plan = [{"desc": "Add the blue widget to cart", "status": "active"},
            {"desc": "Apply discount code", "status": "pending"}]
    out = reconcile_plan_with_ledger(plan, CHECKLIST)
    assert out is not None
    assert out[0]["status"] == "done"        # locked sub-goal's step → done
    assert out[1]["status"] == "active"      # remaining → activated
    # nothing locked → no change
    assert reconcile_plan_with_ledger(plan, [{"desc": "x", "verified": False}]) is None


def test_backstop_blocks_redo_of_locked_but_not_distinct_remaining():
    # re-doing Y (the locked sub-goal) is detected
    redo = {"verb": "click", "target_name": "Add blue widget", "text": ""}
    assert targets_locked_subgoal(redo, CHECKLIST) is not None
    # the DISTINCT remaining action (X) is NOT falsely blocked
    do_x = {"verb": "click", "target_name": "Apply discount SAVE10", "text": ""}
    assert targets_locked_subgoal(do_x, CHECKLIST) is None
    # non-state-changing verbs are never blocked
    assert targets_locked_subgoal({"verb": "scroll"}, CHECKLIST) is None
    # too few identity tokens → no block (conservative)
    assert targets_locked_subgoal({"verb": "click", "target_name": "Go"}, CHECKLIST) is None


def test_subgoal_lock_flag(monkeypatch):
    monkeypatch.delenv("V29_ENABLED", raising=False)
    monkeypatch.delenv("V29_SUBGOAL_LOCK", raising=False)
    assert ff.subgoal_lock_enabled() is True
    monkeypatch.setenv("V29_SUBGOAL_LOCK", "0")
    assert ff.subgoal_lock_enabled() is False
    monkeypatch.setenv("V29_SUBGOAL_LOCK", "1")
    monkeypatch.setenv("V29_ENABLED", "0")
    assert ff.subgoal_lock_enabled() is False    # master kill-switch wins


def test_wiring():
    ow = (Path(__file__).parent / "overwatch.py").read_text()
    assert "compose_rejection" in ow and "reconcile_plan_with_ledger" in ow
    bw = (Path(__file__).parent / "workers" / "base_worker.py").read_text()
    assert "render_lock_list" in bw and "targets_locked_subgoal" in bw
    assert 'state.get("done_blocked", 0) > 0' in bw   # backstop only in danger zone


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
