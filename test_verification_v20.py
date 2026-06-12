"""Unit tests for V20 Outcome Verification (the evidence-grounded done-gate).

Guarantees under test:
  - should_accept: only a confident, achieved verdict passes.
  - rejection_feedback: actionable (names the gap + one concrete next move).
  - The L4 gate: judge-accept → pass + mission_success + cited evidence;
    judge-reject → retry + correction_context the worker can act on;
    judge unavailable → legacy heuristic fallback (no regression);
    auth URL → blocked before any LLM call;
    MAX_DONE_BLOCKS rejections → honest finalize (escalate), never an
    infinite block-loop.

Run: .venv/bin/python -m pytest test_verification_v20.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import dom_parser
import overwatch
from outcome_judge import (
    MAX_DONE_BLOCKS,
    DoneVerdict,
    build_judge_messages,
    rejection_feedback,
    should_accept,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_should_accept_requires_confident_achieved():
    ok = DoneVerdict(achieved=True, confidence=0.9, evidence="cart shows item")
    assert should_accept(ok) is True
    low = DoneVerdict(achieved=True, confidence=0.3, evidence="maybe")
    assert should_accept(low) is False
    no = DoneVerdict(achieved=False, confidence=0.95, evidence="no cart change",
                     missing="cart not updated")
    assert should_accept(no) is False


def test_rejection_feedback_is_actionable():
    v = DoneVerdict(
        achieved=False, confidence=0.8, evidence="product page only",
        missing="cart page does not list the product",
        next_hint="open the cart page and check the item is listed",
    )
    fb = rejection_feedback(v)
    assert "cart page does not list the product" in fb
    assert "open the cart page" in fb
    assert "done" in fb.lower()


def test_judge_messages_carry_evidence_pack():
    msgs = build_judge_messages(
        objective="Add a water bottle to the cart",
        success_criteria="Cart shows the bottle",
        url="https://example.com/viewcart",
        dom_markdown="- [e1] button: Place order",
        history_tail="click(e2) → OK",
        claim="I added it",
        page_text="My Cart (1) AQUENCH 1L Stainless Steel Bottle ₹253",
    )
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    for needle in ("Add a water bottle", "Cart shows the bottle",
                   "viewcart", "Place order", "click(e2)", "I added it",
                   "AQUENCH 1L Stainless Steel"):
        assert needle in user, needle
    # The claim must be framed as NOT evidence (anti-parrot).
    assert "NOT evidence" in user
    # The judge must derive task-specific proof, not run a fixed checklist.
    assert "SITUATIONALLY" in msgs[0]["content"]


def test_judge_messages_without_page_text_omit_section():
    msgs = build_judge_messages(
        objective="x", success_criteria="", url="u", dom_markdown="d",
        history_tail="", claim="",
    )
    assert "Visible page text" not in msgs[1]["content"]


# ═══════════════════════════════════════════════════════════════════════════════
#  L4 gate — fixtures
# ═══════════════════════════════════════════════════════════════════════════════

class _Page:
    url = "https://shop.example/item/42"

    async def evaluate(self, js):
        return "BODY TEXT: item added to cart successfully"


def _state(**over) -> dict:
    base = {
        "objective": "Add the bottle to the cart",
        "success_criteria": "Cart page lists the bottle",
        "proposed_action": {"verb": "done", "screen_state": "cart shows bottle"},
        "dom_markdown": "- [e1] link: Go to cart",
        "history_compressed": "step5 click(Add to cart) → OK",
        "history": [],
        "plan_steps": [],
        "plan_cursor": 0,
        "step_number": 6,
        "done_blocked": 0,
    }
    base.update(over)
    return base


def _mk_invoke(verdict):
    calls = []
    async def invoke(chain, messages, schema, breaker, health_tracker=None):
        calls.append(messages)
        if isinstance(verdict, Exception):
            raise verdict
        return verdict, "mock-model"
    invoke.calls = calls
    return invoke


@pytest.fixture(autouse=True)
def _fresh_snapshot(monkeypatch):
    """The gate must judge the page AS IT IS NOW — stub the fresh extract."""
    async def fake_extract(page, target_hint=None, timeout=5.0):
        return {"elements": [{"id": "e1"}],
                "markdown": "- [e1] link: Go to cart (FRESH)"}
    monkeypatch.setattr(dom_parser, "extract", fake_extract)
    yield
    overwatch.configure_outcome_judge(None, [], None, None)


# ═══════════════════════════════════════════════════════════════════════════════
#  L4 gate — behavior
# ═══════════════════════════════════════════════════════════════════════════════

def test_judge_accept_passes_with_evidence():
    v = DoneVerdict(achieved=True, confidence=0.9,
                    evidence="'Go to cart' visible — item added")
    invoke = _mk_invoke(v)
    overwatch.configure_outcome_judge(invoke, ["m"], None, None)

    out = asyncio.run(overwatch._layer_4_cove_check(_state(), _Page(), {}))
    assert out["overwatch_verdict"] == "pass"
    assert out["mission_success"] is True
    assert "Go to cart" in out["done_evidence"]
    # Judge saw the FRESH snapshot, not the stale state markdown — and the
    # rendered body text (where non-clickable proof lives).
    sent = str(invoke.calls[0])
    assert "(FRESH)" in sent
    assert "item added to cart successfully" in sent


def test_judge_reject_blocks_with_actionable_feedback():
    v = DoneVerdict(achieved=False, confidence=0.9, evidence="product page only",
                    missing="cart does not list the bottle",
                    next_hint="open the cart page")
    overwatch.configure_outcome_judge(_mk_invoke(v), ["m"], None, None)

    out = asyncio.run(overwatch._layer_4_cove_check(_state(), _Page(), {}))
    assert out["overwatch_verdict"] == "retry"
    assert out["done_blocked"] == 1
    assert out.get("mission_success") is not True
    cc = out.get("correction_context", "")
    assert "cart does not list the bottle" in cc
    assert "open the cart page" in cc


def test_done_block_cap_finalizes_honestly():
    v = DoneVerdict(achieved=False, confidence=0.9, evidence="no proof",
                    missing="no confirmation anywhere")
    overwatch.configure_outcome_judge(_mk_invoke(v), ["m"], None, None)

    st = _state(done_blocked=MAX_DONE_BLOCKS - 1)
    out = asyncio.run(overwatch._layer_4_cove_check(st, _Page(), {}))
    assert out["overwatch_verdict"] == "escalate"   # → finalize
    assert out["mission_success"] is False
    assert "UNVERIFIED" in out["done_evidence"]


def test_auth_page_blocks_before_judge():
    class _AuthPage:
        url = "https://shop.example/login?next=/cart"
    invoke = _mk_invoke(DoneVerdict(achieved=True, confidence=1.0, evidence="x"))
    overwatch.configure_outcome_judge(invoke, ["m"], None, None)

    out = asyncio.run(overwatch._layer_4_cove_check(_state(), _AuthPage(), {}))
    assert out["overwatch_verdict"] == "retry"
    assert invoke.calls == []          # no LLM spent on an obvious non-success


def test_judge_unavailable_falls_back_to_heuristics():
    # Unconfigured judge → legacy heuristic path. Build a state the heuristics
    # accept: plan complete + critical action present in the trail.
    overwatch.configure_outcome_judge(None, [], None, None)
    st = _state(
        plan_steps=[{"desc": "add bottle to cart", "status": "done"}],
        history=[{"step": 5, "action": "click on Add to cart", "outcome": "→ OK"}],
    )
    out = asyncio.run(overwatch._layer_4_cove_check(st, _Page(), {}))
    assert out["overwatch_verdict"] == "pass"
    assert out["mission_success"] is True
    assert "heuristic" in out["done_evidence"]


def test_judge_error_also_falls_back():
    overwatch.configure_outcome_judge(_mk_invoke(RuntimeError("503")), ["m"], None, None)
    st = _state(
        plan_steps=[{"desc": "add bottle to cart", "status": "done"}],
        history=[{"step": 5, "action": "click on Add to cart", "outcome": "→ OK"}],
    )
    out = asyncio.run(overwatch._layer_4_cove_check(st, _Page(), {}))
    # Fallback heuristics accept this state — verification degrades, never crashes.
    assert out["overwatch_verdict"] == "pass"


# ═══════════════════════════════════════════════════════════════════════════════
#  Final outcome audit (shutdown WITHOUT a verified done)
# ═══════════════════════════════════════════════════════════════════════════════

def test_final_audit_confirms_goal_on_shutdown():
    """Budget exhausted but the goal evidence is on screen → verified success."""
    v = DoneVerdict(achieved=True, confidence=0.95,
                    evidence="'Go to cart' state + cart count 1 visible")
    overwatch.configure_outcome_judge(_mk_invoke(v), ["m"], None, None)

    achieved, evidence = asyncio.run(
        overwatch.final_outcome_audit(_state(), _Page()))
    assert achieved is True
    assert "Go to cart" in evidence


def test_final_audit_honest_failure():
    v = DoneVerdict(achieved=False, confidence=0.9, evidence="still on homepage",
                    missing="no cart evidence anywhere")
    overwatch.configure_outcome_judge(_mk_invoke(v), ["m"], None, None)

    achieved, evidence = asyncio.run(
        overwatch.final_outcome_audit(_state(), _Page()))
    assert achieved is False
    assert "no cart evidence" in evidence


def test_final_audit_unverifiable_when_judge_offline():
    overwatch.configure_outcome_judge(None, [], None, None)
    achieved, evidence = asyncio.run(
        overwatch.final_outcome_audit(_state(), _Page()))
    assert achieved is False
    assert "unverifiable" in evidence


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
