"""Unit tests for V29 / Phase 1 — the Reality Monitor (screen-reality reconciliation).

Covers the blind-execution cure end to end at the logic layer:
  • classify_reality: CONTRADICTED / CONFIRMED / UNCLEAR / NULL across the real
    failure shapes (error toast, out-of-stock, unexpected auth redirect, success).
  • the guidance bus: a CONTRADICTED note outranks every other transient directive.
  • clear_transient: a resolved step wipes the discrepancy (no bleed-through).
  • feature flags: master kill-switch + per-feature switches behave correctly.
  • wiring guards: Overwatch + BrainState actually carry the new layer.

Pure logic — no browser, no network.
Run: .venv/bin/python -m pytest test_reality_v29.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import feature_flags as ff
from reality import (classify_reality, CONFIRMED, CONTRADICTED, UNCLEAR, NULL)
from cognition import build_guidance, clear_transient


# ═══════════════════════════════════════════════════════════════════════════════
#  classify_reality — the deterministic core
# ═══════════════════════════════════════════════════════════════════════════════

def test_contradiction_new_error_text():
    rv = classify_reality(
        expected_change="the item is added to the cart and the count increments",
        verb="click",
        pre_text="Product page. In stock. Add to Cart.",
        post_text="Error: this item is out of stock and cannot be added.",
        critic_success=True,  # the DOM DID change → CriticV12 would say 'progress'
    )
    assert rv.status == CONTRADICTED
    assert "out of stock" in rv.note or "error" in rv.note.lower()


def test_no_contradiction_when_error_word_already_present():
    # 'error' present BEFORE and AFTER → not introduced by the action → not a flag.
    rv = classify_reality(
        expected_change="the cart updates",
        verb="click",
        pre_text="Error banner is shown at the top.",
        post_text="Error banner is shown at the top still.",
        critic_success=True,
    )
    assert rv.status != CONTRADICTED


def test_confirmed_via_affirmative_marker():
    rv = classify_reality(
        expected_change="the item is added to the cart",
        verb="click",
        pre_text="Realme Buds product page.",
        post_text="Added to cart. Go to cart to checkout.",
        critic_success=True,
    )
    assert rv.status == CONFIRMED


def test_confirmed_via_predicted_navigation():
    rv = classify_reality(
        expected_change="the page will redirect to the product detail page",
        verb="click",
        pre_url="https://shop.example/search?q=buds",
        post_url="https://shop.example/product/realme-buds-3",
        pre_text="search results",
        post_text="Realme Buds Wireless 3 — product details",
        critic_success=True,
    )
    assert rv.status == CONFIRMED


def test_confirmed_via_predicted_tokens():
    rv = classify_reality(
        expected_change="a blue settings panel opens showing the widget options",
        verb="click",
        pre_text="dashboard home",
        post_text="blue settings panel widget options visible",
        critic_success=True,
    )
    assert rv.status == CONFIRMED


def test_unclear_when_change_but_no_match():
    rv = classify_reality(
        expected_change="the settings panel opens",
        verb="click",
        pre_text="home page",
        post_text="completely unrelated content appeared here",
        critic_success=True,
    )
    assert rv.status == UNCLEAR


def test_null_when_no_prediction_and_no_change():
    rv = classify_reality(
        expected_change="",
        verb="click",
        pre_text="same page",
        post_text="same page",
        critic_success=False,
    )
    assert rv.status == NULL


def test_non_reconcilable_verb_is_null():
    # scroll / wait / goto are handled elsewhere — reconciliation is for actions
    # with a predicted UI effect (click/type/press_enter).
    rv = classify_reality(expected_change="more items load", verb="scroll",
                          post_text="error something went wrong", critic_success=True)
    assert rv.status == NULL


def test_unexpected_auth_redirect_is_contradiction():
    rv = classify_reality(
        expected_change="the item is added to the cart",
        verb="click",
        pre_url="https://shop.example/product/123",
        post_url="https://shop.example/account/login?next=/cart",
        pre_text="product page", post_text="Please sign in to continue.",
        critic_success=True,
    )
    assert rv.status == CONTRADICTED


def test_expected_auth_redirect_is_not_contradiction():
    # If the task/prediction IS to log in, landing on a login page is correct.
    rv = classify_reality(
        expected_change="clicking Log in opens the Google sign in page",
        verb="click",
        pre_url="https://hashnode.com/",
        post_url="https://accounts.google.com/signin",
        pre_text="hashnode home", post_text="Sign in with Google",
        critic_success=True,
    )
    assert rv.status != CONTRADICTED


# ═══════════════════════════════════════════════════════════════════════════════
#  Guidance bus — reality outranks every other transient directive
# ═══════════════════════════════════════════════════════════════════════════════

def _count_blocks(s: str) -> int:
    return s.count("═══ PRIORITY GUIDANCE")


def test_reality_note_is_top_priority():
    state = {
        "reality_note": "Predicted add-to-cart but the page shows out of stock.",
        "goal_complete_hint": "GOAL COMPLETE — finish now.",
        "consecutive_identical_actions": 5,
        "correction_context": "scroll down",
        "recovery_advice": "dismiss popup",
    }
    out = build_guidance(state)
    assert _count_blocks(out) == 1
    assert "SCREEN-REALITY MISMATCH" in out
    assert "out of stock" in out
    assert "finish now" not in out.lower()
    assert "scroll down" not in out


def test_no_reality_note_falls_through_to_existing_priorities():
    # Empty reality_note must not change the established win>rep>esc>rec order.
    out = build_guidance({"reality_note": "", "goal_complete_hint": "verify and finish"})
    assert "SCREEN-REALITY MISMATCH" not in out
    assert "verify and finish" in out


def test_clear_transient_wipes_reality_fields():
    d = clear_transient()
    assert d["reality_note"] == ""
    assert d["reality_status"] == ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature flags — the anti-regression switchboard
# ═══════════════════════════════════════════════════════════════════════════════

def test_flags_default_on(monkeypatch):
    for k in ("V29_ENABLED", "V29_REALITY", "V29_REALITY_LLM",
              "V29_CLARITY_CONSENSUS", "V29_STAGNATION"):
        monkeypatch.delenv(k, raising=False)
    assert ff.v29_enabled() is True
    assert ff.reality_enabled() is True
    assert ff.reality_llm_enabled() is True
    assert ff.clarity_consensus_enabled() is True


def test_master_kill_switch_reverts_everything(monkeypatch):
    monkeypatch.setenv("V29_ENABLED", "0")
    assert ff.v29_enabled() is False
    assert ff.reality_enabled() is False
    assert ff.reality_llm_enabled() is False
    assert ff.clarity_consensus_enabled() is False
    assert ff.stagnation_enabled() is False


def test_per_feature_switch_isolated(monkeypatch):
    monkeypatch.delenv("V29_ENABLED", raising=False)
    monkeypatch.setenv("V29_REALITY", "0")
    assert ff.v29_enabled() is True          # master still on
    assert ff.reality_enabled() is False     # this organ off
    assert ff.reality_llm_enabled() is False  # depends on reality
    assert ff.stagnation_enabled() is True   # siblings unaffected


# ═══════════════════════════════════════════════════════════════════════════════
#  Wiring guards — the new layer is actually integrated (cheap regression alarms)
# ═══════════════════════════════════════════════════════════════════════════════

def test_overwatch_wires_reality_monitor():
    src = (Path(__file__).parent / "overwatch.py").read_text()
    assert "classify_reality" in src
    assert "reality_enabled" in src


def test_brain_state_has_reality_fields():
    src = (Path(__file__).parent / "brain_state.py").read_text()
    assert "reality_status" in src
    assert "reality_note" in src


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
