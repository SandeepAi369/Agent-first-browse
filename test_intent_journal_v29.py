"""Unit tests for V29 — Atomic Intent Journaling (handoff-amnesia / double-toggle fix).

Covers (pure logic + durable ledger; no browser/network):
  • make_intent / should_journal / signatures.
  • outcome classification: a TIMEOUT is 'uncertain' (the dangerous double-apply
    case), a clean OK is 'confirmed'.
  • same_action: detect re-proposing the unconfirmed prior action.
  • render_hesitation: the strict "do not blindly repeat" handoff warning, with
    extra severity for IRREVERSIBLE actions.
  • durable atomic ledger: persist → read → resolve round-trip.
  • flags + clear_transient + wiring guards (Overwatch write-ahead, base_worker
    hesitation + repeat-guard, BrainState field).

Run: .venv/bin/python -m pytest test_intent_journal_v29.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import feature_flags as ff
import intent_journal as ij
from cognition import clear_transient


def _proposed(verb="click", element_id="e5", text=None, risk="IRREVERSIBLE"):
    return {"verb": verb, "element_id": element_id, "text": text,
            "target_name": "Add to Cart", "risk_level": risk}


# ═══════════════════════════════════════════════════════════════════════════════
#  Intent payload + classification
# ═══════════════════════════════════════════════════════════════════════════════

def test_make_intent_shape():
    e = ij.make_intent(_proposed(), step_number=3)
    assert e["verb"] == "click" and e["element_id"] == "e5"
    assert e["step"] == 4 and e["status"] == "executing"
    assert e["risk_level"] == "IRREVERSIBLE"
    assert e["signature"] == ij.action_signature("click", "e5", None)
    assert e["ts"] > 0 and e["iso"]


def test_should_journal_only_side_effecting_verbs():
    assert ij.should_journal("click") and ij.should_journal("type")
    assert ij.should_journal("press_enter")
    assert not ij.should_journal("scroll")
    assert not ij.should_journal("wait")
    assert not ij.should_journal("goto")


def test_timeout_is_uncertain_ok_is_confirmed():
    assert ij.is_uncertain_outcome("→ CLICK INEFFECTIVE: Click timed out (30s)")
    assert ij.is_uncertain_outcome("→ CRASHED: boom")
    assert not ij.is_uncertain_outcome("→ OK (click via cdp, DOM changed)")
    assert ij.classify_status("→ OK (click via cdp)") == "confirmed"
    assert ij.classify_status("→ CLICK INEFFECTIVE: Click timed out (30s)") == "uncertain"
    assert ij.classify_status("→ CLICK INEFFECTIVE: nothing happened") == "executed"


def test_same_action_detects_repeat():
    e = ij.make_intent(_proposed(element_id="e5", text="x"), 0)
    assert ij.same_action(e, {"verb": "click", "element_id": "e5", "text": "x"})
    assert not ij.same_action(e, {"verb": "click", "element_id": "e6", "text": "x"})
    assert not ij.same_action(e, {"verb": "type", "element_id": "e5", "text": "x"})
    assert not ij.same_action(None, {"verb": "click"})


def test_hesitation_is_universal_situational_rule():
    block = ij.render_hesitation(ij.make_intent(_proposed(risk="IRREVERSIBLE"), 0))
    low = block.lower()
    assert "pending-action ledger" in low
    assert "do not blindly repeat" in low      # the core, always-present rule
    assert "universal rule" in low             # situational, not a taxonomy
    assert "live dom" in low                    # decide from the actual screen
    assert ij.render_hesitation(None) == ""


def test_hesitation_covers_unknown_foreign_control_without_keywords():
    # No toggle word, non-English label, plain REVERSIBLE → still the full universal
    # rule. Proves the protection is NOT dependent on the keyword lists.
    e = ij.make_intent({"verb": "click", "element_id": "e9",
                        "target_name": "Confirmer l'opération 42",
                        "risk_level": "REVERSIBLE"}, 0)
    assert e["hazard"] == "state-change"        # generic safe fallback
    low = ij.render_hesitation(e).lower()
    assert "universal rule" in low and "live dom" in low
    assert "do not blindly repeat" in low


# ── Hazard classification: a TOGGLE is the worst double-apply case, and it is
#    usually classified REVERSIBLE — so it must be caught on its own merits. ──

def test_toggle_detected_by_kind_even_when_reversible():
    assert ij.is_toggle_like(element_kind="checkbox")
    assert ij.is_toggle_like(target_name="Remember me")
    assert ij.is_toggle_like(target_name="Dark mode", text="toggle")
    assert not ij.is_toggle_like(target_name="Submit order", element_kind="button")


def test_hazard_class_priorities():
    # a bare checkbox classified REVERSIBLE is still a TOGGLE hazard
    assert ij.hazard_class("click", "Accept cookies", "", "REVERSIBLE", "checkbox") == "toggle"
    # star/follow are toggles even though the risk label is reversible
    assert ij.hazard_class("click", "Star this repo", "", "REVERSIBLE") == "toggle"
    # a true commit is irreversible
    assert ij.hazard_class("click", "Place order", "", "IRREVERSIBLE") == "irreversible"
    # a plain mutating click is state-change
    assert ij.hazard_class("type", "Quantity field", "3", "CAUTIOUS") == "state-change"


def test_toggle_hint_is_soft_and_keeps_universal_rule():
    # Detected toggle → the flip-back hint appears, but ONLY as a soft heuristic;
    # the universal situational rule is still the backbone.
    e = ij.make_intent({"verb": "click", "element_id": "e7", "target_name": "Email notifications",
                        "risk_level": "REVERSIBLE"}, 0, element_kind="switch")
    assert e["hazard"] == "toggle"
    low = ij.render_hesitation(e).lower()
    assert "invert" in low                      # flip-back reasoning, as an example
    assert "heuristic hint" in low              # explicitly non-authoritative
    assert "verify yourself" in low
    assert "universal rule" in low              # the universal rule remains


# ═══════════════════════════════════════════════════════════════════════════════
#  Durable atomic ledger
# ═══════════════════════════════════════════════════════════════════════════════

def test_durable_ledger_roundtrip(tmp_path):
    p = str(tmp_path / "intent.json")
    assert ij.read_pending(p) is None              # nothing yet
    entry = ij.make_intent(_proposed(), 0)
    ij.persist_intent(entry, p)                    # write-ahead
    got = ij.read_pending(p)
    assert got is not None and got["signature"] == entry["signature"]
    ij.resolve_intent(p)                           # confirmed → clear
    assert ij.read_pending(p) is None


def test_durable_ledger_never_raises_on_bad_path():
    # Best-effort I/O: a bogus path must not raise into the step.
    ij.persist_intent({"x": 1}, "/nonexistent_dir_zzz/ııı/intent.json")
    assert ij.read_pending("/nonexistent_dir_zzz/ııı/intent.json") is None
    ij.resolve_intent("/nonexistent_dir_zzz/ııı/intent.json")  # no raise


# ═══════════════════════════════════════════════════════════════════════════════
#  Flags + clear + wiring guards
# ═══════════════════════════════════════════════════════════════════════════════

def test_intent_journal_flag(monkeypatch):
    monkeypatch.delenv("V29_ENABLED", raising=False)
    monkeypatch.delenv("V29_INTENT_JOURNAL", raising=False)
    assert ff.intent_journal_enabled() is True
    monkeypatch.setenv("V29_INTENT_JOURNAL", "0")
    assert ff.intent_journal_enabled() is False
    monkeypatch.setenv("V29_INTENT_JOURNAL", "1")
    monkeypatch.setenv("V29_ENABLED", "0")
    assert ff.intent_journal_enabled() is False   # master kill-switch wins


def test_clear_transient_resolves_journal():
    assert clear_transient()["last_attempted_action"] is None


def test_wiring_guards():
    ow = (Path(__file__).parent / "overwatch.py").read_text()
    assert "persist_intent" in ow and "Intent journaled" in ow
    assert "resolve_intent" in ow                       # cleared on verified success
    bw = (Path(__file__).parent / "workers" / "base_worker.py").read_text()
    assert "render_hesitation" in bw and "repeating_uncertain" in bw
    bs = (Path(__file__).parent / "brain_state.py").read_text()
    assert "last_attempted_action" in bs


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
