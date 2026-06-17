"""Phase B tests — WebDreamer revival (predictive simulation, Clarity-gated).

The simulation itself is LLM-driven (not unit-testable offline), so these pin the
PURE decision/gate logic + the wiring, which is what governs WHEN it fires and
WHETHER it overrides — the parts that must never burn compute on obvious steps.

Run: .venv/bin/python -m pytest test_phase_b_v29.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import feature_flags as ff
from web_dreamer import (should_invoke_dreamer, should_override_with_dreamer,
                         extract_situation, situational_adjustment, adjusted_score,
                         select_best_evaluation, CandidateAction, CandidateEvaluation)


def _ev(verb, score, eid=None):
    return CandidateEvaluation(candidate=CandidateAction(action_type=verb, element_id=eid),
                               score=score)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cost gate — only fire on high-stakes / stuck / confused steps
# ═══════════════════════════════════════════════════════════════════════════════

def test_gate_fires_on_irreversible():
    assert should_invoke_dreamer(element_count=3, action_risk_level="IRREVERSIBLE",
                                 consecutive_no_progress=0, step_number=1, same_url_streak=0)


def test_gate_fires_when_stuck_or_confused():
    assert should_invoke_dreamer(5, "REVERSIBLE", consecutive_no_progress=2,
                                 step_number=4, same_url_streak=0)            # stuck
    assert should_invoke_dreamer(10, "REVERSIBLE", consecutive_no_progress=0,
                                 step_number=4, same_url_streak=3)            # confused
    assert should_invoke_dreamer(20, "CAUTIOUS", consecutive_no_progress=0,
                                 step_number=4, same_url_streak=0)            # complex+cautious


def test_gate_skips_obvious_steps():
    # simple page, low risk, early, not stuck → DO NOT burn compute
    assert should_invoke_dreamer(3, "REVERSIBLE", consecutive_no_progress=0,
                                 step_number=1, same_url_streak=0) is False


# ═══════════════════════════════════════════════════════════════════════════════
#  Override decision — replace only a confident, genuinely DIFFERENT action
# ═══════════════════════════════════════════════════════════════════════════════

def test_override_when_confident_and_different():
    assert should_override_with_dreamer(0.8, "click", "e9", "click", "e3") is True
    assert should_override_with_dreamer(0.8, "type", "e3", "click", "e3") is True


def test_no_override_when_same_action_just_confirmed():
    # dreamer's best == the worker's pick → confirm, don't churn
    assert should_override_with_dreamer(0.95, "click", "e3", "click", "e3") is False


def test_no_override_when_low_score():
    assert should_override_with_dreamer(0.4, "click", "e9", "click", "e3") is False


# ═══════════════════════════════════════════════════════════════════════════════
#  Flag + wiring
# ═══════════════════════════════════════════════════════════════════════════════

def test_webdreamer_flag(monkeypatch):
    monkeypatch.delenv("V29_ENABLED", raising=False)
    monkeypatch.delenv("V29_WEBDREAMER", raising=False)
    assert ff.webdreamer_enabled() is True
    monkeypatch.setenv("V29_WEBDREAMER", "0")
    assert ff.webdreamer_enabled() is False
    monkeypatch.setenv("V29_WEBDREAMER", "1")
    monkeypatch.setenv("V29_ENABLED", "0")
    assert ff.webdreamer_enabled() is False   # master kill-switch wins


def test_webdreamer_wired():
    bw = (Path(__file__).parent / "workers" / "base_worker.py").read_text()
    assert "plan_and_select" in bw and "should_invoke_dreamer" in bw
    assert "should_override_with_dreamer" in bw
    assert "clarity_sig.uncertain" in bw          # Clarity-gated
    assert "dreamer=None" in bw                    # plumbed param
    bg = (Path(__file__).parent / "brain_graph.py").read_text()
    assert bg.count("dreamer=_DREAMER,") == 3      # all three worker nodes
    bs = (Path(__file__).parent / "brain_state.py").read_text()
    assert "webdreamer_runs" in bs


# ═══════════════════════════════════════════════════════════════════════════════
#  Situational tuning — the 4 universal situations (real selection tests)
# ═══════════════════════════════════════════════════════════════════════════════

def test_extract_situation_reads_state_signals():
    s = extract_situation({"state_change_score": 0.6, "same_url_streak": 1})
    assert s["reveal"] is True
    assert extract_situation({"state_change_score": 0.1, "same_url_streak": 1})["reveal"] is False
    assert extract_situation({"same_url_streak": 4})["stuckness"] == 1.0
    assert extract_situation({"scroll_stuck_streak": 2})["scroll_productive"] == 0.0
    assert extract_situation({"scroll_stuck_streak": 0})["scroll_productive"] == 1.0


def test_situation_1_REVEAL_penalizes_scroll_rewards_engage():
    # last click revealed a toggle (big DOM change, static URL) → evaluate it, don't scroll away
    situ = extract_situation({"state_change_score": 0.6, "same_url_streak": 1})
    assert situational_adjustment("scroll", situ) < 0
    assert situational_adjustment("goto", situ) < 0
    assert situational_adjustment("click", situ) > 0
    # selection flips from scroll → click on a tie
    evals = [_ev("scroll", 0.5), _ev("click", 0.5, "e1")]
    assert select_best_evaluation(evals, None).candidate.action_type == "scroll"   # baseline
    assert select_best_evaluation(evals, situ).candidate.action_type == "click"     # tuned


def test_situation_2_INFINITE_SCROLL_not_penalized():
    # productive scroll (nothing stale) → scroll keeps full value (feed not broken)
    situ = extract_situation({"scroll_stuck_streak": 0, "state_change_score": 0.0})
    assert situational_adjustment("scroll", situ) == 0.0          # NO penalty
    evals = [_ev("scroll", 0.6), _ev("click", 0.5, "e1")]
    assert select_best_evaluation(evals, situ).candidate.action_type == "scroll"


def test_situation_3_SPA_reveal_but_not_stuck_does_not_elevate_goto():
    # JS-rendered content (reveal) but the agent is flowing → goto NOT made reckless
    situ = extract_situation({"state_change_score": 0.6, "same_url_streak": 1})
    assert situational_adjustment("goto", situ) < 0               # reveal penalty dominates
    evals = [_ev("goto", 0.5), _ev("click", 0.5, "e1")]
    assert select_best_evaluation(evals, situ).candidate.action_type == "click"


def test_situation_4_DESPERATION_GOTO_when_stuck():
    # stuck on a static URL, no reveal → exploration (goto) becomes rational
    situ = extract_situation({"same_url_streak": 4, "state_change_score": 0.0})
    assert situational_adjustment("goto", situ) > 0.10
    evals = [_ev("click", 0.5, "e1"), _ev("goto", 0.5)]
    assert select_best_evaluation(evals, situ).candidate.action_type == "goto"


def test_adjusted_score_baseline_is_identity_when_no_situation():
    assert adjusted_score(0.5, "scroll", None) == 0.5
    assert adjusted_score(0.5, "goto", None) == 0.5
    # selection with no situation = plain argmax (vacuum baseline)
    evals = [_ev("scroll", 0.7), _ev("goto", 0.5)]
    assert select_best_evaluation(evals, None).candidate.action_type == "scroll"


def test_situational_flag(monkeypatch):
    monkeypatch.delenv("V29_ENABLED", raising=False)
    monkeypatch.delenv("V29_WEBDREAMER", raising=False)
    monkeypatch.delenv("V29_WEBDREAMER_SITUATIONAL", raising=False)
    assert ff.webdreamer_situational_enabled() is True
    monkeypatch.setenv("V29_WEBDREAMER_SITUATIONAL", "0")
    assert ff.webdreamer_situational_enabled() is False          # instant fallback
    monkeypatch.setenv("V29_WEBDREAMER_SITUATIONAL", "1")
    monkeypatch.setenv("V29_WEBDREAMER", "0")
    assert ff.webdreamer_situational_enabled() is False          # parent gate
    monkeypatch.setenv("V29_WEBDREAMER", "1")
    monkeypatch.setenv("V29_ENABLED", "0")
    assert ff.webdreamer_situational_enabled() is False          # master gate


def test_situational_wiring():
    wd = (Path(__file__).parent / "web_dreamer.py").read_text()
    assert "def situational_adjustment" in wd and "def select_best_evaluation" in wd
    assert "situation: dict | None = None" in wd                 # plan_and_select param
    bw = (Path(__file__).parent / "workers" / "base_worker.py").read_text()
    assert "situation=state" in bw                                # state signals passed through


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
