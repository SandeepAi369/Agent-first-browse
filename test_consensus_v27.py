"""Unit tests for V27/P2 — multi-model consensus + abstention.

Guarantees:
  - confidence-weighted (CISC) vote picks the right winner;
  - agreement / mean_conf math;
  - abstain when voters split, proceed when they agree;
  - distinct-base-model selection (the independent voters);
  - the ensemble sampler polls each model once and tolerates failures.

Run: .venv/bin/python -m pytest test_consensus_v27.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import consensus as C
from consensus import (
    weighted_vote, should_abstain, canonical_action_key,
    distinct_base_model_clients, count_distinct_base_models, sample_ensemble,
)


def _s(verb, eid, text="", conf=0.7):
    return {"verb": verb, "element_id": eid, "text": text, "confidence": conf}


# ── voting ──

def test_unanimous_proceeds():
    v = weighted_vote([_s("click", "e5", conf=0.9)] * 3)
    assert v.verb == "click" and v.element_id == "e5"
    assert v.agreement == 1.0
    assert should_abstain(v) is False


def test_majority_two_of_three_proceeds():
    v = weighted_vote([_s("click", "e5", conf=0.8),
                       _s("click", "e5", conf=0.8),
                       _s("click", "e9", conf=0.8)])
    assert v.element_id == "e5"
    # agreement = 1.6 / 2.4 = 0.667 ≥ 0.6 → proceed
    assert round(v.agreement, 2) == 0.67
    assert should_abstain(v) is False


def test_three_way_split_abstains():
    v = weighted_vote([_s("click", "e1"), _s("click", "e2"), _s("click", "e3")])
    assert v.n_unique == 3
    # agreement = 0.7 / 2.1 = 0.333 < 0.6 → abstain
    assert round(v.agreement, 2) == 0.33
    assert should_abstain(v) is True


def test_confidence_weighting_overrides_count():
    # Two low-conf votes for e1 vs one high-conf vote for e2.
    v = weighted_vote([_s("click", "e1", conf=0.3),
                       _s("click", "e1", conf=0.3),
                       _s("click", "e2", conf=0.95)])
    # weights: e1=0.6, e2=0.95 → CISC winner is e2 despite fewer votes
    assert v.element_id == "e2"


def test_low_confidence_winner_abstains():
    v = weighted_vote([_s("click", "e5", conf=0.3)] * 3)
    assert v.agreement == 1.0          # unanimous...
    assert v.mean_conf == 0.3          # ...but jointly unconfident
    assert should_abstain(v) is True   # mean_conf < THETA_CONF


def test_canonical_key_normalizes():
    assert canonical_action_key("Click", "e5", "Hello World ") == \
           canonical_action_key("click", "e5", "Hello World")
    assert canonical_action_key("click", "e5", "a") != \
           canonical_action_key("click", "e6", "a")


def test_empty_samples():
    assert weighted_vote([]) is None
    assert should_abstain(None) is True


# ── distinct base-model selection ──

class _MC:
    def __init__(self, name):
        self.name = name


def test_distinct_base_models():
    chain = [_MC("groq:openai/gpt-oss-120b:0"), _MC("groq:openai/gpt-oss-120b:1"),
             _MC("nvidia-text:openai/gpt-oss-120b:0"),
             _MC("gemini-text:gemma-4-31b-it:0"), _MC("nvidia-text:openai/gpt-oss-20b:0")]
    assert count_distinct_base_models(chain) == 3  # gpt-oss-120b, gemma-4, gpt-oss-20b
    picks = distinct_base_model_clients(chain, n=3)
    bases = {p.name.split(":")[1] for p in picks}
    assert len(bases) == 3
    # exclude the model we already have a vote from
    picks2 = distinct_base_model_clients(chain, n=2,
                                         exclude_base="openai/gpt-oss-120b")
    bases2 = {p.name.split(":")[1] for p in picks2}
    assert "openai/gpt-oss-120b" not in bases2


# ── ensemble sampler ──

def test_sample_ensemble_polls_each_and_tolerates_failure():
    calls = []

    async def fake_invoke(chain, messages, schema, breaker, health_tracker=None):
        name = chain[0].name
        calls.append(name)
        if "bad" in name:
            raise RuntimeError("voter down")
        return (f"decision_for_{name}", name)

    models = [_MC("groq:good/m1:0"), _MC("nvidia:bad/m2:0"), _MC("x:good/m3:0")]
    out = asyncio.run(sample_ensemble(models, ["msg"], object,
                                      fake_invoke, None, None))
    assert len(calls) == 3              # polled all three
    assert len(out) == 2               # the failing voter dropped out


# ── dynamic cascade consensus (V28: time-vs-accuracy) ──

class _Dec:
    """Minimal WorkerAction-like decision for cascade tests."""
    def __init__(self, action_type, element_id=None, text=None, confidence=0.7,
                 url=None, x=None, y=None):
        self.action_type = action_type; self.element_id = element_id
        self.text = text; self.confidence = confidence
        self.url = url; self.x = x; self.y = y; self.missing_data = ""; self.rationale = ""


def test_structural_ok():
    from consensus import structural_ok
    assert structural_ok(_Dec("click", "e5"), {"e5": {}}) is True
    assert structural_ok(_Dec("click", "e9"), {"e5": {}}) is False   # id not in map
    assert structural_ok(_Dec("click", None), {}) is False           # no target
    assert structural_ok(_Dec("click", None, x=10, y=20), {}) is True  # coords ok
    assert structural_ok(_Dec("goto", url="https://x.com"), {}) is True
    assert structural_ok(_Dec("goto"), {}) is False
    assert structural_ok(_Dec("done"), {}) is True


def _cascade(primary, voters_decisions, force=False, smap=None):
    """Drive cascade_consensus with a fake invoke that returns scripted votes."""
    from consensus import cascade_consensus
    seq = list(voters_decisions)
    calls = {"n": 0}

    async def fake_invoke(chain, messages, schema, breaker, health_tracker=None):
        calls["n"] += 1
        d = seq.pop(0)
        return (d, chain[0].name)

    chain = [_MC("groq:openai/gpt-oss-120b:0"),
             _MC("nvidia:openai/gpt-oss-20b:0"),
             _MC("gemini:gemma-4-31b-it:0")]
    res = asyncio.run(cascade_consensus(
        primary_decision=primary, primary_model="groq:openai/gpt-oss-120b:0",
        messages=["m"], schema=object, invoke_fn=fake_invoke, chain=chain,
        breaker=None, health_tracker=None, selector_map=smap or {"e5": {}, "e6": {}},
        force_escalate=force))
    return res, calls["n"]


def test_cascade_tier1_confident_executes_immediately():
    res, n = _cascade(_Dec("click", "e5", confidence=0.95), [])
    assert res.path == "primary_confident"
    assert res.extra_calls == 0 and n == 0     # NO secondary polled — fast path
    assert res.abstain is False


def test_cascade_low_confidence_polls_secondary_and_agrees():
    # primary low conf; secondary agrees → execute after 1 extra call
    res, n = _cascade(_Dec("click", "e5", confidence=0.4),
                      [_Dec("click", "e5", confidence=0.9)])
    assert res.path == "secondary_agree"
    assert res.extra_calls == 1 and n == 1
    assert res.abstain is False


def test_cascade_disagreement_escalates_to_tertiary_and_votes():
    # primary low; secondary disagrees; tertiary breaks the tie toward e6
    res, n = _cascade(_Dec("click", "e5", confidence=0.4),
                      [_Dec("click", "e6", confidence=0.9),
                       _Dec("click", "e6", confidence=0.9)])
    assert res.extra_calls == 2 and n == 2
    assert res.path == "voted"
    assert res.decision.element_id == "e6"     # tertiary tipped the vote


def test_cascade_three_way_split_abstains():
    res, n = _cascade(_Dec("click", "e5", confidence=0.4),
                      [_Dec("click", "e6", confidence=0.8),
                       _Dec("wait", confidence=0.8)])
    assert res.abstain is True and res.path == "abstain"


def test_cascade_force_escalate_overrides_high_confidence():
    # Even a 0.95 primary must get a second opinion when the agent is stuck.
    res, n = _cascade(_Dec("click", "e5", confidence=0.95),
                      [_Dec("click", "e5", confidence=0.9)], force=True)
    assert res.extra_calls == 1 and res.path == "secondary_agree"


def test_cascade_malformed_primary_escalates_despite_confidence():
    # confident but element_id not in the selector map → structurally unsound
    res, n = _cascade(_Dec("click", "e99", confidence=0.95),
                      [_Dec("click", "e5", confidence=0.9)], smap={"e5": {}})
    assert res.extra_calls >= 1                 # did NOT take the fast path


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
