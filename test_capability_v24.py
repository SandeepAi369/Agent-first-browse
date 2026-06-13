"""Unit tests for V24 — Agentic Capability Gate + dual-mode (free/premium).

Guarantees under test (mirror STRATEGY.md invariants):
  - Mode detection: auto→premium iff PREMIUM_API_KEY; explicit premium/free force.
  - Premium pipeline: one key/model serves text+vision; separate vision model
    honored; missing model → empty (caller falls back to free).
  - Registry in premium mode skips the probe (trusted) and the worker chain is
    the single premium model.
  - Premium misconfigured → safe fallback to free.
  - Role separation: worker chain is tier ≤ WORKER_MAX_TIER, never empty.
  - Capability gate: incapable combos excluded; if that would EMPTY a pipeline,
    allowlisted-alive models are restored (never empty — invariant #2).

Run: .venv/bin/python -m pytest test_capability_v24.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import model_registry as MR
from model_registry import (
    AGENTIC_TEXT_ALLOWLIST,
    WORKER_MAX_TIER,
    ModelClient,
    ModelRegistry,
    _build_premium_pipeline,
    get_agent_mode,
    get_model_tier,
)


def _mc(name: str, provider: str, pipeline: str = "text") -> ModelClient:
    return ModelClient(name=name, client=None, provider=provider, pipeline=pipeline)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("PREMIUM_API_KEY", "PREMIUM_API_KEYS", "PREMIUM_MODEL",
              "PREMIUM_VISION_MODEL", "PREMIUM_BASE_URL", "PREMIUM_PROVIDER", "AGENT_MODE"):
        monkeypatch.delenv(k, raising=False)
    yield
    ModelRegistry.reset()


# ═══════════════════════════════════════════════════════════════════════════════
#  Mode detection
# ═══════════════════════════════════════════════════════════════════════════════

def test_auto_is_free_without_premium_key(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "auto")
    assert get_agent_mode() == "free"


def test_auto_is_premium_with_premium_key(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "auto")
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")
    assert get_agent_mode() == "premium"


def test_explicit_free_overrides_premium_key(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "free")
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")
    assert get_agent_mode() == "free"


def test_explicit_premium_forces(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "premium")
    assert get_agent_mode() == "premium"


# ═══════════════════════════════════════════════════════════════════════════════
#  Premium pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def test_premium_one_key_serves_text_and_vision(monkeypatch):
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")
    monkeypatch.setenv("PREMIUM_MODEL", "gpt-5")
    text, vision = _build_premium_pipeline()
    assert len(text) == 1 and len(vision) == 1
    assert text[0].pipeline == "text" and vision[0].pipeline == "vision"
    assert text[0].provider == "premium"
    assert "gpt-5" in text[0].name and "gpt-5" in vision[0].name


def test_premium_separate_vision_model(monkeypatch):
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")
    monkeypatch.setenv("PREMIUM_MODEL", "claude-opus-4")
    monkeypatch.setenv("PREMIUM_VISION_MODEL", "claude-vision")
    text, vision = _build_premium_pipeline()
    assert "claude-opus-4" in text[0].name
    assert "claude-vision" in vision[0].name


def test_premium_missing_model_returns_empty(monkeypatch):
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")  # no PREMIUM_MODEL
    text, vision = _build_premium_pipeline()
    assert text == [] and vision == []


def test_premium_multiple_keys(monkeypatch):
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-a,sk-b")
    monkeypatch.setenv("PREMIUM_MODEL", "gpt-5")
    text, _ = _build_premium_pipeline()
    assert len(text) == 2  # one instance per key for headroom


def test_registry_premium_skips_probe(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "premium")
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")
    monkeypatch.setenv("PREMIUM_MODEL", "gpt-5")
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    assert r.mode == "premium"
    assert r._probed is True                       # trusted — never probes
    assert r.get_worker_chain_names() == ["premium-text:gpt-5:0"]
    assert r.get_vision_chain_names() == ["premium-vision:gpt-5:0"]


def test_registry_premium_misconfigured_falls_back_to_free(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "premium")
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")   # but NO PREMIUM_MODEL
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    assert r.mode == "free"                          # safe fallback
    assert r._probed is False                        # free mode will probe


# ═══════════════════════════════════════════════════════════════════════════════
#  Role separation (worker chain)
# ═══════════════════════════════════════════════════════════════════════════════

def test_worker_chain_only_top_tier(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "free")
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    r._text_pipeline = [
        _mc("groq:openai/gpt-oss-120b:0", "groq"),       # tier 0
        _mc("nvidia-text:openai/gpt-oss-20b:0", "nvidia"),  # tier 1
        _mc("cerebras:qwen-3-235b-a22b:0", "cerebras"),  # tier 2 — excluded from worker
        _mc("x:glm-5.1:0", "x"),                         # tier 2 — excluded
    ]
    wc = r.get_worker_chain_names()
    assert all(get_model_tier(n) <= WORKER_MAX_TIER for n in wc), wc
    assert "groq:openai/gpt-oss-120b:0" in wc
    assert "cerebras:qwen-3-235b-a22b:0" not in wc


def test_worker_chain_never_empty(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "free")
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    r._text_pipeline = [_mc("cerebras:qwen-3-235b-a22b:0", "cerebras")]  # only tier 2
    # No tier-0/1 model → worker must fall back to the full chain, not empty.
    assert r.get_worker_chain_names() == ["cerebras:qwen-3-235b-a22b:0"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Capability gate
# ═══════════════════════════════════════════════════════════════════════════════

def test_gate_excludes_incapable(monkeypatch):
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    pipe = [
        _mc("groq:openai/gpt-oss-120b:0", "groq"),
        _mc("nvidia-text:google/gemma-4-31b-it:0", "nvidia"),
    ]
    incapable = {("nvidia", "gemma-4-31b-it", "text")}   # gemma-on-NVIDIA can't structure
    out = [m.name for m in r._apply_capability_gate(pipe, set(), incapable, "TEXT")]
    assert "groq:openai/gpt-oss-120b:0" in out
    assert "nvidia-text:google/gemma-4-31b-it:0" not in out


def test_gate_excludes_dead(monkeypatch):
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    pipe = [
        _mc("nvidia-text:openai/gpt-oss-120b:0", "nvidia"),
        _mc("nvidia-vision:google/gemma-4-32b-it:0", "nvidia", "vision"),
    ]
    dead = {("nvidia", "gemma-4-32b-it", "vision")}
    out = [m.name for m in r._apply_capability_gate(pipe, dead, set(), "VISION")]
    assert "nvidia-vision:google/gemma-4-32b-it:0" not in out
    assert "nvidia-text:openai/gpt-oss-120b:0" in out


def test_gate_safety_floor_never_empties(monkeypatch):
    # If the gate would remove EVERY model, restore allowlisted-alive ones.
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    pipe = [_mc("groq:openai/gpt-oss-120b:0", "groq")]  # allowlisted
    incapable = {("groq", "gpt-oss-120b", "text")}      # (falsely) flagged incapable
    out = r._apply_capability_gate(pipe, set(), incapable, "TEXT")
    assert len(out) == 1, "pipeline must never be emptied by the gate"


def test_allowlist_has_proven_models():
    for m in ("gpt-oss-120b", "gpt-oss-20b", "gemma-4-31b-it"):
        assert m in AGENTIC_TEXT_ALLOWLIST


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
