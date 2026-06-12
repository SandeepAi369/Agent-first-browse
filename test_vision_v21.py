"""Unit tests for V21 Vision-on-Demand (the a11y-DOM ⇄ vision "thinker" toggle).

Guarantees under test:
  - a11y DOM is the DEFAULT: no trigger ⇒ no vision consult.
  - The thinker triggers vision only when it should: worker self-flag, the
    ladder's vision rung (force_vision), or an ineffective-action streak.
  - Per-task budget caps consults; 'wait' never consults.
  - apply_vision_verdict overrides ONLY on a confident, concrete verdict, and
    drops stale a11y coords so the chosen element id wins (V19 re-resolves them).
  - Vision unavailable / errored ⇒ a11y decision stands (no regression).
  - consult_vision attaches a real screenshot via the model layer's base64 path,
    and the toggle is per-step (force_vision consumed, reverts to a11y).

Run: .venv/bin/python -m pytest test_vision_v21.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import mcp_tools
import vision_consult
from vision_consult import (
    MAX_VISION_CONSULTS,
    VisionVerdict,
    apply_vision_verdict,
    consult_vision,
    should_consult_vision,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Trigger logic — the thinker decides when to open its eyes
# ═══════════════════════════════════════════════════════════════════════════════

def test_default_is_a11y_no_consult():
    consult, reason = should_consult_vision(
        needs_vision=False, state={}, action_type="click")
    assert consult is False
    assert reason == ""


def test_worker_self_request_triggers():
    consult, reason = should_consult_vision(
        needs_vision=True, state={}, action_type="click")
    assert consult is True
    assert "request" in reason


def test_ladder_vision_rung_triggers():
    consult, reason = should_consult_vision(
        needs_vision=False, state={"force_vision": True}, action_type="click")
    assert consult is True
    assert "ladder" in reason


def test_ineffective_streak_triggers():
    consult, reason = should_consult_vision(
        needs_vision=False, state={"ineffective_streak": 2}, action_type="click")
    assert consult is True
    assert "ineffective" in reason

    # One ineffective action is not yet enough.
    consult, _ = should_consult_vision(
        needs_vision=False, state={"ineffective_streak": 1}, action_type="click")
    assert consult is False


def test_budget_caps_consults():
    consult, reason = should_consult_vision(
        needs_vision=True, state={"vision_consults": MAX_VISION_CONSULTS},
        action_type="click")
    assert consult is False
    assert "budget" in reason


def test_wait_never_consults():
    consult, _ = should_consult_vision(
        needs_vision=True, state={}, action_type="wait")
    assert consult is False


# ═══════════════════════════════════════════════════════════════════════════════
#  Verdict application — vision refines the a11y action only when confident
# ═══════════════════════════════════════════════════════════════════════════════

def _base_action():
    return {"verb": "click", "element_id": "e3", "x": 100.0, "y": 200.0,
            "text": None, "reasoning": "a11y guess"}

def test_confident_verdict_overrides_and_drops_coords():
    v = VisionVerdict(observation="the real Add to Cart is the orange one",
                      action_type="click", element_id="e7", reasoning="orange CTA",
                      confidence=0.9)
    out, overridden = apply_vision_verdict(_base_action(), v)
    assert overridden is True
    assert out["verb"] == "click"
    assert out["element_id"] == "e7"
    assert out["x"] is None and out["y"] is None     # stale coords dropped
    assert out["vision_used"] is True
    assert out["reasoning"].startswith("[vision]")


def test_low_confidence_keeps_a11y():
    v = VisionVerdict(observation="unsure", action_type="click", element_id="e7",
                      reasoning="maybe", confidence=0.3)
    out, overridden = apply_vision_verdict(_base_action(), v)
    assert overridden is False
    assert out["element_id"] == "e3"                  # untouched


def test_none_action_keeps_a11y():
    v = VisionVerdict(observation="screenshot matches the DOM plan",
                      action_type="none", reasoning="nothing to change",
                      confidence=0.95)
    out, overridden = apply_vision_verdict(_base_action(), v)
    assert overridden is False
    assert out["element_id"] == "e3"


def test_null_verdict_keeps_a11y():
    out, overridden = apply_vision_verdict(_base_action(), None)
    assert overridden is False
    assert out == _base_action()


# ═══════════════════════════════════════════════════════════════════════════════
#  consult_vision — screenshot attach + graceful unavailability
# ═══════════════════════════════════════════════════════════════════════════════

def test_consult_unavailable_when_no_vision_chain():
    out = asyncio.run(consult_vision(
        invoke_fn=lambda *a, **k: None, vision_chain=[], breaker=None,
        health_tracker=None, objective="x", question="y", a11y_markdown="z"))
    assert out == (None, "")


def test_consult_attaches_screenshot_and_returns_verdict(monkeypatch):
    # Stub the screenshot (no browser) and the vision model call.
    async def fake_shot(full_page=False):
        return {"ok": True, "base64": "ZmFrZQ==", "error": ""}
    monkeypatch.setattr(mcp_tools, "mcp_screenshot", fake_shot)

    captured = {}
    async def fake_invoke(chain, messages, schema, breaker, base64_image=None, health_tracker=None):
        captured["image"] = base64_image
        captured["schema"] = schema
        return VisionVerdict(observation="seen", action_type="click",
                             element_id="e9", reasoning="r", confidence=0.8), "vlm-1"
    verdict, model = asyncio.run(consult_vision(
        fake_invoke, ["vlm"], None, None,
        objective="add to cart", question="which button?",
        a11y_markdown="- [e9] button: Add to cart"))
    assert verdict is not None and verdict.element_id == "e9"
    assert model == "vlm-1"
    assert captured["image"] == "ZmFrZQ=="          # screenshot actually attached
    assert captured["schema"] is VisionVerdict


def test_consult_screenshot_failure_falls_back(monkeypatch):
    async def fail_shot(full_page=False):
        return {"ok": False, "base64": "", "error": "boom"}
    monkeypatch.setattr(mcp_tools, "mcp_screenshot", fail_shot)
    out = asyncio.run(consult_vision(
        lambda *a, **k: None, ["vlm"], None, None,
        objective="x", question="y", a11y_markdown="z"))
    assert out == (None, "")


def test_consult_model_error_falls_back(monkeypatch):
    async def fake_shot(full_page=False):
        return {"ok": True, "base64": "abc", "error": ""}
    monkeypatch.setattr(mcp_tools, "mcp_screenshot", fake_shot)
    async def boom(*a, **k):
        raise RuntimeError("vision 503")
    verdict, model = asyncio.run(consult_vision(
        boom, ["vlm"], None, None, objective="x", question="y", a11y_markdown="z"))
    assert verdict is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
