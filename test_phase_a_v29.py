"""Phase A tests — DOM Diffing (state-change) + Hybrid Primitives / clean feedback.

Pure logic + monkeypatched async handlers — no browser, no network.
Run: .venv/bin/python -m pytest test_phase_a_v29.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import feature_flags as ff
import mcp_tools
from dom_diff import signal_vector_diff, state_change_score, progress_phrase
from action_feedback import (FailureClass, classify_failure, render_failure,
                             render_success)


def _sig(**kw):
    base = {"allCount": 100, "childCount": 5, "dialogs": 0, "expanded": 0,
            "interactives": 20, "activeTag": "BODY", "scrollH": 2000, "path": "/x"}
    base.update(kw)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
#  DOM Diffing — the subtle-change detector (the stagnation-loop fix)
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_change_is_not_meaningful():
    d = signal_vector_diff(_sig(), _sig())
    assert d["meaningful"] is False and d["changed"] == 0


def test_overlay_appearing_is_meaningful_even_with_no_node_count_change():
    # the exact failure case: a dialog opens but the interactive set barely moves
    d = signal_vector_diff(_sig(dialogs=0), _sig(dialogs=1))
    assert d["meaningful"] is True and d["dialogs_delta"] == 1
    assert "an overlay/dialog appeared" == progress_phrase(d)


def test_overlay_closing_detected():
    d = signal_vector_diff(_sig(dialogs=1), _sig(dialogs=0))
    assert d["meaningful"] and d["dialogs_delta"] == -1
    assert "closed" in progress_phrase(d)


def test_aria_expanded_toggle_is_meaningful():
    d = signal_vector_diff(_sig(expanded=0), _sig(expanded=1))
    assert d["meaningful"] and "expand" in progress_phrase(d).lower()


def test_route_change_is_meaningful():
    d = signal_vector_diff(_sig(path="/a"), _sig(path="/b"))
    assert d["meaningful"] and "path" in d["keys"]


def test_single_jittery_signal_is_not_meaningful():
    # allCount +1 is within tolerance; one tiny change alone ≠ meaningful
    d = signal_vector_diff(_sig(allCount=100), _sig(allCount=101))
    assert d["meaningful"] is False


def test_two_signals_moving_is_meaningful():
    d = signal_vector_diff(_sig(childCount=5, interactives=20),
                           _sig(childCount=7, interactives=25))
    assert d["meaningful"] is True and d["changed"] >= 2


def test_none_inputs_are_safe():
    assert signal_vector_diff(None, _sig())["meaningful"] is False
    assert signal_vector_diff(_sig(), None)["meaningful"] is False


# ── state_change_score: unified [0,1] value signal ──

def test_score_is_bounded_and_monotonic():
    assert state_change_score() == 0.0
    s_url = state_change_score(url_changed=True)
    assert 0.45 <= s_url <= 1.0
    s_dialog = state_change_score(vector={"dialogs_delta": 1, "changed": 1})
    assert s_dialog >= 0.3
    # everything at once stays clamped
    s_max = state_change_score(url_changed=True, semantic_changed=True,
                               element_delta=10, new_count=10, disappeared_count=10,
                               vector={"dialogs_delta": 1, "expanded_delta": 1, "changed": 8})
    assert s_max == 1.0


def test_diffing_flag(monkeypatch):
    monkeypatch.delenv("V29_ENABLED", raising=False)
    monkeypatch.delenv("V29_DIFFING", raising=False)
    assert ff.diffing_enabled() is True
    monkeypatch.setenv("V29_DIFFING", "0")
    assert ff.diffing_enabled() is False
    monkeypatch.setenv("V29_DIFFING", "1")
    monkeypatch.setenv("V29_ENABLED", "0")
    assert ff.diffing_enabled() is False


def test_critic_verdict_carries_state_change_score():
    # the Verdict slot exists and defaults to 0.0 (no regression to existing fields)
    from orchestrator.critic_v12 import Verdict
    v = Verdict(success=True, reason="x")
    assert hasattr(v, "state_change_score") and v.state_change_score == 0.0
    v2 = Verdict(success=True, reason="y", state_change_score=0.7)
    assert v2.state_change_score == 0.7


# ═══════════════════════════════════════════════════════════════════════════════
#  Hybrid Primitives — clean semantic feedback (FailureClass) + new primitives
# ═══════════════════════════════════════════════════════════════════════════════

def test_classify_failure_is_universal_and_semantic():
    assert classify_failure("click", "Click timed out (30s)") == FailureClass.TIMEOUT
    assert classify_failure("goto", "Domain blocked: x") == FailureClass.BLOCKED
    assert classify_failure("click", "no live node for id") == FailureClass.NOT_FOUND
    assert classify_failure("click", "click ineffective: nothing") == FailureClass.NO_EFFECT
    assert classify_failure("type", "input not editable") == FailureClass.INPUT_FAILED
    assert classify_failure("click", "something weird") == FailureClass.UNKNOWN


def test_render_failure_preserves_detector_substrings_and_hides_strategy():
    # Reality Monitor + Intent Journal depend on 'timed out'/'timeout' surviving.
    msg = render_failure(FailureClass.TIMEOUT, "Click timed out (30s)")
    assert "timed out" in msg and "timeout" in msg.lower()
    assert "FAILED" in msg and "[timeout]" in msg
    # internal tier/strategy names must NOT leak into the LLM's context
    assert "cdp" not in msg.lower() and "js_click" not in msg.lower()


def test_render_success_is_terse_with_no_strategy_leak():
    assert render_success() == "→ OK"
    assert render_success(["navigated", "DOM changed"]) == "→ OK (navigated, DOM changed)"
    assert "cdp" not in render_success(["DOM changed"]).lower()


def test_hybrid_primitives_flag(monkeypatch):
    monkeypatch.delenv("V29_ENABLED", raising=False)
    monkeypatch.delenv("V29_HYBRID_PRIMITIVES", raising=False)
    assert ff.hybrid_primitives_enabled() is True
    monkeypatch.setenv("V29_HYBRID_PRIMITIVES", "0")
    assert ff.hybrid_primitives_enabled() is False
    monkeypatch.setenv("V29_HYBRID_PRIMITIVES", "1")
    monkeypatch.setenv("V29_ENABLED", "0")
    assert ff.hybrid_primitives_enabled() is False


# ── new primitive handlers (fake page; reuse existing keyboard/__aid backends) ──

class _FakeKbd:
    def __init__(self):
        self.pressed = []

    async def press(self, key):
        self.pressed.append(key)


class _FakePage:
    def __init__(self, eval_ret=None):
        self.keyboard = _FakeKbd()
        self._eval_ret = eval_ret
        self.viewport_size = {"width": 1000, "height": 800}

    async def evaluate(self, js, arg=None):
        return self._eval_ret


async def test_mcp_press_key():
    fake = _FakePage()
    mcp_tools.set_page(fake)
    r = await mcp_tools.mcp_press_key("Escape")
    assert r["success"] and fake.keyboard.pressed == ["Escape"]
    assert (await mcp_tools.mcp_press_key(""))["success"] is False


async def test_mcp_select_option_native_and_fallback():
    mcp_tools.set_page(_FakePage(eval_ret={"ok": True, "selected": "Large"}))
    r = await mcp_tools.mcp_select_option("e5", "Large")
    assert r["success"] and r["selected"] == "Large"
    # not a native <select> → clean reason so the agent falls back to clicking
    mcp_tools.set_page(_FakePage(eval_ret={"ok": False, "reason": "not a native select element"}))
    r2 = await mcp_tools.mcp_select_option("e5", "Large")
    assert r2["success"] is False and "native select" in r2["error"]
    assert (await mcp_tools.mcp_select_option(None, "Large"))["success"] is False


def test_new_primitives_wired_end_to_end():
    ow = (Path(__file__).parent / "overwatch.py").read_text()
    assert "select_option" in ow and "mcp_hover" in ow and "mcp_press_key" in ow
    assert "_fmt_fail" in ow                       # clean semantic failure path
    mr = (Path(__file__).parent / "moe_router.py").read_text()
    assert "select_option" in mr and "hover" in mr
    bw = (Path(__file__).parent / "workers" / "base_worker.py").read_text()
    assert "select_option" in bw and "press_key" in bw


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
