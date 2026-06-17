"""Characterization tests for the Adaptive Perception Engine — P0.

P0 is a behavior-identical Tier-1 passthrough, so these tests pin exactly that:
  • PerceptionResult wraps the snapshot contract 1:1.
  • Tier-1 delegates to the EXISTING mcp_snapshot (no second pipeline).
  • the router returns the Tier-1 result unchanged, and is pluggable (modularity).
  • the universal sufficiency heuristic is a pure element-count rule.
  • UNIVERSAL-ONLY guard: the engine source contains NO site/domain/commerce
    literals — enforced mechanically so the mandate can never silently regress.
  • flags + wiring guards.

Pure logic + monkeypatched async snapshot — no browser, no network.
Run: .venv/bin/python -m pytest test_perception_engine_v29.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import mcp_tools
import feature_flags as ff
import perception_engine as pe
from perception_engine import (PerceptionResult, Tier1A11yStrategy, perceive,
                               is_sufficient, default_strategies,
                               strict_viewport_filter)

_SNAP = {
    "elements": [{"id": "e1"}, {"id": "e2"}],
    "markdown": "# page",
    "selector_map": {"e1": {}, "e2": {}},
    "element_count": 2,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Contract: PerceptionResult mirrors the snapshot dict 1:1
# ═══════════════════════════════════════════════════════════════════════════════

def test_perception_result_wraps_snapshot_exactly():
    r = PerceptionResult.from_snapshot(_SNAP, tier=1, strategy="a11y_dom")
    assert r.elements == _SNAP["elements"]
    assert r.markdown == _SNAP["markdown"]
    assert r.selector_map == _SNAP["selector_map"]
    assert r.element_count == 2
    assert r.tier == 1 and r.strategy == "a11y_dom"


def test_from_snapshot_tolerates_empty():
    r = PerceptionResult.from_snapshot({}, tier=1, strategy="x")
    assert r.elements == [] and r.markdown == "" and r.element_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier-1 delegates to the EXISTING mcp_snapshot (no duplicate pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tier1_delegates_to_mcp_snapshot(monkeypatch):
    async def _fake():
        return dict(_SNAP)
    monkeypatch.setattr(mcp_tools, "mcp_snapshot", _fake)
    r = await Tier1A11yStrategy().extract(page=None, ctx={})
    assert r.element_count == 2 and r.markdown == "# page" and r.tier == 1


async def test_perceive_is_tier1_passthrough(monkeypatch):
    async def _fake():
        return dict(_SNAP)
    monkeypatch.setattr(mcp_tools, "mcp_snapshot", _fake)
    r = await perceive(page=None, ctx={"objective": "anything"})
    # identical contract, unchanged
    assert r.elements == _SNAP["elements"]
    assert r.selector_map == _SNAP["selector_map"]
    assert r.tier == 1 and r.strategy == "a11y_dom" and r.sufficient is True


async def test_router_is_pluggable_modularity(monkeypatch):
    class _Fake:
        name = "fake"
        tier = 9
        async def extract(self, page, ctx):
            return PerceptionResult(elements=[{"id": "z"}], element_count=1,
                                    tier=9, strategy="fake")
    r = await perceive(page=None, strategies=[_Fake()])
    assert r.strategy == "fake" and r.tier == 9
    # element_count 1 < floor 2 → flagged sparse (universal heuristic)
    assert r.sufficient is False and "sparse" in r.note


# ═══════════════════════════════════════════════════════════════════════════════
#  Universal sufficiency heuristic
# ═══════════════════════════════════════════════════════════════════════════════

def test_is_sufficient_is_pure_count_rule():
    assert is_sufficient(PerceptionResult(element_count=2)) is True
    assert is_sufficient(PerceptionResult(element_count=5)) is True
    assert is_sufficient(PerceptionResult(element_count=1)) is False
    assert is_sufficient(PerceptionResult(element_count=0)) is False


def test_default_strategies_is_tier1_only_at_p0():
    s = default_strategies()
    assert len(s) == 1 and s[0].tier == 1 and s[0].name == "a11y_dom"


# ═══════════════════════════════════════════════════════════════════════════════
#  AP-P1 — Strict viewport filter (universal; recall-preserving)
# ═══════════════════════════════════════════════════════════════════════════════

# vw=1000, vh=800. e1 on-screen; e2 off-screen noise link (y=-34, the exact case);
# e3 off-screen BUTTON (always kept); e4 off-screen link relevant to the goal.
_VP_ELS = [
    {"id": "e1", "kind": "button", "text": "Submit", "x": 500, "y": 400, "hint": ""},
    {"id": "e2", "kind": "link", "text": "Footer privacy", "x": 500, "y": -34, "hint": ""},
    {"id": "e3", "kind": "button", "text": "Confirm choice", "x": 500, "y": 1200, "hint": ""},
    {"id": "e4", "kind": "link", "text": "realme buds wireless", "x": 500, "y": 1500, "hint": ""},
]
_VP_MAP = {"e1": {}, "e2": {}, "e3": {}, "e4": {}}
_VP_MD = (
    "- 🔘 **[e1]** button: Submit → (500,400)\n"
    "- 🔗 **[e2]** link: Footer privacy → (500,-34)\n"
    "- 🔘 **[e3]** button: Confirm choice → (500,1200)\n"
    "- 🔗 **[e4]** link: realme buds wireless → (500,1500)"
)


def _run_vp(goal_tokens):
    return strict_viewport_filter(_VP_ELS, _VP_MAP, _VP_MD, vw=1000, vh=800,
                                  goal_tokens=goal_tokens)


def test_strict_drops_offscreen_noise_keeps_actionable_and_goal():
    els, smap, md, dropped, n_off = _run_vp({"realme", "buds"})
    ids = {e["id"] for e in els}
    assert ids == {"e1", "e3", "e4"}        # e2 (off-screen noise link) dropped
    assert dropped == 1 and n_off == 2
    # on-screen unchanged (no offscreen key); off-screen-kept tagged offscreen
    assert "offscreen" not in next(e for e in els if e["id"] == "e1")
    assert next(e for e in els if e["id"] == "e3")["offscreen"] is True
    assert next(e for e in els if e["id"] == "e4")["offscreen"] is True
    # selector_map filtered in lockstep
    assert set(smap.keys()) == {"e1", "e3", "e4"}
    # markdown: e2 line gone; e3/e4 tagged; e1 untouched
    assert "[e2]" not in md and "Footer privacy" not in md
    assert md.count("offscreen — scroll to reach") == 2
    assert "[e1]" in md and "(offscreen" not in md.split("\n")[0]


def test_y_negative_34_is_dropped_when_not_critical():
    # the user's literal example: a non-actionable element at y=-34 must vanish
    els, _, _, dropped, _ = strict_viewport_filter(
        [{"id": "e9", "kind": "other", "text": "promo banner", "x": 100, "y": -34}],
        {"e9": {}}, "- • **[e9]** other: promo banner → (100,-34)", vw=1000, vh=800,
        goal_tokens=set())
    assert els == [] and dropped == 1


def test_offscreen_goal_relevance_preserves_link():
    # same link, but now NOT goal-relevant → dropped
    els, *_ = _run_vp(goal_tokens=set())
    assert {e["id"] for e in els} == {"e1", "e3"}   # e4 link no longer preserved


def test_coordless_elements_pass_through_unchanged():
    els, smap, md, dropped, n_off = strict_viewport_filter(
        [{"id": "e1"}, {"id": "e2"}], {"e1": {}, "e2": {}}, "# page",
        vw=1000, vh=800, goal_tokens=set())
    assert els == [{"id": "e1"}, {"id": "e2"}] and dropped == 0 and n_off == 0


def test_strict_viewport_flag(monkeypatch):
    monkeypatch.delenv("V29_ENABLED", raising=False)
    monkeypatch.delenv("V29_ADAPTIVE_PERCEPTION", raising=False)
    monkeypatch.delenv("V29_STRICT_VIEWPORT", raising=False)
    assert ff.strict_viewport_enabled() is True
    monkeypatch.setenv("V29_STRICT_VIEWPORT", "0")
    assert ff.strict_viewport_enabled() is False
    monkeypatch.setenv("V29_STRICT_VIEWPORT", "1")
    monkeypatch.setenv("V29_ADAPTIVE_PERCEPTION", "0")
    assert ff.strict_viewport_enabled() is False   # parent gate wins


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL-ONLY mandate — enforced mechanically on the source
# ═══════════════════════════════════════════════════════════════════════════════

def test_engine_source_has_no_site_or_commerce_literals():
    src = (Path(__file__).parent / "perception_engine.py").read_text().lower()
    forbidden = [
        # brands / specific sites
        "flipkart", "amazon", "imdb", "reddit", "youtube", "hashnode",
        "saucedemo", "twitter", "github.com", "google.com",
        # commerce-specific action hardcodes
        "add to cart", "add to bag", "buy now", "place order", "checkout",
    ]
    hits = [w for w in forbidden if w in src]
    assert not hits, f"UNIVERSAL-ONLY violation — site/commerce literal(s) in engine: {hits}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Flags + wiring
# ═══════════════════════════════════════════════════════════════════════════════

def test_adaptive_perception_flag(monkeypatch):
    monkeypatch.delenv("V29_ENABLED", raising=False)
    monkeypatch.delenv("V29_ADAPTIVE_PERCEPTION", raising=False)
    assert ff.adaptive_perception_enabled() is True
    monkeypatch.setenv("V29_ADAPTIVE_PERCEPTION", "0")
    assert ff.adaptive_perception_enabled() is False
    monkeypatch.setenv("V29_ADAPTIVE_PERCEPTION", "1")
    monkeypatch.setenv("V29_ENABLED", "0")
    assert ff.adaptive_perception_enabled() is False  # master kill-switch wins


def test_perceive_node_wires_engine_with_direct_fallback():
    bg = (Path(__file__).parent / "brain_graph.py").read_text()
    assert "perception_engine" in bg and "adaptive_perception_enabled" in bg
    assert "_direct_snapshot" in bg          # hard fallback preserved
    assert "mcp_snapshot" in bg              # the proven Tier-1 call still present


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
