"""Unit and headless Chromium integration tests for the V2 Browser Engine.

Verifies:
  - Dual-mode engine switch (`legacy` default vs `v2` explicit), keeping
    legacy behavior 100% intact.
  - V2SnapshotCompressor compact format, V34 secret redaction preservation,
    same-URL incremental delta compression, refresh interval, and invalidation.
  - Adaptive DOM-quietness settle speed vs static sleep.
  - Paint-order / elementFromPoint occlusion filtering on modals + safety guard.
  - Tier-2 Accessibility-Tree escalation when Tier-1 DOM is sparse.
  - V2 resilient actions: stale-ref semantic rebind on SPA re-render, clean
    STALE_REF fail on unrecoverable detachment, disabled short-circuit, fixed
    window.__aid state safety net, and bounded transient retry.
  - LangGraph native RetryPolicy attached ONLY in v2 mode (legacy untouched).
  - UNIVERSAL-ONLY mandate on `src/v2_engine/perception.py`.

Run: .venv/bin/python -m pytest tests/test_v2_engine.py -v
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

import dom_parser
import feature_flags as ff
from brain_graph import build_brain_graph
from v2_engine import (
    ENGINE_MODE_LEGACY,
    ENGINE_MODE_V2,
    get_engine_mode,
    is_v2_mode,
    normalize_engine_mode,
    set_engine_mode,
)
from v2_engine.actions import inspect_live_ref, v2_click, v2_type
from v2_engine.graph_policy import detect_stealth_driver, is_transient_graph_error, v2_retry_policy
from v2_engine.metrics import RunMetrics, estimate_tokens
from v2_engine.perception import (
    V2SnapshotCompressor,
    extract_tier2_a11y_fallback,
    filter_occluded_elements,
    perceive_v2,
    wait_for_dom_settle,
)

SRC = Path(__file__).resolve().parents[1] / "src"


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Dual-Mode Switch & UNIVERSAL-ONLY Mandate
# ═══════════════════════════════════════════════════════════════════════════════

def test_engine_mode_switching_and_defaults(monkeypatch):
    monkeypatch.delenv("AGENT_ENGINE_MODE", raising=False)
    assert get_engine_mode() == ENGINE_MODE_LEGACY
    assert is_v2_mode() is False
    assert ff.v2_engine_enabled() is False

    assert set_engine_mode("v2") == ENGINE_MODE_V2
    assert get_engine_mode() == ENGINE_MODE_V2
    assert is_v2_mode() is True
    assert ff.v2_engine_enabled() is True

    assert set_engine_mode("legacy") == ENGINE_MODE_LEGACY
    assert is_v2_mode() is False

    with pytest.raises(ValueError):
        normalize_engine_mode("invalid_mode")


def test_v2_perception_source_is_universal_only():
    src = (SRC / "v2_engine" / "perception.py").read_text(encoding="utf-8").lower()
    forbidden = [
        "flipkart", "amazon", "imdb", "reddit", "youtube", "hashnode",
        "saucedemo", "twitter", "github.com", "google.com",
        "add to cart", "add to bag", "buy now", "place order", "checkout",
    ]
    hits = [w for w in forbidden if w in src]
    assert not hits, f"UNIVERSAL-ONLY violation in v2_engine/perception.py: {hits}"


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Compact Ref Format & Incremental Delta Compression
# ═══════════════════════════════════════════════════════════════════════════════

def test_snapshot_compressor_full_then_delta_and_refresh():
    comp = V2SnapshotCompressor(full_refresh_interval=3)
    els_1 = [
        {"id": "e1", "kind": "link", "text": "Home", "hint": "/home"},
        {"id": "e2", "kind": "link", "text": "Docs", "hint": "/docs"},
        {"id": "e3", "kind": "input", "text": "Email [empty]", "hint": ""},
        {"id": "e4", "kind": "input", "text": "Password [filled: value hidden, 12ch]", "hint": ""},
        {"id": "e5", "kind": "button", "text": "Submit", "hint": ""},
    ]

    md1, kind1 = comp.render("https://app.test/form", els_1, modal_active=False)
    assert kind1 == "full"
    assert '[e1] link "Home" (/home)' in md1
    assert "[filled: value hidden, 12ch]" in md1

    # Step 2 on same URL: Email filled + new confirmation banner button appears → DELTA
    els_2 = [
        {"id": "e1", "kind": "link", "text": "Home", "hint": "/home"},
        {"id": "e2", "kind": "link", "text": "Docs", "hint": "/docs"},
        {"id": "e3", "kind": "input", "text": 'Email [filled: "a@b.com" 7ch]', "hint": ""},
        {"id": "e4", "kind": "input", "text": "Password [filled: value hidden, 12ch]", "hint": ""},
        {"id": "e5", "kind": "button", "text": "Submit", "hint": ""},
        {"id": "e6", "kind": "button", "text": "Confirm", "hint": ""},
    ]
    md2, kind2 = comp.render("https://app.test/form", els_2)
    assert kind2 == "delta"
    assert "# DELTA (+1 added, ~1 changed" in md2
    assert '+ [e6] button "Confirm"' in md2
    assert '~ [e3] input "Email [filled:' in md2
    assert "UNCHANGED:" in md2 and "[e5]" in md2

    # Step 3 on same URL: still delta (streak=2 < interval-1=2 → refreshes on step 4; here interval=3 → streak 2 refreshes!)
    md3, kind3 = comp.render("https://app.test/form", els_2)
    assert kind3 == "delta"
    md4, kind4 = comp.render("https://app.test/form", els_2)
    assert kind4 == "full"  # periodic full refresh prevents drift

    # URL change always resets to full snapshot
    _, kind5 = comp.render("https://app.test/next", els_2)
    assert kind5 == "full"


# ═══════════════════════════════════════════════════════════════════════════════
#  3. LangGraph RetryPolicy & Stealth Driver Detection
# ═══════════════════════════════════════════════════════════════════════════════

def test_langgraph_retry_policy_predicate_and_dual_mode_graph(monkeypatch):
    assert is_transient_graph_error(TimeoutError("page timed out")) is True
    assert is_transient_graph_error(ConnectionError("reset")) is True
    assert is_transient_graph_error(RuntimeError("HTTP 429 rate limit")) is True
    assert is_transient_graph_error(ValueError("bad schema")) is False
    assert is_transient_graph_error(KeyError("missing")) is False

    pol = v2_retry_policy(max_attempts=3)
    assert pol.max_attempts == 3

    monkeypatch.delenv("AGENT_ENGINE_MODE", raising=False)
    g_legacy = build_brain_graph(mode="legacy")
    g_v2 = build_brain_graph(mode="v2")

    assert g_legacy.nodes["perceive"].retry_policy is None
    assert g_legacy.nodes["overwatch"].retry_policy is None
    assert g_v2.nodes["perceive"].retry_policy is not None
    assert g_v2.nodes["planner"].retry_policy is not None
    assert g_v2.nodes["overwatch"].retry_policy is not None
    assert g_v2.nodes["router"].retry_policy is None

    drv = detect_stealth_driver()
    assert drv["driver"] in ("playwright", "patchright")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Live Headless Chromium Integration Tests (Settle, Occlusion, Tier-2, Actions)
# ═══════════════════════════════════════════════════════════════════════════════

def test_v2_adaptive_settle_and_occlusion_filtering_live():
    html = """<!doctype html><html><head><style>
      .bg-btn { position: fixed; left: 200px; top: 200px; width: 120px; height: 40px; }
      .backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 50; }
      .dlg { position: fixed; left: 160px; top: 160px; width: 320px; height: 220px;
             background: white; z-index: 60; padding: 16px; }
    </style></head><body>
      <button class="bg-btn" id="hidden-behind">Hidden Behind Modal</button>
      <div class="backdrop"></div>
      <div class="dlg" role="dialog" aria-modal="true">
        <button id="modal-ok">Accept Modal</button>
        <button id="modal-cancel">Dismiss Modal</button>
      </div>
    </body></html>"""

    async def run():
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1000, "height": 700})
            await page.set_content(html)

            t0 = time.perf_counter()
            settle_ms = await wait_for_dom_settle(page, quiet_ms=100, max_wait_ms=500)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Must be drastically faster than legacy's 1500ms unconditional sleep
            assert settle_ms < 800 and elapsed_ms < 900

            metrics = RunMetrics(mode="v2")
            pr = await perceive_v2(page, ctx={"objective": "accept dialog"}, metrics=metrics, do_settle=True)
            texts = [e["text"] for e in pr.elements]
            assert any("Accept Modal" in t for t in texts), texts
            assert any("Dismiss Modal" in t for t in texts), texts
            assert not any("Hidden Behind Modal" in t for t in texts), texts
            assert "# ACTIVE MODAL/DIALOG" in pr.markdown
            assert metrics.steps[0].occluded_filtered >= 1
            assert estimate_tokens(pr.markdown) > 0
        finally:
            await browser.close()
            await pw.stop()

    asyncio.run(run())


def test_v2_occlusion_safety_guard_preserves_elements_under_transparent_shield():
    html = """<!doctype html><html><head><style>
      .shield { position: fixed; inset: 0; z-index: 999; background: transparent; }
    </style></head><body>
      <button id="b1">First</button>
      <button id="b2">Second</button>
      <div class="shield"></div>
    </body></html>"""

    async def run():
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 800, "height": 600})
            await page.set_content(html)
            dom_data = await dom_parser.extract(page)
            els = dom_data["elements"]
            smap = {e["id"]: e for e in els}
            kept, _, pruned, _ = await filter_occluded_elements(page, els, smap)
            assert pruned == 0
            assert len(kept) == len(els)
        finally:
            await browser.close()
            await pw.stop()

    asyncio.run(run())


def test_v2_actions_stale_ref_rebind_state_safety_net_and_retry_live():
    html = """<!doctype html><html><body>
      <div id="box">
        <button id="go" style="position:absolute;left:40px;top:40px;">Launch Rocket</button>
      </div>
      <button id="dis" disabled style="position:absolute;left:40px;top:120px;">Locked Action</button>
      <input id="cb" type="checkbox" style="position:absolute;left:40px;top:200px;" />
    </body></html>"""

    async def run():
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1000, "height": 700})
            await page.set_content(html)
            snap = await dom_parser.extract(page)
            els = snap["elements"]
            go_el = next(e for e in els if "Launch Rocket" in e["text"])
            dis_el = next(e for e in els if "Locked Action" in e["text"])
            cb_el = next(e for e in els if e["id"] != go_el["id"] and e["id"] != dis_el["id"])

            metrics = RunMetrics(mode="v2")

            # 1. Disabled element fails fast without wasting click retries
            dis_res = await v2_click(page, element_id=dis_el["id"], elements_list=els, metrics=metrics)
            assert dis_res["success"] is False
            assert "disabled" in dis_res["error"].lower()

            # 2. SPA re-render detaches #go and mounts new button at shifted position → V2 rebinds!
            await page.evaluate("""() => {
                document.getElementById('box').innerHTML =
                    '<button id="go2" style="position:absolute;left:400px;top:260px;">Launch Rocket</button>';
                window.__launched = false;
                document.getElementById('go2').addEventListener('click', () => { window.__launched = true; });
            }""")
            assert (await inspect_live_ref(page, go_el["id"]))["found"] is False

            async def _real_mouse_click(x=0, y=0, element_id=None):
                await page.mouse.click(x, y)
                return {"success": True, "strategy": "mouse", "navigated": False, "dom_changed": True, "error": ""}

            rebound_res = await v2_click(
                page,
                x=go_el["x"],
                y=go_el["y"],
                element_id=go_el["id"],
                elements_list=els,
                metrics=metrics,
                click_fn=_real_mouse_click,
            )
            assert rebound_res["success"] is True
            assert await page.evaluate("() => !!window.__launched") is True
            assert metrics.total_stale_ref_recoveries == 1

            # 3. Unrecoverable detached element returns clean STALE_REF instead of blind click
            await page.evaluate("() => document.getElementById('box').innerHTML = ''")
            stale_res = await v2_click(
                page,
                x=go_el["x"],
                y=go_el["y"],
                element_id=go_el["id"],
                elements_list=els,
                metrics=metrics,
                click_fn=_real_mouse_click,
            )
            assert stale_res["success"] is False
            assert stale_res.get("stale_ref") is True

            # 4. Fixed V32 window.__aid state safety net overrides false-negative click failure
            async def _false_negative_click(x=0, y=0, element_id=None):
                await page.evaluate("() => { document.getElementById('cb').checked = true; }")
                return {
                    "success": False, "strategy": "", "navigated": False,
                    "dom_changed": False, "error": "Click ineffective",
                }

            cb_res = await v2_click(
                page,
                x=cb_el["x"],
                y=cb_el["y"],
                element_id=cb_el["id"],
                elements_list=els,
                metrics=metrics,
                click_fn=_false_negative_click,
            )
            assert cb_res["success"] is True
            assert cb_res["strategy"] == "v2_aid_state_verified"
            assert metrics.total_state_safety_overrides == 1

            # 5. Bounded transient retry succeeds on second attempt for v2_type
            attempts = {"n": 0}

            async def _flaky_type(text="", x=0, y=0, element_id=None):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    return {"success": False, "actual_length": 0, "error": "Element detached during typing"}
                return {"success": True, "actual_length": len(text), "strategy": "cdp", "error": ""}

            type_res = await v2_type(
                page,
                text="hello",
                x=cb_el["x"],
                y=cb_el["y"],
                element_id=cb_el["id"],
                elements_list=els,
                metrics=metrics,
                type_fn=_flaky_type,
            )
            assert type_res["success"] is True
            assert attempts["n"] == 2
            assert metrics.total_action_retries == 1
        finally:
            await browser.close()
            await pw.stop()

    asyncio.run(run())


async def test_tier2_a11y_checked_and_disabled_string_truthiness(monkeypatch):
    import a11y_parser

    async def _fake_extract(page, timeout=3.0):
        return {
            "source": "cdp_ax_tree",
            "refid_text": "",
            "elements": [
                {"ref": "e1", "role": "menuitemcheckbox", "name": "Unchecked Str", "properties": {"checked": "false"}},
                {"ref": "e2", "role": "checkbox", "name": "Unchecked Bool", "properties": {"checked": False}},
                {"ref": "e3", "role": "checkbox", "name": "Checked Bool", "properties": {"checked": True}},
                {"ref": "e4", "role": "menuitemcheckbox", "name": "Checked Str", "properties": {"checked": "true"}},
                {"ref": "e5", "role": "checkbox", "name": "Mixed Str", "properties": {"checked": "mixed"}},
                {
                    "ref": "e6", "role": "button", "name": "Enabled Str",
                    "properties": {"disabled": "false"},
                },
                {
                    "ref": "e7", "role": "button", "name": "Disabled Str",
                    "properties": {"disabled": "true"},
                },
            ],
        }

    monkeypatch.setattr(a11y_parser, "extract", _fake_extract)
    pr = await extract_tier2_a11y_fallback(page=None)
    assert pr is not None
    by_id = {e["id"]: e["text"] for e in pr.elements}
    assert by_id["e1"] == "Unchecked Str"
    assert by_id["e2"] == "Unchecked Bool"
    assert by_id["e3"] == "Checked Bool [checked]"
    assert by_id["e4"] == "Checked Str [checked]"
    assert by_id["e5"] == "Mixed Str [checked]"
    assert by_id["e6"] == "Enabled Str"
    assert by_id["e7"] == "Disabled Str [disabled]"


def test_v2_click_label_wrapped_and_for_associated_checkbox_live():
    html = """<!doctype html><html><body>
      <label id="lbl-nested" style="position:absolute;left:30px;top:30px;">
        <input id="cb-nested" type="checkbox" /> Enable Canary Rollout
      </label>
      <label id="lbl-for" for="cb-ext" style="position:absolute;left:30px;top:90px;">
        Accept Release Notes
      </label>
      <input id="cb-ext" type="checkbox" style="position:absolute;left:220px;top:90px;" />
    </body></html>"""

    async def run():
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 800, "height": 600})
            await page.set_content(html)
            snap = await dom_parser.extract(page)
            els = snap["elements"]
            nested_lbl = next(e for e in els if "Enable Canary Rollout" in e["text"])
            for_lbl = next(e for e in els if "Accept Release Notes" in e["text"])

            metrics = RunMetrics(mode="v2")

            async def _false_neg_nested(x=0, y=0, element_id=None):
                await page.evaluate("() => { document.getElementById('cb-nested').checked = true; }")
                return {
                    "success": False, "strategy": "", "navigated": False,
                    "dom_changed": False, "error": "Click ineffective",
                }

            r1 = await v2_click(
                page, x=nested_lbl["x"], y=nested_lbl["y"],
                element_id=nested_lbl["id"], elements_list=els,
                metrics=metrics, click_fn=_false_neg_nested,
            )
            assert r1["success"] is True and r1["strategy"] == "v2_aid_state_verified"

            async def _false_neg_ext(x=0, y=0, element_id=None):
                await page.evaluate("() => { document.getElementById('cb-ext').checked = true; }")
                return {
                    "success": False, "strategy": "", "navigated": False,
                    "dom_changed": False, "error": "Click ineffective",
                }

            r2 = await v2_click(
                page, x=for_lbl["x"], y=for_lbl["y"],
                element_id=for_lbl["id"], elements_list=els,
                metrics=metrics, click_fn=_false_neg_ext,
            )
            assert r2["success"] is True and r2["strategy"] == "v2_aid_state_verified"
            assert metrics.total_state_safety_overrides == 2
        finally:
            await browser.close()
            await pw.stop()

    asyncio.run(run())
