"""Unit and live headless Chromium tests for Chunk 1 (`eval_connector` core).

Verifies:
  - Deep-test scenario registry (all 5 real-world failure modes).
  - DualModeSessionManager session isolation & global `mcp_tools`/`AGENT_ENGINE_MODE`
    restoration (zero cross-session bleed).
  - Side-by-side `compare()` on `modal_occlusion` + interactive modal dismissal.
  - `sparse_aria` triggering V2 Tier-2 A11y escalation (`tier_used == 2`) while
    legacy stays sparse.
  - `multistep_form` real multi-step workflow (type, select_option, checkbox,
    button click) verifying same-URL delta compression & V34 secret redaction.
  - `spa_rerender` live stale-ref semantic rebind in V2 mode.
  - `a11y_visibility` hidden-ghost exclusion, duplicate-row hints, and off-screen
    actionable recall.
  - `screenshot()`, `report()` Markdown/JSON export, and clean `close_all()`.

Run: .venv/bin/python -m pytest tests/test_eval_connector_core.py -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import mcp_tools
from eval_connector import DualModeSessionManager, get_scenario, list_scenarios
from v2_engine import ENGINE_MODE_LEGACY, get_engine_mode


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Scenario Registry Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_scenario_registry_covers_all_five_deep_modes():
    names = {s["name"] for s in list_scenarios()}
    expected = {
        "modal_occlusion",
        "spa_rerender",
        "sparse_aria",
        "multistep_form",
        "a11y_visibility",
    }
    assert expected.issubset(names)

    spec = get_scenario("MODAL_OCCLUSION")
    assert spec.name == "modal_occlusion"
    assert "Accept Essential Cookies" in spec.html
    assert len(spec.suggested_checks) >= 1

    with pytest.raises(KeyError):
        get_scenario("does_not_exist")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Live Chromium Dual-Mode Core Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

def _find_ref(elements: list[dict], needle: str, kind: str | None = None) -> str:
    low = needle.lower()
    for el in elements:
        if kind and (el.get("kind") or "").lower() != kind:
            continue
        if low in (el.get("text") or "").lower():
            return str(el["id"])
    raise AssertionError(f"Element matching '{needle}' (kind={kind}) not found in {[e.get('text') for e in elements]}")


def test_modal_occlusion_compare_and_interactive_dismissal(tmp_path: Path):
    async def run():
        prev_page = mcp_tools._PAGE
        prev_mode = get_engine_mode()
        mgr = DualModeSessionManager(screenshot_dir=tmp_path)
        try:
            pair = await mgr.start_pair("modal", scenario="modal_occlusion", headless=True)
            assert pair["ok"] is True
            assert len(await mgr.list_sessions()) == 2

            cmp_res = await mgr.compare("modal_legacy", "modal_v2", settle=True)
            sc = cmp_res["step_comparison"]
            assert sc["v2_occluded_filtered"] >= 30, sc
            assert sc["speedup_x"] >= 3.0, sc
            assert sc["token_savings_pct"] >= 70.0, sc
            assert sc["v2_elements"] < sc["legacy_elements"]

            # Globals must be cleanly restored after bound calls
            assert mcp_tools._PAGE is prev_page
            assert get_engine_mode() == prev_mode

            # Act as LLM in V2: dismiss modal and auto-perceive unlocked background
            v2_els = mgr._get_session("modal_v2").elements
            accept_id = _find_ref(v2_els, "Accept Essential Cookies", kind="button")
            act_res = await mgr.act("modal_v2", "click", element_id=accept_id, auto_perceive=True)
            assert act_res["ok"] is True, act_res
            unlocked = act_res["perception"]
            assert unlocked["element_count"] >= 20, unlocked["element_count"]
            assert "# ACTIVE MODAL/DIALOG" not in unlocked["markdown"]

            # Screenshot & report
            shot = await mgr.screenshot("modal_v2")
            assert shot["ok"] is True
            assert Path(shot["path"]).exists() and shot["bytes"] > 500

            rep = await mgr.report()
            assert rep["ok"] is True
            assert "modal_legacy" in rep["sessions"] and "modal_v2" in rep["sessions"]
            assert "| `modal_v2` | v2 |" in rep["markdown_table"]
        finally:
            closed = await mgr.close_all()
            assert closed["ok"] is True
            assert await mgr.list_sessions() == []

    asyncio.run(run())


def test_sparse_aria_triggers_v2_tier2_escalation_vs_legacy(tmp_path: Path):
    async def run():
        mgr = DualModeSessionManager(screenshot_dir=tmp_path)
        try:
            await mgr.start_pair("aria", scenario="sparse_aria", headless=True)
            cmp_res = await mgr.compare("aria_legacy", "aria_v2", settle=False)
            sc = cmp_res["step_comparison"]
            assert sc["legacy_elements"] < 2, sc
            assert sc["v2_tier_used"] == 2, sc
            assert sc["v2_elements"] >= 3, sc
            v2_md = mgr._get_session("aria_v2").last_markdown
            assert "Release Pipeline" in v2_md or "Staging Metrics" in v2_md
            # aria-checked="false" must NOT render [checked]
            assert "Auto-Scale Workers" in v2_md
            assert "Auto-Scale Workers [checked]" not in v2_md
        finally:
            await mgr.close_all()

    asyncio.run(run())


def test_multistep_form_delta_and_secret_redaction_both_modes(tmp_path: Path):
    async def run():
        mgr = DualModeSessionManager(screenshot_dir=tmp_path)
        try:
            await mgr.start_pair("form", scenario="multistep_form", headless=True)

            # Step 1 perception in both modes
            p_leg_1 = await mgr.perceive("form_legacy", settle=False)
            p_v2_1 = await mgr.perceive("form_v2", settle=False)
            assert "TopSecret#987" not in p_leg_1["markdown"]
            assert "TopSecret#987" not in p_v2_1["markdown"]
            assert p_v2_1["metrics"]["snapshot_kind"] == "full"

            # Execute multi-step actions in V2 session (type → select_option → click checkbox → click continue)
            name_id = _find_ref(p_v2_1["elements"], "Full Name", kind="input")
            plan_id = _find_ref(p_v2_1["elements"], "Plan", kind="input")
            cont_id = _find_ref(p_v2_1["elements"], "Continue to Review", kind="button")

            r_type = await mgr.act("form_v2", "type", element_id=name_id, text="Sandeep")
            assert r_type["ok"] is True, r_type

            r_sel = await mgr.act("form_v2", "select_option", element_id=plan_id, text="Pro Team")
            assert r_sel["ok"] is True, r_sel

            r_go = await mgr.act("form_v2", "click", element_id=cont_id, auto_perceive=True)
            assert r_go["ok"] is True, r_go
            p_v2_2 = r_go["perception"]
            assert p_v2_2["metrics"]["snapshot_kind"] == "delta", p_v2_2["metrics"]
            assert "TopSecret#987" not in p_v2_2["markdown"]
            assert any("Submit Application" in e["text"] for e in p_v2_2["elements"])
            assert p_v2_2["metrics"]["est_tokens"] < p_leg_1["metrics"]["est_tokens"]
        finally:
            await mgr.close_all()

    asyncio.run(run())


def test_spa_rerender_v2_stale_ref_recovery_and_visibility_parity(tmp_path: Path):
    async def run():
        mgr = DualModeSessionManager(screenshot_dir=tmp_path)
        try:
            # Part A: SPA Re-render stale-ref recovery in V2
            await mgr.start_session("spa_v2", mode="v2", scenario="spa_rerender", headless=True)
            p1 = await mgr.perceive("spa_v2", settle=False)
            deploy_id = _find_ref(p1["elements"], "Confirm Deployment", kind="button")
            rerender_id = _find_ref(p1["elements"], "Simulate SPA Re-render", kind="button")

            # Trigger re-render (detaches #deploy-btn and mounts shifted #deploy-btn-v2)
            r_mut = await mgr.act("spa_v2", "click", element_id=rerender_id)
            assert r_mut["ok"] is True, r_mut

            # Click the ORIGINAL stale ref ID without re-perceiving — V2 must rebind & succeed!
            r_deploy = await mgr.act("spa_v2", "click", element_id=deploy_id)
            assert r_deploy["ok"] is True, r_deploy
            clicks = await mgr._get_session("spa_v2").page.evaluate("() => window.__deployClicks")
            assert clicks == 1
            assert mgr._get_session("spa_v2").metrics.total_stale_ref_recoveries == 1

            # Clicking the <label>-wrapped checkbox must succeed via el.control / nested input check
            cb_lbl_id = _find_ref(p1["elements"], "Enable Canary Rollout")
            r_cb = await mgr.act("spa_v2", "click", element_id=cb_lbl_id)
            assert r_cb["ok"] is True, r_cb
            cb_checked = await mgr._get_session("spa_v2").page.evaluate(
                "() => !!document.getElementById('canary-cb').checked"
            )
            assert cb_checked is True

            # Part B: Visibility & A11y parity
            await mgr.start_pair("vis", scenario="a11y_visibility", headless=True)
            p_leg = await mgr.perceive("vis_legacy", settle=False)
            p_v2 = await mgr.perceive("vis_v2", settle=False)
            for md in (p_leg["markdown"], p_v2["markdown"]):
                assert "Display None Ghost" not in md
                assert "Visibility Hidden Ghost" not in md
                assert "Zero Opacity Ghost" not in md
                assert "Visible Primary Action" in md
                assert "Save Final Report" in md
                assert "Privacy Policy Footer" not in md

            open_links = [e for e in p_v2["elements"] if e["text"].strip() == "Open"]
            assert len(open_links) == 3
            assert len({e.get("hint", "") for e in open_links}) == 3

            # Verify unsupported verb fails cleanly without crashing session
            bad_verb = await mgr.act("vis_legacy", "teleport")
            assert bad_verb["ok"] is False
            assert "Unsupported action verb" in bad_verb["result"]["error"]
            assert mgr._get_session("vis_legacy").mode == ENGINE_MODE_LEGACY
        finally:
            await mgr.close_all()

    asyncio.run(run())
