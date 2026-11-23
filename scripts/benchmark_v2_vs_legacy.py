#!/usr/bin/env python3
"""Side-by-Side Live Benchmark & Test Runner: Legacy vs V2 Browser Engine.

Lets you live-test and compare `--mode legacy` and `--mode v2` on identical
real-world URLs, built-in local stress suites, or full LangGraph E2E tasks.

Examples:
    # 1. Local stress suite (zero API keys, zero network — real Chromium):
    .venv/bin/python scripts/benchmark_v2_vs_legacy.py --suite local

    # 2. Live real-world URLs side-by-side (perception, settle, occlusion, tokens):
    .venv/bin/python scripts/benchmark_v2_vs_legacy.py \\
        --urls https://news.ycombinator.com https://the-internet.herokuapp.com/login

    # 3. Full end-to-end LangGraph run_brain comparison (uses your .env LLM keys):
    .venv/bin/python scripts/benchmark_v2_vs_legacy.py \\
        --e2e "Log in to https://the-internet.herokuapp.com/login as tomsmith / SuperSecretPassword!"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dom_parser  # noqa: E402
from perception_engine import _viewport_of, strict_viewport_filter  # noqa: E402
from v2_engine.actions import v2_click  # noqa: E402
from v2_engine.graph_policy import detect_stealth_driver  # noqa: E402
from v2_engine.metrics import RunMetrics, estimate_tokens  # noqa: E402
from v2_engine.perception import V2SnapshotCompressor, perceive_v2  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
#  Built-in Local Stress Fixtures (zero network, deterministic live Chromium)
# ═══════════════════════════════════════════════════════════════════════════════

_MODAL_OCCLUSION_HTML = """<!doctype html><html><head><title>Modal Occlusion Test</title>
<style>
  body { font-family: sans-serif; margin: 0; padding: 20px; }
  .bg-grid { display: grid; grid-template-columns: repeat(4, 180px); gap: 12px; }
  .card { border: 1px solid #ccc; padding: 10px; }
  .backdrop {
    position: fixed; inset: 0; background: rgba(0,0,0,0.65); z-index: 100;
    display: flex; align-items: center; justify-content: center;
  }
  .modal {
    background: #fff; padding: 24px; width: 380px; border-radius: 8px; z-index: 101;
  }
</style></head><body>
  <header><nav>
    <a href="/home">Home</a> <a href="/catalog">Catalog</a>
    <a href="/pricing">Pricing</a> <a href="/docs">Docs</a>
  </nav></header>
  <main class="bg-grid">
    __BG_CARDS__
  </main>
  <div class="backdrop">
    <div class="modal" role="dialog" aria-modal="true" aria-label="Consent Dialog">
      <h2>Cookie Preferences</h2>
      <p>Choose how we store session preferences.</p>
      <button id="accept">Accept Essential Cookies</button>
      <button id="manage">Manage Settings</button>
    </div>
  </div>
</body></html>
""".replace(
    "__BG_CARDS__",
    "".join(
        f'<div class="card"><a href="/item/{i}">Product Item {i}</a>'
        f'<button>Select Item {i}</button></div>'
        for i in range(20)
    ),
)

_SPA_MULTI_STEP_HTML = """<!doctype html><html><head><title>SPA Multi-Step Form</title></head><body>
  <header><nav>
    <a href="/dash">Dashboard</a> <a href="/settings">Settings</a>
    <a href="/billing">Billing</a> <a href="/team">Team</a>
    <a href="/audit">Audit Log</a> <a href="/support">Support</a>
  </nav></header>
  <main>
    <form id="profile-form">
      <label for="f-name">Full Name</label><input id="f-name" type="text" placeholder="Full Name" />
      <label for="f-email">Work Email</label><input id="f-email" type="email" placeholder="Work Email" />
      <label for="f-pass">Account Password</label><input id="f-pass" type="password" value="UltraSecret#42" />
      <label for="f-role">Role</label>
      <select id="f-role"><option>Engineer</option><option>Researcher</option><option>Admin</option></select>
      <label><input id="f-tos" type="checkbox" /> Accept Terms</label>
      <button id="save-btn" type="button">Save Profile</button>
    </form>
    <section id="links">
      __EXTRA_LINKS__
    </section>
  </main>
</body></html>
""".replace(
    "__EXTRA_LINKS__",
    "".join(f'<a href="/doc/{i}">Documentation Section {i}</a> ' for i in range(24)),
)

_STALE_REF_HTML = """<!doctype html><html><head><title>SPA Re-render Test</title></head><body>
  <main id="app">
    <div id="mount">
      <button id="primary-action" style="position:absolute;left:40px;top:40px;padding:12px 20px;">
        Confirm Deployment
      </button>
    </div>
    <label style="position:absolute;left:40px;top:240px;">
      <input id="agree-cb" type="checkbox" /> Enable Canary Rollout
    </label>
  </main>
</body></html>
"""


async def _run_legacy_perceive(page) -> dict[str, Any]:
    """Mirror legacy `perceive_node`: 1500ms static sleep + dom_parser + strict_viewport."""
    t_total = time.perf_counter()
    t_settle = time.perf_counter()
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=10000)
        await page.wait_for_timeout(1500)
    except Exception:
        pass
    settle_ms = round((time.perf_counter() - t_settle) * 1000.0, 1)

    t_ext = time.perf_counter()
    dom_data = await dom_parser.extract(page, timeout=5.0)
    raw_els = dom_data.get("elements", []) or []
    smap = {e["id"]: e for e in raw_els if e.get("id")}
    vw, vh = _viewport_of(page)
    els, _smap, md, _dropped, _n_off = strict_viewport_filter(
        raw_els, smap, dom_data.get("markdown", ""), vw=vw, vh=vh,
    )
    extract_ms = round((time.perf_counter() - t_ext) * 1000.0, 1)
    total_ms = round((time.perf_counter() - t_total) * 1000.0, 1)
    return {
        "mode": "legacy",
        "settle_ms": settle_ms,
        "extract_ms": extract_ms,
        "total_ms": total_ms,
        "raw_elements": len(raw_els),
        "emitted_elements": len(els),
        "occluded_filtered": 0,
        "markdown_chars": len(md),
        "est_tokens": estimate_tokens(md),
        "snapshot_kind": "full",
    }


async def _run_v2_perceive(page, compressor: V2SnapshotCompressor, step_num: int = 0) -> dict[str, Any]:
    metrics = RunMetrics(mode="v2")
    pr = await perceive_v2(
        page,
        ctx={"objective": "benchmark", "step_number": step_num},
        compressor=compressor,
        metrics=metrics,
        do_settle=True,
        use_mcp_snapshot=False,
    )
    st = metrics.steps[-1]
    return {
        "mode": "v2",
        "settle_ms": st.settle_ms,
        "extract_ms": st.extract_ms,
        "total_ms": st.total_perceive_ms,
        "raw_elements": st.raw_elements,
        "emitted_elements": pr.element_count,
        "occluded_filtered": st.occluded_filtered,
        "markdown_chars": st.markdown_chars,
        "est_tokens": st.est_tokens,
        "snapshot_kind": st.snapshot_kind,
    }


async def _benchmark_stale_ref_recovery(browser) -> dict[str, Any]:
    """Compare legacy blind coordinate click vs V2 stale-ref semantic rebind."""
    page = await browser.new_page(viewport={"width": 1000, "height": 700})
    try:
        await page.set_content(_STALE_REF_HTML)
        snap = await dom_parser.extract(page)
        target = next(e for e in snap["elements"] if "Confirm Deployment" in e["text"])

        # Simulate SPA re-render: old node detached, new button rendered 320px right & 100px down
        await page.evaluate("""() => {
            const m = document.getElementById('mount');
            m.innerHTML = '<button id="primary-action-v2" '
                + 'style="position:absolute;left:360px;top:140px;padding:12px 20px;">'
                + 'Confirm Deployment</button>';
            window.__clickedNew = false;
            document.getElementById('primary-action-v2').addEventListener('click', () => { window.__clickedNew = true; });
        }""")

        # Legacy resolve_element fails on detached node → falls back to old (x, y) coords
        leg_res = await dom_parser.resolve_element(page, target["id"])
        legacy_stale_resolved = bool(leg_res.get("ok"))

        # V2 click detects detached ref, semantically rebinds to live button, and clicks fresh coords
        metrics = RunMetrics(mode="v2")

        async def _click_coords(x=0, y=0, element_id=None):
            await page.mouse.click(x, y)
            return {"success": True, "strategy": "mouse", "navigated": False, "dom_changed": True, "error": ""}

        v2_res = await v2_click(
            page,
            x=target["x"],
            y=target["y"],
            element_id=target["id"],
            elements_list=snap["elements"],
            metrics=metrics,
            click_fn=_click_coords,
        )
        clicked_new = await page.evaluate("() => !!window.__clickedNew")
        return {
            "legacy_resolved_after_rerender": legacy_stale_resolved,
            "v2_click_success": bool(v2_res.get("success")),
            "v2_clicked_rerendered_button": bool(clicked_new),
            "v2_stale_ref_recoveries": metrics.total_stale_ref_recoveries,
        }
    finally:
        await page.close()


async def run_local_suite(headless: bool = True) -> dict[str, Any]:
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=headless)
    rows: list[dict[str, Any]] = []
    try:
        # Scenario 1: Modal-occluded storefront
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.set_content(_MODAL_OCCLUSION_HTML)
        leg1 = await _run_legacy_perceive(page)
        v2_1 = await _run_v2_perceive(page, V2SnapshotCompressor())
        rows.append({"scenario": "local:modal_occlusion", "legacy": leg1, "v2": v2_1})
        await page.close()

        # Scenario 2: Multi-step SPA form (Step 1 full, Step 2 after input mutation → delta!)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.set_content(_SPA_MULTI_STEP_HTML)
        comp = V2SnapshotCompressor()
        leg_s1 = await _run_legacy_perceive(page)
        v2_s1 = await _run_v2_perceive(page, comp, step_num=0)
        rows.append({"scenario": "local:spa_form_step1_full", "legacy": leg_s1, "v2": v2_s1})

        await page.fill("#f-name", "Sandeep")
        leg_s2 = await _run_legacy_perceive(page)
        v2_s2 = await _run_v2_perceive(page, comp, step_num=1)
        rows.append({"scenario": "local:spa_form_step2_delta", "legacy": leg_s2, "v2": v2_s2})
        await page.close()

        # Scenario 3: SPA re-render stale-ref recovery
        stale_report = await _benchmark_stale_ref_recovery(browser)
    finally:
        await browser.close()
        await pw.stop()

    return {"driver": detect_stealth_driver(), "scenarios": rows, "stale_ref_test": stale_report}


async def run_live_urls(urls: list[str], steps_per_url: int = 2, headless: bool = True) -> dict[str, Any]:
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=headless)
    rows: list[dict[str, Any]] = []
    try:
        for url in urls:
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=25000)
                comp = V2SnapshotCompressor()
                for step_idx in range(steps_per_url):
                    label = f"{url} [step {step_idx + 1}]"
                    leg = await _run_legacy_perceive(page)
                    v2 = await _run_v2_perceive(page, comp, step_num=step_idx)
                    rows.append({"scenario": label, "legacy": leg, "v2": v2})
            except Exception as e:
                rows.append({"scenario": url, "error": str(e)[:160]})
            finally:
                await page.close()
    finally:
        await browser.close()
        await pw.stop()

    return {"driver": detect_stealth_driver(), "scenarios": rows}


def run_e2e_side_by_side(objective: str, headless: bool = False) -> dict[str, Any]:
    """Run full LangGraph `run_v16.py` sequentially in legacy mode and v2 mode."""
    py = sys.executable
    results: dict[str, Any] = {"objective": objective, "runs": {}}

    for mode in ("legacy", "v2"):
        cmd = [py, str(SRC / "run_v16.py"), "run", "--mode", mode]
        if headless:
            cmd.append("--headless")
        cmd.append(objective)

        env = {**os.environ, "AGENT_ENGINE_MODE": mode}
        print(f"\n{'=' * 72}\n▶ Running E2E [{mode.upper()}]: {' '.join(cmd[:-1])} <objective>\n{'=' * 72}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        elapsed_s = round(time.perf_counter() - t0, 2)
        results["runs"][mode] = {
            "exit_code": proc.returncode,
            "wall_seconds": elapsed_s,
        }

    return results


def _print_comparison_table(report: dict[str, Any]) -> None:
    drv = report.get("driver", {})
    print("\n" + "=" * 96)
    print(f" Agent-first-browse Side-by-Side Benchmark (LEGACY vs V2) | driver={drv.get('driver', '?')}")
    print("=" * 96)
    header = (
        f"{'Scenario':<32} | {'Leg ms':>7} {'V2 ms':>7} {'Speedup':>7} | "
        f"{'Leg Tok':>7} {'V2 Tok':>7} {'Saved':>7} | {'Occl':>4} {'Kind':<6}"
    )
    print(header)
    print("-" * len(header))

    for row in report.get("scenarios", []):
        sc = row["scenario"][:32]
        if "error" in row:
            print(f"{sc:<32} | ERROR: {row['error']}")
            continue
        leg = row["legacy"]
        v2 = row["v2"]
        speedup = (leg["total_ms"] / v2["total_ms"]) if v2["total_ms"] > 0 else 0.0
        tok_saved = (
            (1.0 - (v2["est_tokens"] / leg["est_tokens"])) * 100.0
            if leg["est_tokens"] > 0
            else 0.0
        )
        print(
            f"{sc:<32} | {leg['total_ms']:>7.1f} {v2['total_ms']:>7.1f} {speedup:>6.1f}x | "
            f"{leg['est_tokens']:>7d} {v2['est_tokens']:>7d} {tok_saved:>6.1f}% | "
            f"{v2['occluded_filtered']:>4d} {v2['snapshot_kind']:<6}"
        )

    stale = report.get("stale_ref_test")
    if stale:
        print("-" * len(header))
        print(
            f"Stale-Ref Recovery Test: legacy_resolved={stale['legacy_resolved_after_rerender']} | "
            f"v2_clicked_rerendered_button={stale['v2_clicked_rerendered_button']} "
            f"(recoveries={stale['v2_stale_ref_recoveries']})"
        )
    print("=" * 96 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Side-by-side live benchmark runner comparing Legacy vs V2 browser engine."
    )
    parser.add_argument(
        "--suite",
        choices=["local"],
        help="Run built-in local Chromium stress suite (modal occlusion, delta form, stale-ref recovery).",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="One or more live URLs to benchmark side-by-side (legacy vs v2 perception).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of consecutive perception steps per live URL (default: 2 to test delta compression).",
    )
    parser.add_argument(
        "--e2e",
        type=str,
        help="Run full LangGraph run_brain end-to-end on the same objective in legacy then v2 mode.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Launch visible headed browser instead of headless (for --suite/--urls).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode during --e2e runs.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        help="Optional path to save full JSON benchmark results.",
    )

    args = parser.parse_args()
    if not (args.suite or args.urls or args.e2e):
        args.suite = "local"

    report: dict[str, Any] = {}
    if args.suite == "local":
        report = asyncio.run(run_local_suite(headless=not args.headed))
        _print_comparison_table(report)
    elif args.urls:
        report = asyncio.run(run_live_urls(args.urls, steps_per_url=max(1, args.steps), headless=not args.headed))
        _print_comparison_table(report)
    elif args.e2e:
        report = run_e2e_side_by_side(args.e2e, headless=args.headless)
        print("\n=== E2E SIDE-BY-SIDE SUMMARY ===")
        print(json.dumps(report, indent=2))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
