"""Unit + integration tests for V19 Perception & Grounding Precision.

Guarantees under test:
  - Every extracted element gets a live handle in window.__aid; resolve_element
    returns FRESH center coords for the exact node (drift-proof), scrolling it
    into view first.
  - A detached / unknown id resolves to {ok: False} → callers fall back to
    snapshot coordinates (strict superset, no regression).
  - Repeated-label controls ("Buy", "comments") get DISTINCT disambiguation
    hints (href + row context) so the LLM can pick the right one.
  - mcp_ground_action tries the registry (Layer 0) before the 60px-nearest snap,
    and falls through cleanly when the registry misses.

Run: .venv/bin/python -m pytest test_perception_v19.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import dom_parser
import ghost_input
import mcp_tools


# A deterministic, network-free fixture: a tall page (forces scroll), three rows
# each with an identical "Buy" label but a unique product + href, plus a button
# we can delete to test the detached path.
FIXTURE_HTML = """
<!doctype html><html><body>
<div style="height:900px">spacer</div>
<ul>
  <li>Alpha Widget <a href="/buy?id=1">Buy</a></li>
  <li>Beta Gadget <a href="/buy?id=2">Buy</a></li>
  <li>Gamma Device <a href="/buy?id=3">Buy</a></li>
</ul>
<button id="killme" type="button">Remove Me</button>
<div style="height:400px">spacer2</div>
</body></html>
"""


async def _new_page():
    from playwright.async_api import async_playwright
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)
    page = await browser.new_page(viewport={"width": 1280, "height": 800})
    await page.set_content(FIXTURE_HTML, wait_until="domcontentloaded")
    return pw, browser, page


# ═══════════════════════════════════════════════════════════════════════════════
#  Live Playwright tests (the real proof)
# ═══════════════════════════════════════════════════════════════════════════════

def test_repeated_labels_get_distinct_hints():
    async def run():
        pw, browser, page = await _new_page()
        try:
            data = await dom_parser.extract(page)
            buys = [e for e in data["elements"] if (e["text"] or "").strip() == "Buy"]
            assert len(buys) == 3, [e["text"] for e in data["elements"]]
            hints = [e.get("hint", "") for e in buys]
            # Each Buy must carry its unique href, and all three hints differ.
            assert all("/buy?id=" in h for h in hints), hints
            assert len(set(hints)) == 3, hints
            # Row context should surface the product name for at least one.
            assert any("Alpha" in h or "Beta" in h or "Gamma" in h for h in hints), hints
        finally:
            await browser.close(); await pw.stop()
    asyncio.run(run())


def test_registry_resolves_fresh_coords_after_scroll():
    async def run():
        pw, browser, page = await _new_page()
        try:
            data = await dom_parser.extract(page)
            # Pick the last "Buy" (furthest down → likely off-screen initially).
            buys = [e for e in data["elements"] if (e["text"] or "").strip() == "Buy"]
            target = buys[-1]["id"]
            r = await dom_parser.resolve_element(page, target)
            assert r["ok"] is True, r
            assert r["tag"] == "A"
            # After scrollIntoView(center), the element is within the viewport.
            assert r["onscreen"] is True, r
            assert isinstance(r["x"], int) and isinstance(r["y"], int)
        finally:
            await browser.close(); await pw.stop()
    asyncio.run(run())


def test_detached_node_returns_not_ok():
    async def run():
        pw, browser, page = await _new_page()
        try:
            data = await dom_parser.extract(page)
            # Find the registered id for #killme, then remove it from the DOM.
            kill_id = None
            for e in data["elements"]:
                r = await dom_parser.resolve_element(page, e["id"])
                if r.get("ok") and "remove me" in (r.get("text", "") or "").lower():
                    kill_id = e["id"]; break
            assert kill_id is not None, "killme button not extracted"
            await page.evaluate("() => document.getElementById('killme').remove()")
            r2 = await dom_parser.resolve_element(page, kill_id)
            assert r2["ok"] is False
            assert r2["reason"] in ("detached", "not_in_registry"), r2
        finally:
            await browser.close(); await pw.stop()
    asyncio.run(run())


def test_unknown_id_misses_gracefully():
    async def run():
        pw, browser, page = await _new_page()
        try:
            await dom_parser.extract(page)
            r = await dom_parser.resolve_element(page, "e9999")
            assert r == {"ok": False, "reason": "not_in_registry"}
            r2 = await dom_parser.resolve_element(page, "")
            assert r2["ok"] is False
        finally:
            await browser.close(); await pw.stop()
    asyncio.run(run())


# ═══════════════════════════════════════════════════════════════════════════════
#  V19.1 — Primary-action recall (the Flipkart "Add to Cart not found" fix)
# ═══════════════════════════════════════════════════════════════════════════════

# A dense product-page shape: the real ADD TO CART / BUY NOW buttons sit in a
# top action column, surrounded by 150 decoy rows whose link text merely CONTAINS
# "buy now". After scrolling down (as the agent does hunting for the button) the
# action buttons are ABOVE the viewport — exactly where the old off-screen cull
# discarded them before the LLM could ever see them.
DENSE_PRODUCT_HTML = """
<!doctype html><html><body>
<div style="position:absolute;left:20px;top:300px">
  <button class="add">ADD TO CART</button>
  <button class="buy">BUY NOW</button>
</div>
<div style="height:1200px">price, offers, EMI, bank offers...</div>
<ul>__ROWS__</ul>
<div style="height:2000px">specifications, reviews...</div>
</body></html>
""".replace(
    "__ROWS__",
    "".join(
        f'<li><a href="/p/item{i}">Similar Product {i} buy now offer deal {i}</a>'
        f'<button>View {i}</button></li>'
        for i in range(150)
    ),
)


def test_offscreen_action_buttons_are_rescued():
    async def run():
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        try:
            await page.set_content(DENSE_PRODUCT_HTML, wait_until="domcontentloaded")
            # Scroll so the action buttons are ABOVE the viewport (the failure case).
            await page.evaluate("window.scrollTo(0, 1400)")
            data = await dom_parser.extract(page)
            els = data["elements"]
            texts = [(e["text"] or "").upper() for e in els]
            assert any("ADD TO CART" in t for t in texts), texts[:20]
            assert any("BUY NOW" in t for t in texts), texts[:20]
            # Budget must stay bounded — the boost reorders, it must NOT flood the
            # map with the 150 off-screen "buy now" decoy links.
            assert len(els) <= 90, len(els)
            decoys = [t for t in texts if "SIMILAR PRODUCT" in t]
            assert len(decoys) < 60, f"decoy flooding: {len(decoys)}"
            # The rescued off-screen button resolves to fresh, scrolled-in coords.
            cart = next(e for e in els if "ADD TO CART" in (e["text"] or "").upper())
            r = await dom_parser.resolve_element(page, cart["id"])
            assert r["ok"] is True and r["onscreen"] is True, r
            assert r["tag"] == "BUTTON", r
        finally:
            await browser.close(); await pw.stop()
    asyncio.run(run())


def test_div_rendered_action_bar_is_captured_and_deduped():
    """The real Flipkart case: 'Add to cart' / 'Buy at ₹…' are styled <DIV>s in a
    fixed bar (NOT <button>), wrapped in several nested divs. They must be
    surfaced as clickable buttons, and the nested clones collapsed to ~one entry
    per physical button so the element budget isn't flooded."""
    async def run():
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 1080})
        try:
            await page.set_content(
                '<body style="margin:0;height:1080px">'
                '<div style="height:900px">product images, price, highlights</div>'
                '<div style="position:fixed;bottom:0;left:0;width:100%;height:56px;display:flex">'
                '  <div style="flex:1"><div><span>Add to cart</span></div></div>'
                '  <div style="flex:1"><div><span>Buy at ₹180</span></div></div>'
                '</div></body>',
                wait_until="domcontentloaded",
            )
            data = await dom_parser.extract(page)
            atc = [e for e in data["elements"]
                   if (e["text"] or "").strip().lower().startswith("add to cart")]
            buy = [e for e in data["elements"] if "buy at" in (e["text"] or "").lower()]
            # Surfaced at all (this is what was failing on live Flipkart) ...
            assert atc, [e["text"] for e in data["elements"]]
            assert buy, [e["text"] for e in data["elements"]]
            # ... as clickable buttons ...
            assert atc[0]["kind"] == "button", atc
            # ... and deduped: nested clones of one button collapse (not ~7 copies).
            assert len(atc) <= 3, [(e["id"], e["x"], e["y"]) for e in atc]
            # The chosen button resolves via the V19 registry.
            r = await dom_parser.resolve_element(page, atc[0]["id"])
            assert r["ok"] is True, r
        finally:
            await browser.close(); await pw.stop()
    asyncio.run(run())


def test_long_promo_text_not_treated_as_action():
    """A long <a> whose text merely contains 'buy now' must NOT be rescued when
    off-screen (only short, button-like primary actions are)."""
    async def run():
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        try:
            await page.set_content(
                '<div style="height:1600px">spacer</div>'
                '<a href="/x" style="position:absolute;top:50px">'
                'Limited time offer — buy now and save big on everything today</a>',
                wait_until="domcontentloaded",
            )
            await page.evaluate("window.scrollTo(0, 1500)")  # link is far above
            data = await dom_parser.extract(page)
            assert not any("Limited time offer" in (e["text"] or "") for e in data["elements"])
        finally:
            await browser.close(); await pw.stop()
    asyncio.run(run())


# ═══════════════════════════════════════════════════════════════════════════════
#  V19.2 — Scroll actually moves the viewport (the Flipkart "scroll forever" fix)
# ═══════════════════════════════════════════════════════════════════════════════

# A Flipkart-shaped layout: a FIXED left column with its own overflow scroller
# (the product image/buttons panel) over a tall main body. If the cursor sits on
# the left column — where a prior click leaves it — a raw mouse-wheel is swallowed
# by that column and the window never moves, so perception never advances.
STICKY_COLUMN_HTML = """
<body style="margin:0;height:5000px">
  <div id="sticky" style="position:fixed;left:0;top:0;width:300px;height:100%;overflow:auto">
    <div style="height:4000px">sticky column</div>
  </div>
  <div style="margin-left:320px;height:5000px">main body content</div>
</body>
"""


def test_ghost_scroll_moves_viewport_despite_wheel_swallow():
    async def run():
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        try:
            await page.set_content(STICKY_COLUMN_HTML, wait_until="domcontentloaded")
            # Cursor parked over the sticky column (the post-click position that
            # swallows the wheel and caused the "scroll forever" stall).
            await page.mouse.move(60, 400)
            assert await page.evaluate("() => window.scrollY") == 0
            await ghost_input.ghost_scroll(page, 600)
            moved = await page.evaluate("() => window.scrollY")
            # The viewport must actually advance, not stay pinned at the top.
            assert moved >= 400, f"viewport did not move (scrollY={moved})"
        finally:
            await browser.close(); await pw.stop()
    asyncio.run(run())


# ═══════════════════════════════════════════════════════════════════════════════
#  Grounding Layer-0 precedence (mocked — no browser)
# ═══════════════════════════════════════════════════════════════════════════════

class _DummyPage:
    url = "about:blank"
    async def evaluate(self, *a, **k):
        raise AssertionError("Layer 0 should have returned before any page.evaluate")


def test_ground_action_prefers_registry(monkeypatch):
    async def fake_resolve(page, eid, timeout=2.0):
        return {"ok": True, "x": 111, "y": 222, "tag": "BUTTON", "text": "Submit"}
    monkeypatch.setattr(dom_parser, "resolve_element", fake_resolve)
    mcp_tools.set_page(_DummyPage())

    out = asyncio.run(mcp_tools.mcp_ground_action(
        element_id="e5", x=None, y=None,
        selector_map={"e5": {"x": 999, "y": 999}},  # must be IGNORED in favor of fresh
        elements_list=[],
    ))
    assert out["grounded"] is True
    assert (out["x"], out["y"]) == (111.0, 222.0)
    assert "registry" in out["reason"]


def test_ground_action_falls_through_on_registry_miss(monkeypatch):
    async def fake_miss(page, eid, timeout=2.0):
        return {"ok": False, "reason": "not_in_registry"}
    monkeypatch.setattr(dom_parser, "resolve_element", fake_miss)
    mcp_tools.set_page(_DummyPage())

    # Registry misses → Layer 1 selector_map resolution (no page.evaluate needed).
    out = asyncio.run(mcp_tools.mcp_ground_action(
        element_id="e5", x=None, y=None,
        selector_map={"e5": {"x": 50, "y": 60}},
        elements_list=[],
    ))
    assert out["grounded"] is True
    assert (out["x"], out["y"]) == (50.0, 60.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
