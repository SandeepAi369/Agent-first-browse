"""MCP Tool Wrappers — Model Context Protocol interface to existing browser primitives.

Wraps the battle-tested execution layer (cdp_click, cdp_input, ghost_input,
dom_parser, overlay_detector) as MCP-compatible tool functions.

These tools are the ONLY way worker nodes interact with the browser.
They never commit state — they return results that Overwatch validates.

Design:
  - Each tool delegates to an existing, proven module
  - Tool results are structured dicts, not raw strings
  - Page reference is injected at graph construction time
  - All tools are async for Playwright compatibility

References:
  - MCP (Anthropic, Nov 2024): standardized agent-to-tool protocol
  - Playwright MCP: accessibility-tree-first interaction pattern
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

try:
    from app.logger import get_logger
    logger = get_logger("mcp_tools")
except ImportError:
    logger = logging.getLogger("mcp_tools")

# ═══════════════════════════════════════════════════════════════════════════════
#  Tool Registry — holds the live page reference
# ═══════════════════════════════════════════════════════════════════════════════

_PAGE = None  # Set by brain_graph.py at startup
_SELECTOR_MAP: dict[str, dict] = {}


def set_page(page) -> None:
    """Inject the live Playwright page reference (called once at startup)."""
    global _PAGE
    _PAGE = page


def set_selector_map(smap: dict) -> None:
    """Update the current selector map (called each perception cycle)."""
    global _SELECTOR_MAP
    _SELECTOR_MAP = smap


def _get_page():
    if _PAGE is None:
        raise RuntimeError("MCP tools not initialized — call set_page() first")
    return _PAGE


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Navigate
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_navigate(url: str) -> dict:
    """Navigate the browser to a URL.

    Delegates to: page.goto()
    Returns: {"success": bool, "url": str, "error": str}
    """
    page = _get_page()
    try:
        # Domain safety check (existing module)
        from execution_safety import is_domain_allowed
        if not is_domain_allowed(url):
            return {"success": False, "url": page.url, "error": f"Domain blocked: {url[:80]}"}

        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        # A fresh page resets the real mouse to (0,0) and the visual cursor is
        # hidden — re-place the arrow at the tracked position so it never flashes
        # at the top-left corner (which would mislead a vision screenshot).
        try:
            from ghost_input import resync_visual_cursor
            await resync_visual_cursor(page)
        except Exception:
            pass
        return {"success": True, "url": page.url, "error": ""}
    except Exception as e:
        logger.warning("mcp_navigate failed: %s", e)
        return {"success": False, "url": page.url, "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Click
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_click(
    x: float, y: float,
    element_id: str | None = None,
) -> dict:
    """Click an element at (x, y) with overlay penetration and CDP resilience.

    Delegates to:
      1. overlay_detector.smart_click_with_penetration()
      2. ghost_input.ghost_move_to() — Bézier curve humanization
      3. cdp_click.resilient_click() — 4-strategy waterfall

    Returns: {"success": bool, "strategy": str, "navigated": bool, "dom_changed": bool, "error": str}
    """
    page = _get_page()

    try:
        # V19: Resolve the chosen element to FRESH, identity-verified coords from
        # the exact node the LLM picked (drift-proof). Falls back to snapshot
        # coords if the registry has no live node for this id.
        if element_id:
            import dom_parser
            r = await dom_parser.resolve_element(page, element_id)
            if r.get("ok"):
                x, y = float(r["x"]), float(r["y"])
                logger.info("🎯 Resolved %s → fresh (%d,%d) [%s '%s']",
                            element_id, int(x), int(y), r.get("tag", ""),
                            (r.get("text", "") or "")[:25])
                await asyncio.sleep(0.2)  # let scroll settle
            else:
                logger.debug("resolve %s miss (%s) — using snapshot coords",
                             element_id, r.get("reason"))

        # Step 1: Overlay penetration
        from overlay_detector import smart_click_with_penetration
        penetration = await smart_click_with_penetration(page, x, y)
        if penetration.get("overlay_bypassed"):
            logger.info("🎯 Overlay penetrated at (%d, %d) via %s", x, y, penetration.get("method", "?"))
            await asyncio.sleep(0.1)

        # Step 2: Humanized mouse movement
        from ghost_input import ghost_move_to
        await ghost_move_to(page, x, y)

        # Step 3: CDP resilient click
        from cdp_click import resilient_click
        click_result = await asyncio.wait_for(
            resilient_click(page, x, y, max_retries=4, settle_ms=800),
            timeout=30.0,
        )

        if click_result.success:
            return {
                "success": True,
                "strategy": click_result.strategy,
                "navigated": click_result.navigation,
                "dom_changed": click_result.dom_changed,
                "error": "",
            }
        else:
            # ── V32 SAFETY NET: element-id state verification ──────────────
            # If the coordinate-based Visual Truth Override in cdp_click.py
            # missed the state change (e.g. elementFromPoint returned the
            # label instead of the input), use the element_id registry to
            # check the actual element's state directly.
            # ───────────────────────────────────────────────────────────────
            if element_id:
                try:
                    state_check = await page.evaluate("""
                    (eid) => {
                        const el = document.querySelector(`[__aid="${eid}"]`);
                        if (!el) return {found: false};
                        return {
                            found: true,
                            checked: !!el.checked,
                            selected: !!el.selected,
                            value: (el.value || '').slice(0, 50),
                            ariaChecked: el.getAttribute('aria-checked') || '',
                        };
                    }
                    """, element_id)
                    if state_check.get("found") and (
                        state_check.get("checked") or
                        state_check.get("ariaChecked") == "true"
                    ):
                        logger.info(
                            "✅ V32 SAFETY NET: click engine reported failure but "
                            "element %s state is checked=%s. Overriding to success.",
                            element_id, state_check.get("checked"),
                        )
                        return {
                            "success": True,
                            "strategy": "element_state_verified",
                            "navigated": False,
                            "dom_changed": False,
                            "error": "",
                        }
                except Exception as e:
                    logger.debug("V32 safety net check failed: %s", e)

            return {
                "success": False,
                "strategy": "",
                "navigated": False,
                "dom_changed": False,
                "error": f"Click ineffective: {click_result.error[:80]}",
            }

    except asyncio.TimeoutError:
        return {"success": False, "strategy": "", "navigated": False,
                "dom_changed": False, "error": "Click timed out (30s)"}
    except Exception as e:
        logger.warning("mcp_click failed: %s", e)
        return {"success": False, "strategy": "", "navigated": False,
                "dom_changed": False, "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Type
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_type(
    text: str,
    x: float, y: float,
    element_id: str | None = None,
    clear_first: bool = True,
) -> dict:
    """Type text into an element with CDP resilient typing.

    Delegates to: cdp_input.resilient_type()
    Returns: {"success": bool, "strategy": str, "actual_length": int, "error": str}
    """
    page = _get_page()

    try:
        # V19: Resolve to fresh, identity-verified coords from the chosen node.
        if element_id:
            import dom_parser
            r = await dom_parser.resolve_element(page, element_id)
            if r.get("ok"):
                x, y = float(r["x"]), float(r["y"])
                logger.info("🎯 Resolved %s → fresh (%d,%d) [%s '%s']",
                            element_id, int(x), int(y), r.get("tag", ""),
                            (r.get("text", "") or "")[:25])
                await asyncio.sleep(0.2)  # let scroll settle
            else:
                logger.debug("resolve %s miss (%s) — using snapshot coords",
                             element_id, r.get("reason"))

        from cdp_input import resilient_type
        type_result = await asyncio.wait_for(
            resilient_type(
                page, text, x=x, y=y,
                clear_first=clear_first, max_retries=3,
            ),
            timeout=60.0,
        )

        if type_result["success"]:
            return {
                "success": True,
                "strategy": type_result["strategy"],
                "actual_length": type_result["actual_length"],
                "error": "",
            }
        else:
            return {
                "success": False,
                "strategy": "",
                "actual_length": 0,
                "error": "Type unverified after all strategies",
            }

    except asyncio.TimeoutError:
        return {"success": False, "strategy": "", "actual_length": 0, "error": "Type timed out (60s)"}
    except Exception as e:
        logger.warning("mcp_type failed: %s", e)
        return {"success": False, "strategy": "", "actual_length": 0, "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Scroll
# ═══════════════════════════════════════════════════════════════════════════════

_SCROLL_METRICS_JS = """
() => {
  const se = document.scrollingElement || document.documentElement || document.body;
  return {
    y: Math.round(window.scrollY || (se && se.scrollTop) || 0),
    h: Math.round((se && se.scrollHeight) || (document.body && document.body.scrollHeight) || 0),
    vh: Math.round(window.innerHeight || (se && se.clientHeight) || 0),
  };
}
"""


async def mcp_scroll(pixels: int = 600) -> dict:
    """Scroll the page down WITH feedback (V29 smart scroll).

    Delegates the humanized scroll to ghost_input.ghost_scroll (FROZEN — unchanged)
    and only MEASURES the viewport before/after, so the brain knows whether the page
    actually moved and whether it has reached the bottom. This is what lets the agent
    stop scrolling into a wall instead of looping.

    Returns: {success, error, scrolled_px, at_bottom, position, page_height, viewport_h}
    """
    page = _get_page()
    try:
        try:
            before = await page.evaluate(_SCROLL_METRICS_JS)
        except Exception:
            before = {}
        from ghost_input import ghost_scroll
        await asyncio.wait_for(ghost_scroll(page, pixels), timeout=10.0)
        try:
            after = await page.evaluate(_SCROLL_METRICS_JS)
        except Exception:
            after = {}
        by = int(before.get("y", 0) or 0)
        ay = int(after.get("y", by) or by)
        ph = int(after.get("h", before.get("h", 0)) or 0)
        vh = int(after.get("vh", before.get("vh", 0)) or 0)
        at_bottom = bool(ph and vh and (ay + vh >= ph - 4))
        return {"success": True, "error": "", "scrolled_px": ay - by,
                "at_bottom": at_bottom, "position": ay,
                "page_height": ph, "viewport_h": vh}
    except asyncio.TimeoutError:
        return {"success": False, "error": "Scroll timed out (10s)",
                "scrolled_px": 0, "at_bottom": False}
    except Exception as e:
        logger.warning("mcp_scroll failed: %s", e)
        return {"success": False, "error": str(e)[:200],
                "scrolled_px": 0, "at_bottom": False}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Press Enter
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_press_enter() -> dict:
    """Press the Enter key.

    Returns: {"success": bool, "error": str}
    """
    page = _get_page()
    try:
        await page.keyboard.press("Enter")
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_press_enter failed: %s", e)
        return {"success": False, "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════════════════
#  V29 Phase A — Expanded primitives (hover / select_option / press_key)
#  Universal handlers reusing existing backends (ghost_move_to, keyboard, __aid).
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_press_key(key: str) -> dict:
    """Press a single key or chord (e.g. 'Escape', 'Tab', 'ArrowDown', 'Control+A').
    Reuses Playwright keyboard. Universal — Playwright validates the key name."""
    page = _get_page()
    key = (key or "").strip()
    if not key or len(key) > 40:
        return {"success": False, "error": "no/invalid key specified"}
    try:
        await page.keyboard.press(key)
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_press_key failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


async def mcp_hover(element_id: str | None = None, x: float = 0, y: float = 0) -> dict:
    """Hover over an element (reveal menus/tooltips/submenus). Resolves fresh
    coordinates via the V19 registry, then reuses the humanized ghost_move_to."""
    page = _get_page()
    try:
        if element_id:
            import dom_parser
            r = await dom_parser.resolve_element(page, element_id)
            if r.get("ok"):
                x, y = float(r["x"]), float(r["y"])
        from ghost_input import ghost_move_to
        await ghost_move_to(page, x, y)
        await asyncio.sleep(0.15)  # let hover-triggered UI settle
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_hover failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


_SELECT_OPTION_JS = r"""
(args) => {
  const el = window.__aid && window.__aid[args.id];
  if (!el) return {ok:false, reason:'no live node for id'};
  if ((el.tagName||'').toUpperCase() !== 'SELECT') return {ok:false, reason:'not a native select element'};
  const want = String(args.value||''), wl = want.toLowerCase();
  let m = null;
  for (const o of el.options) { if (o.value === want || (o.textContent||'').trim() === want) { m = o; break; } }
  if (!m) { for (const o of el.options) { if ((o.textContent||'').toLowerCase().includes(wl)) { m = o; break; } } }
  if (!m) return {ok:false, reason:'option not found: '+want.slice(0,40)};
  el.value = m.value;
  el.dispatchEvent(new Event('input', {bubbles:true}));
  el.dispatchEvent(new Event('change', {bubbles:true}));
  return {ok:true, selected:(m.textContent||'').trim().slice(0,40)};
}
"""


async def mcp_select_option(element_id: str | None, value: str) -> dict:
    """Choose an option in a NATIVE <select> by value or visible text (sets the
    value + dispatches input/change). Universal: works on any native dropdown via
    the __aid registry node; returns a clear reason if the target isn't a <select>
    (so the agent falls back to clicking a custom/ARIA dropdown)."""
    page = _get_page()
    if not element_id:
        return {"success": False, "error": "select_option requires an element_id"}
    try:
        res = await page.evaluate(_SELECT_OPTION_JS, {"id": element_id, "value": value or ""})
        if res.get("ok"):
            return {"success": True, "selected": res.get("selected", ""), "error": ""}
        return {"success": False, "error": res.get("reason", "select failed")}
    except Exception as e:
        logger.warning("mcp_select_option failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


# ═══════════════════════════════════════════════════════════════════════════════
#  V33: Comprehensive Browser Action Suite — New Primitives
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_drag_and_drop(
    element_id: str | None = None,
    x: float = 0, y: float = 0,
    target_x: float = 0, target_y: float = 0,
) -> dict:
    """Drag from (x, y) to (target_x, target_y) using CDP mouse events.

    Resolves source coordinates from element_id if available.
    Simulates: mousedown → slow mousemove → mouseup (with humanized Bézier path).
    Works for sliders, CAPTCHA puzzles, reorder lists, range inputs.
    """
    page = _get_page()
    try:
        # Resolve source from element registry if available
        if element_id:
            import dom_parser
            r = await dom_parser.resolve_element(page, element_id)
            if r.get("ok"):
                x, y = float(r["x"]), float(r["y"])

        if x == 0 and y == 0:
            return {"success": False, "error": "drag_and_drop requires source coordinates or element_id"}
        if target_x == 0 and target_y == 0:
            return {"success": False, "error": "drag_and_drop requires target_x, target_y coordinates"}

        # Humanized move to source
        from ghost_input import ghost_move_to
        await ghost_move_to(page, x, y)
        await asyncio.sleep(0.1)

        # CDP mousedown at source
        cdp = page.context.browser.contexts[0].pages[0]
        await cdp.evaluate("""
        ([x, y]) => {
            const el = document.elementFromPoint(x, y);
            if (el) {
                el.dispatchEvent(new MouseEvent('mousedown', {
                    bubbles: true, cancelable: true,
                    clientX: x, clientY: y, button: 0
                }));
            }
        }
        """, [x, y])
        await asyncio.sleep(0.15)

        # Humanized move to target (simulates dragging)
        # Move in steps for dragover events
        steps = 8
        for i in range(1, steps + 1):
            ix = x + (target_x - x) * i / steps
            iy = y + (target_y - y) * i / steps
            await page.mouse.move(ix, iy)
            await asyncio.sleep(0.03)

        # Dispatch dragover/mouseover at target
        await page.evaluate("""
        ([x, y]) => {
            const el = document.elementFromPoint(x, y);
            if (el) {
                el.dispatchEvent(new MouseEvent('mouseover', {
                    bubbles: true, clientX: x, clientY: y
                }));
                el.dispatchEvent(new MouseEvent('mousemove', {
                    bubbles: true, clientX: x, clientY: y
                }));
            }
        }
        """, [target_x, target_y])
        await asyncio.sleep(0.1)

        # CDP mouseup at target
        await page.mouse.up()

        # Also fire drop event for HTML5 drag-and-drop
        await page.evaluate("""
        ([x, y]) => {
            const el = document.elementFromPoint(x, y);
            if (el) {
                el.dispatchEvent(new MouseEvent('mouseup', {
                    bubbles: true, cancelable: true,
                    clientX: x, clientY: y, button: 0
                }));
            }
        }
        """, [target_x, target_y])

        logger.info("✅ drag_and_drop: (%.0f,%.0f) → (%.0f,%.0f)", x, y, target_x, target_y)
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_drag_and_drop failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


async def mcp_upload_file(element_id: str | None, file_path: str) -> dict:
    """Upload a file to an <input type='file'> element.

    Uses Playwright's set_input_files() which bypasses the OS file picker.
    Resolves element via the __aid registry.
    """
    page = _get_page()
    if not element_id:
        return {"success": False, "error": "upload_file requires an element_id"}
    if not file_path:
        return {"success": False, "error": "upload_file requires a file_path"}
    try:
        import os
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path[:80]}"}

        # Find the actual input[type=file] element using __aid
        locator = page.locator(f'[__aid="{element_id}"]')
        count = await locator.count()
        if count == 0:
            # Fallback: try data-eid
            locator = page.locator(f'[data-eid="{element_id}"]')
            count = await locator.count()
        if count == 0:
            return {"success": False, "error": f"Element {element_id} not found in DOM"}

        await locator.first.set_input_files(file_path)
        logger.info("✅ upload_file: '%s' → %s", os.path.basename(file_path), element_id)
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_upload_file failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


async def mcp_scroll_directional(direction: str = "down", amount: int = 500) -> dict:
    """Scroll the page in a specific direction by a given pixel amount.

    Supports: up, down, left, right. Uses page.mouse.wheel() for precision.
    For carousels, sidebars, infinite scrolls, and targeted navigation.
    """
    page = _get_page()
    direction = (direction or "down").lower().strip()
    amount = max(50, min(amount or 500, 5000))  # Clamp to [50, 5000]

    direction_map = {
        "down":  (0,  amount),
        "up":    (0, -amount),
        "right": (amount,  0),
        "left":  (-amount, 0),
    }

    if direction not in direction_map:
        return {"success": False, "error": f"Invalid direction '{direction}'. Use: up, down, left, right"}

    try:
        dx, dy = direction_map[direction]
        await page.mouse.wheel(dx, dy)
        await asyncio.sleep(0.3)  # Let scroll settle
        logger.info("✅ scroll_directional: %s by %dpx", direction, amount)
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_scroll_directional failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Snapshot (DOM Perception)
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_snapshot() -> dict:
    """Take an accessibility-tree snapshot of the current page.

    Delegates to: dom_parser.extract()
    Returns: {"elements": list, "markdown": str, "element_count": int, "selector_map": dict}
    """
    page = _get_page()
    try:
        import dom_parser
        dom_data = await dom_parser.extract(page, target_hint=None, timeout=5.0)
        elements_list = dom_data.get("elements", [])

        # Build selector map
        smap: dict[str, dict] = {}
        for el in elements_list:
            eid = el.get("id", el.get("ref", ""))
            if eid:
                smap[eid] = el

        # Update global selector map
        set_selector_map(smap)

        return {
            "elements": elements_list,
            "markdown": dom_data.get("markdown", ""),
            "element_count": dom_data.get("element_count", len(elements_list)),
            "selector_map": smap,
            "image_size": dom_data.get("image_size", {}),
        }
    except Exception as e:
        logger.warning("mcp_snapshot failed: %s", e)
        return {"elements": [], "markdown": "", "element_count": 0,
                "selector_map": {}, "image_size": {}}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Screenshot (for on-demand vision consults)
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_screenshot(full_page: bool = False) -> dict:
    """Capture a base64-encoded viewport screenshot.

    Used ONLY when the agent escalates to vision (it cannot resolve the page from
    the a11y DOM alone). Viewport — not full-page — so the image aligns with the
    element-map coordinates the vision model reasons over.

    Returns: {"ok": bool, "base64": str, "error": str}
    """
    page = _get_page()
    try:
        # page.screenshot captures ONLY the browser viewport (the web content
        # area) — never the browser chrome or the OS desktop. So vision is
        # inherently confined to the page; we also pass the exact bounds on so the
        # model knows the coordinate frame (0,0 .. width,height).
        png = await asyncio.wait_for(
            page.screenshot(full_page=full_page, type="png"), timeout=10.0
        )
        vp = page.viewport_size or {"width": 0, "height": 0}
        return {"ok": True, "base64": base64.b64encode(png).decode("utf-8"),
                "width": int(vp.get("width", 0)), "height": int(vp.get("height", 0)),
                "error": ""}
    except Exception as e:
        logger.warning("mcp_screenshot failed: %s", e)
        return {"ok": False, "base64": "", "width": 0, "height": 0, "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Wait
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_wait(ms: int = 800) -> dict:
    """Wait for a specified number of milliseconds.

    Returns: {"success": bool, "waited_ms": int}
    """
    page = _get_page()
    try:
        await page.wait_for_timeout(ms)
        return {"success": True, "waited_ms": ms}
    except Exception as e:
        return {"success": False, "waited_ms": 0}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Login State Detection
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_detect_login() -> dict:
    """Detect login state of the current page.

    Returns: {"logged_in": bool, "has_login_form": bool}
    """
    page = _get_page()
    try:
        login_state = await asyncio.wait_for(page.evaluate("""
        () => {
            const hasProfile = !!document.querySelector(
                '[aria-label*="profile" i], [aria-label*="account" i], '
                + 'img[alt*="avatar" i], img[alt*="profile" i], '
                + '[data-testid*="profile" i], [data-testid*="user" i], '
                + '.user-menu, .profile-menu, #user-nav'
            );
            let hasLoginForm = !!document.querySelector(
                'input[type="password"], '
                + 'form[action*="login" i], form[action*="signin" i]'
            );
            if (!hasLoginForm) {
                const buttons = document.querySelectorAll('button, a[role="button"], input[type="submit"]');
                for (const btn of buttons) {
                    const txt = (btn.textContent || '').trim().toLowerCase();
                    if (['sign in', 'log in', 'login', 'signin'].includes(txt)) {
                        hasLoginForm = true;
                        break;
                    }
                }
            }
            return { hasProfile, hasLoginForm };
        }
        """), timeout=5.0)

        return {
            "logged_in": login_state.get("hasProfile", False),
            "has_login_form": login_state.get("hasLoginForm", False),
        }
    except Exception:
        return {"logged_in": False, "has_login_form": False}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Grounding Validation
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_ground_action(
    element_id: str | None,
    x: float | None,
    y: float | None,
    selector_map: dict | None = None,
    elements_list: list | None = None,
) -> dict:
    """Validate and ground an action's target coordinates.

    Delegates to: advanced_agent._ground_or_reject() logic
    Returns: {"grounded": bool, "x": float, "y": float, "element": dict|None, "reason": str}
    """
    page = _get_page()
    smap = selector_map or _SELECTOR_MAP

    # Layer 0 (V19): Live registry resolution — the EXACT node the LLM chose, with
    # fresh coordinates. Supersedes the "snap to nearest within 60px" heuristic
    # that could land on the wrong neighbour on dense pages.
    if element_id is not None:
        import dom_parser
        r = await dom_parser.resolve_element(page, element_id)
        if r.get("ok"):
            return {"grounded": True, "x": float(r["x"]), "y": float(r["y"]),
                    "element": {"ref": element_id, "name": r.get("text", "")},
                    "reason": "registry resolved (fresh coords)"}

    # Layer 1: Element ID resolution
    if element_id is not None:
        el = smap.get(element_id)
        if el is None:
            return {"grounded": False, "x": 0, "y": 0, "element": None,
                    "reason": f"element_id '{element_id}' not in current snapshot"}
        resolved_x = float(el.get("x", 0))
        resolved_y = float(el.get("y", 0))
        return {"grounded": True, "x": resolved_x, "y": resolved_y,
                "element": el, "reason": "element_id resolved"}

    if x is None or y is None:
        return {"grounded": False, "x": 0, "y": 0, "element": None,
                "reason": "no element_id or coordinates"}

    # Layer 2: elementFromPoint hit-test
    try:
        hit = await page.evaluate(
            "([x,y])=>{const e=document.elementFromPoint(x,y);"
            "if(!e)return null;"
            "const tag=e.tagName.toUpperCase();"
            "const interactive=['A','BUTTON','INPUT','SELECT','TEXTAREA']"
            ".includes(tag)||e.getAttribute('role')==='button'"
            "||e.getAttribute('contenteditable')==='true';"
            "return{tag,interactive,text:(e.textContent||'').trim().slice(0,40)};}",
            [x, y],
        )
        if hit and hit.get("interactive"):
            return {"grounded": True, "x": x, "y": y,
                    "element": {"ref": "hit-test", "name": hit.get("text", "")[:30]},
                    "reason": "hit-test grounded"}
    except Exception:
        pass

    # Layer 3: Nearest-element snap
    elems = elements_list or []
    if not elems:
        return {"grounded": False, "x": x, "y": y, "element": None,
                "reason": "no elements in snapshot"}

    best_el = None
    best_dist = float("inf")
    for el in elems:
        ex = el.get("x", 0)
        ey = el.get("y", 0)
        if ex == 0 and ey == 0:
            continue
        dist = ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_el = el

    threshold = 60.0
    if best_el is None or best_dist > threshold:
        return {"grounded": False, "x": x, "y": y, "element": None,
                "reason": f"nearest element is {best_dist:.0f}px away (>{threshold:.0f}px)"}

    return {"grounded": True, "x": float(best_el.get("x", x)),
            "y": float(best_el.get("y", y)), "element": best_el,
            "reason": f"snapped to nearest ({best_dist:.0f}px)"}
