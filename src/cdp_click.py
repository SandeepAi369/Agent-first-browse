"""CDP-Native Click Engine — Multi-strategy click injection for anti-bot-protected sites.

Provides trusted click methods that bypass Amazon/Shopify/Reddit JS overlay traps
where Playwright's synthetic mouse.click() is silently dropped.

Click Strategy Waterfall:
  1. CDP Input.dispatchMouseEvent (trusted OS-level mouse events)
  2. CDP DOM.focus + JS element.click() (programmatic click via DOM)
  3. Playwright page.mouse.click() (legacy fallback)

Each strategy includes post-click verification to confirm the page responded.

Architecture:
  - overlay_detector.py runs BEFORE this module (neutralizes pointer-events traps)
  - ghost_input.py handles the Bézier mouse movement path
  - This module dispatches the actual click event at the destination

Why Playwright's mouse.click() fails on Amazon:
  Amazon's anti-bot JS listens for the `isTrusted` flag on MouseEvents.
  Playwright dispatches events via the Blink event queue, but certain sites
  filter events that don't originate from the browser's input pipeline.
  CDP's Input.dispatchMouseEvent goes through the browser's native input
  pipeline and produces events with isTrusted=true.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import Page

try:
    from app.logger import get_logger
    logger = get_logger("cdp_click")
except ImportError:
    import logging
    logger = logging.getLogger("cdp_click")


# ═══════════════════════════════════════════════════════════════════════════════
#  Result Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClickResult:
    """Result of a click attempt."""
    success: bool
    strategy: str           # Which strategy produced the click
    navigation: bool        # Did the page navigate after the click?
    dom_changed: bool       # Did the DOM structure change?
    attempts: int           # How many strategies were tried
    error: str = ""         # Error message if failed
    pre_url: str = ""       # URL before click
    post_url: str = ""      # URL after click


# ═══════════════════════════════════════════════════════════════════════════════
#  CDP Session Helper
# ═══════════════════════════════════════════════════════════════════════════════

async def _get_cdp_session(page: Page):
    """Create a CDP session from a Playwright page.
    
    Returns None on failure instead of raising — the caller must handle it.
    """
    try:
        return await page.context.new_cdp_session(page)
    except Exception as e:
        logger.warning("CDP session creation failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre/Post Click State Capture
# ═══════════════════════════════════════════════════════════════════════════════

async def _capture_page_fingerprint(page: Page, timeout: float = 2.0) -> dict:
    """Capture a lightweight fingerprint of the page for change detection.
    
    Returns:
        {
            'url': str,
            'title': str,
            'element_count': int,
            'structural_hash': str,  # MD5 of interactive element structure
        }
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        () => {
            const interactives = document.querySelectorAll(
                'a, button, input, textarea, select, [role="button"], [role="link"], [role="menuitem"]'
            );
            const parts = [];
            interactives.forEach((el, i) => {
                if (i < 100) {  // Cap at 100 elements for performance
                    const rect = el.getBoundingClientRect();
                    parts.push(
                        el.tagName + '|' +
                        (el.textContent || '').trim().slice(0, 20) + '|' +
                        Math.round(rect.x) + ',' + Math.round(rect.y)
                    );
                }
            });
            return {
                url: location.href,
                title: document.title,
                elementCount: interactives.length,
                structure: parts.join(';;')
            };
        }
        """), timeout=timeout)
        
        structure = result.get("structure", "")
        structural_hash = hashlib.md5(structure.encode()).hexdigest()[:12]
        
        return {
            "url": result.get("url", ""),
            "title": result.get("title", ""),
            "element_count": result.get("elementCount", 0),
            "structural_hash": structural_hash,
        }
    except Exception as e:
        logger.debug("Page fingerprint capture failed: %s", e)
        return {
            "url": "",
            "title": "",
            "element_count": 0,
            "structural_hash": "",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 1: CDP Input.dispatchMouseEvent (Trusted Native Events)
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_cdp_mouse_event(
    page: Page,
    x: float,
    y: float,
    button: str = "left",
) -> bool:
    """Dispatch a full mouse click via CDP Input.dispatchMouseEvent.
    
    This goes through the browser's native input pipeline and produces
    MouseEvents with isTrusted=true, which is the key property that
    anti-bot scripts check.
    
    Sequence: mouseMoved → mousePressed → mouseReleased
    
    Each event has realistic timing jitter and proper modifier flags.
    """
    cdp = await _get_cdp_session(page)
    if not cdp:
        return False
    
    try:
        # Map button name to CDP button code
        button_map = {"left": "left", "right": "right", "middle": "middle"}
        cdp_button = button_map.get(button, "left")
        click_count = 1
        
        # Realistic timestamp base (microseconds since epoch)
        ts = time.time()
        
        # Step 1: Mouse moved to position (generates mouseover/mouseenter events)
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": int(x),
            "y": int(y),
            "button": "none",
            "timestamp": ts,
        })
        await asyncio.sleep(random.uniform(0.02, 0.06))
        
        # Step 2: Mouse pressed (generates mousedown event)
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": int(x),
            "y": int(y),
            "button": cdp_button,
            "clickCount": click_count,
            "timestamp": ts + random.uniform(0.03, 0.08),
        })
        
        # Realistic press-to-release delay (humans hold for 50-150ms)
        await asyncio.sleep(random.uniform(0.05, 0.15))
        
        # Step 3: Mouse released (generates mouseup + click events)
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": int(x),
            "y": int(y),
            "button": cdp_button,
            "clickCount": click_count,
            "timestamp": ts + random.uniform(0.10, 0.20),
        })
        
        logger.debug("CDP mouse event click dispatched at (%d, %d)", x, y)
        return True
        
    except Exception as e:
        logger.warning("CDP mouse event failed at (%d, %d): %s", x, y, e)
        return False
    finally:
        try:
            await cdp.detach()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 2: CDP DOM.focus + JS element.click()
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_js_click(
    page: Page,
    x: float,
    y: float,
) -> bool:
    """Find the element at (x, y) and call its .click() method directly.
    
    This bypasses the mouse event pipeline entirely and triggers the
    element's click handler programmatically. Less realistic but more
    reliable when event listeners are filtered.
    
    Also dispatches synthetic PointerEvent and MouseEvent for frameworks
    that listen on those instead of the click handler.
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        (params) => {
            const { x, y } = params;
            const el = document.elementFromPoint(x, y);
            if (!el) return { clicked: false, reason: 'no_element_at_point' };
            
            const tag = el.tagName.toLowerCase();
            const text = (el.textContent || '').trim().slice(0, 40);
            
            try {
                // Step 1: Focus the element (if focusable)
                if (typeof el.focus === 'function') {
                    el.focus();
                }
                
                // Step 2: Dispatch a full event sequence
                // PointerEvent → MouseEvent → click
                const eventInit = {
                    bubbles: true,
                    cancelable: true,
                    view: window,
                    clientX: x,
                    clientY: y,
                    screenX: x,
                    screenY: y,
                    button: 0,
                    buttons: 1,
                };
                
                el.dispatchEvent(new PointerEvent('pointerdown', eventInit));
                el.dispatchEvent(new MouseEvent('mousedown', eventInit));
                el.dispatchEvent(new PointerEvent('pointerup', eventInit));
                el.dispatchEvent(new MouseEvent('mouseup', eventInit));
                el.dispatchEvent(new MouseEvent('click', eventInit));
                
                // Step 3: Also call .click() directly as a final guarantee
                el.click();
                
                return {
                    clicked: true,
                    tag: tag,
                    text: text,
                    reason: 'js_click_dispatched'
                };
            } catch (e) {
                return { clicked: false, reason: 'click_error: ' + e.message };
            }
        }
        """, {"x": x, "y": y}), timeout=3.0)
        
        if result and result.get("clicked"):
            logger.debug(
                "JS click dispatched at (%d, %d) on <%s> '%s'",
                x, y, result.get("tag", "?"), result.get("text", "")[:30],
            )
            return True
        else:
            reason = result.get("reason", "unknown") if result else "eval_returned_none"
            logger.warning("JS click failed at (%d, %d): %s", x, y, reason)
            return False
            
    except Exception as e:
        logger.warning("JS click failed at (%d, %d): %s", x, y, e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3: Playwright page.mouse.click() (Legacy Fallback)
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_playwright_click(
    page: Page,
    x: float,
    y: float,
) -> bool:
    """Standard Playwright mouse click — the existing fallback.
    
    This is what ghost_input.ghost_click() ultimately calls.
    Kept as the final fallback for sites that don't filter synthetic events.
    """
    try:
        await page.mouse.click(x, y)
        logger.debug("Playwright click dispatched at (%d, %d)", x, y)
        return True
    except Exception as e:
        logger.warning("Playwright click failed at (%d, %d): %s", x, y, e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 4: Navigate via href (for <a> elements)
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_direct_navigate(
    page: Page,
    x: float,
    y: float,
) -> bool:
    """If the element at (x, y) is an <a> tag, navigate directly to its href.
    
    This completely bypasses click event handling and is the most reliable
    way to follow links on heavily protected sites.
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        (params) => {
            const { x, y } = params;
            let el = document.elementFromPoint(x, y);
            if (!el) return { found: false };
            
            // Walk up to find the nearest <a> ancestor
            let current = el;
            for (let i = 0; i < 5 && current; i++) {
                if (current.tagName && current.tagName.toLowerCase() === 'a' && current.href) {
                    return {
                        found: true,
                        href: current.href,
                        text: (current.textContent || '').trim().slice(0, 40),
                        target: current.target || '_self',
                    };
                }
                current = current.parentElement;
            }
            return { found: false };
        }
        """, {"x": x, "y": y}), timeout=2.0)
        
        if result and result.get("found") and result.get("href"):
            href = result["href"]
            target = result.get("target", "_self")
            
            # Don't navigate to javascript: or # URLs
            if href.startswith("javascript:") or href == "#":
                logger.debug("Direct navigate skipped: href=%s", href[:60])
                return False
            
            logger.info(
                "🔗 Direct navigate to href: %s (text: '%s')",
                href[:80], result.get("text", "")[:30],
            )
            
            # Navigate in the current tab
            if target == "_blank":
                # For _blank targets, open in same tab to keep control
                await page.goto(href, wait_until="domcontentloaded", timeout=15000)
            else:
                await page.goto(href, wait_until="domcontentloaded", timeout=15000)
            
            return True
        
        return False
        
    except Exception as e:
        logger.warning("Direct navigate failed at (%d, %d): %s", x, y, e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  V32: Element State Capture (Visual Truth Override)
# ═══════════════════════════════════════════════════════════════════════════════

async def _capture_element_state(page: Page, x: float, y: float, timeout: float = 2.0) -> dict:
    """Capture the interactive state of the element at (x, y).

    Returns checked/selected/value/aria-* attributes for state-change detection
    on toggleable elements (radio buttons, checkboxes, dropdowns) where DOM
    structure doesn't change but element state does.

    This is the foundation of the Visual Truth Override — it lets us detect
    that a click WORKED even when the structural fingerprint sees no change.
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        ([x, y]) => {
            const el = document.elementFromPoint(x, y);
            if (!el) return {found: false};
            // Walk up to find the actual input/select (labels often wrap inputs)
            const input = el.closest(
                'input, select, textarea, [role="checkbox"], [role="radio"], '
                + '[role="switch"], [role="option"], [role="tab"]'
            ) || el;
            return {
                found: true,
                tag: input.tagName || '',
                type: (input.type || '').toLowerCase(),
                checked: !!input.checked,
                selected: !!input.selected,
                value: (input.value || '').slice(0, 100),
                ariaChecked: input.getAttribute('aria-checked') || '',
                ariaSelected: input.getAttribute('aria-selected') || '',
                ariaExpanded: input.getAttribute('aria-expanded') || '',
                ariaPressed: input.getAttribute('aria-pressed') || '',
                classList: Array.from(input.classList || []).join(' ').slice(0, 100),
            };
        }
        """, [x, y]), timeout=timeout)
        return result
    except Exception as e:
        logger.debug("Element state capture failed: %s", e)
        return {"found": False}


def _element_state_changed(pre: dict, post: dict) -> tuple[bool, list[str]]:
    """Compare pre/post element state. Returns (changed, list_of_changes).

    Checks standard DOM attributes that toggle on interaction:
    checked, selected, value, aria-checked, aria-selected, aria-expanded, aria-pressed.
    Completely generic — no website-specific logic.
    """
    if not pre.get("found") or not post.get("found"):
        return False, []

    changes = []
    for attr in ("checked", "selected", "value", "ariaChecked",
                 "ariaSelected", "ariaExpanded", "ariaPressed"):
        old = pre.get(attr)
        new = post.get(attr)
        if old != new:
            changes.append(f"{attr}: {old!r} → {new!r}")

    return len(changes) > 0, changes


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point: Resilient Click with Waterfall
# ═══════════════════════════════════════════════════════════════════════════════

async def resilient_click(
    page: Page,
    x: float,
    y: float,
    button: str = "left",
    max_retries: int = 4,
    settle_ms: int = 800,
) -> ClickResult:
    """Click at (x, y) using a multi-strategy waterfall with post-click verification.
    
    Strategy waterfall (tries each until the page responds):
      1. CDP Input.dispatchMouseEvent — trusted native events (best for anti-bot sites)
      2. JS element.click() with full event sequence — programmatic fallback
      3. Direct href navigation — for <a> tags that ignore click events
      4. Playwright mouse.click() — legacy fallback
    
    Post-click verification:
      After each strategy, waits settle_ms and checks if:
      - The URL changed (navigation)
      - The DOM structure changed (modal/dropdown/state change)
      If neither changed, the click was likely silently dropped → try next strategy.
    
    Args:
        page: Playwright Page instance
        x: X coordinate to click
        y: Y coordinate to click  
        button: Mouse button ('left', 'right', 'middle')
        max_retries: Maximum number of strategies to try
        settle_ms: Milliseconds to wait after click for page to respond
    
    Returns:
        ClickResult with success status, strategy used, and diagnostics
    """
    # Capture pre-click state
    pre_fp = await _capture_page_fingerprint(page)
    pre_url = pre_fp.get("url", "")

    # V32: Capture element state BEFORE clicking (for toggleable controls)
    pre_element_state = await _capture_element_state(page, x, y)
    
    strategies = [
        ("cdp_mouse_event", lambda: _strategy_cdp_mouse_event(page, x, y, button)),
        ("js_click", lambda: _strategy_js_click(page, x, y)),
        ("direct_navigate", lambda: _strategy_direct_navigate(page, x, y)),
        ("playwright_click", lambda: _strategy_playwright_click(page, x, y)),
    ]
    
    for attempt, (strategy_name, strategy_fn) in enumerate(strategies[:max_retries]):
        logger.info(
            "Click attempt %d/%d via %s at (%d, %d)",
            attempt + 1, max_retries, strategy_name, int(x), int(y),
        )
        
        try:
            # Execute the click strategy
            ok = await asyncio.wait_for(strategy_fn(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("%s timed out at (%d, %d)", strategy_name, int(x), int(y))
            continue
        except Exception as e:
            logger.warning("%s error at (%d, %d): %s", strategy_name, int(x), int(y), e)
            continue
        
        if not ok:
            logger.debug("%s returned False — trying next strategy", strategy_name)
            continue
        
        # Wait for the page to respond
        await asyncio.sleep(settle_ms / 1000.0)
        
        # Check for page load state
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass  # Page might not need full load for the click to have effect
        
        # Capture post-click state
        post_fp = await _capture_page_fingerprint(page)
        post_url = post_fp.get("url", "")
        
        # Determine if the click had any effect
        navigated = post_url != pre_url and post_url != ""
        dom_changed = (
            post_fp.get("structural_hash") != pre_fp.get("structural_hash")
            and post_fp.get("structural_hash") != ""
        )
        element_delta = abs(
            post_fp.get("element_count", 0) - pre_fp.get("element_count", 0)
        )
        
        click_had_effect = navigated or dom_changed or element_delta >= 2
        
        if click_had_effect:
            effect_desc = []
            if navigated:
                effect_desc.append(f"navigated to {post_url[:60]}")
            if dom_changed:
                effect_desc.append("DOM structure changed")
            if element_delta >= 2:
                effect_desc.append(f"element count changed by {element_delta}")
            
            logger.info(
                "✅ CLICK SUCCESS via %s at (%d, %d): %s",
                strategy_name, int(x), int(y), "; ".join(effect_desc),
            )
            return ClickResult(
                success=True,
                strategy=strategy_name,
                navigation=navigated,
                dom_changed=dom_changed,
                attempts=attempt + 1,
                pre_url=pre_url,
                post_url=post_url,
            )
        
        # ── V32 VISUAL TRUTH OVERRIDE ──────────────────────────────────────
        # For toggleable elements (radio buttons, checkboxes, custom toggles),
        # clicking changes ONLY the element's state attributes (checked,
        # selected, value, aria-*) — NOT the DOM structure, URL, or element
        # count. The fingerprint hash is blind to these changes.
        #
        # CRITICAL: We check here INSIDE the loop, BEFORE trying the next
        # strategy. If we let strategy 2 fire on a checkbox that strategy 1
        # already toggled, it will UNDO the toggle (double-click problem).
        # ──────────────────────────────────────────────────────────────────
        if pre_element_state.get("found"):
            post_element_state = await _capture_element_state(page, x, y)
            changed, change_list = _element_state_changed(
                pre_element_state, post_element_state
            )
            if changed:
                logger.info(
                    "✅ VISUAL TRUTH OVERRIDE via %s at (%d, %d): "
                    "click appeared to fail but element state changed: %s",
                    strategy_name, int(x), int(y),
                    "; ".join(change_list),
                )
                return ClickResult(
                    success=True,
                    strategy=f"{strategy_name}+state_verified",
                    navigation=False,
                    dom_changed=False,
                    attempts=attempt + 1,
                    pre_url=pre_url,
                    post_url=pre_url,
                )
        
        # Click dispatched but no visible effect — try next strategy
        logger.warning(
            "⚠️ %s dispatched at (%d, %d) but page did not respond — trying next",
            strategy_name, int(x), int(y),
        )
    
    # All strategies exhausted
    logger.error(
        "❌ CLICK FAILED: all %d strategies exhausted at (%d, %d)",
        min(max_retries, len(strategies)), int(x), int(y),
    )
    return ClickResult(
        success=False,
        strategy="none",
        navigation=False,
        dom_changed=False,
        attempts=min(max_retries, len(strategies)),
        error="All click strategies exhausted — page did not respond",
        pre_url=pre_url,
        post_url=pre_url,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: CDP-Only Click (for overlay bypass integration)
# ═══════════════════════════════════════════════════════════════════════════════

async def cdp_click(
    page: Page,
    x: float,
    y: float,
    button: str = "left",
) -> bool:
    """Simple CDP click without verification. Used by overlay_detector.
    
    Returns True if the CDP event was dispatched (not whether it had effect).
    """
    return await _strategy_cdp_mouse_event(page, x, y, button)
