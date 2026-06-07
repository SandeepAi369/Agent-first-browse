"""Overlay Detector — Pre-click spatial verification for ghost click penetration.

Detects invisible DOM elements (tracking overlays, modal backdrops, cookie banners)
that sit on top of intended click targets via z-index or pointer-events tricks.

Used by ghost_input.ghost_click() to verify that a click at (x, y) will actually
reach the intended element, and to temporarily neutralize overlays if needed.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from playwright.async_api import Page

from app.logger import get_logger

logger = get_logger("overlay_detector")


async def check_click_target(
    page: Page,
    x: float,
    y: float,
    timeout: float = 2.0,
) -> dict:
    """Check what element actually sits at coordinates (x, y).

    Returns:
        {
            'clear': bool,          # True if no overlay detected
            'overlay_tag': str,     # Tag name of overlay element (if any)
            'overlay_id': str,      # ID of overlay (if any)
            'overlay_class': str,   # Class of overlay (if any)
            'target_tag': str,      # Tag of the element at coordinates
            'is_transparent': bool, # Whether the overlay is visually transparent
            'z_index': str,         # z-index of the overlay
        }
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        (params) => {
            const { x, y } = params;
            const topEl = document.elementFromPoint(x, y);
            if (!topEl) {
                return { clear: true, overlay_tag: '', overlay_id: '',
                         overlay_class: '', target_tag: '',
                         is_transparent: false, z_index: 'auto' };
            }

            const styles = window.getComputedStyle(topEl);
            const opacity = parseFloat(styles.opacity);
            const bgColor = styles.backgroundColor;
            const pointerEvents = styles.pointerEvents;
            const zIndex = styles.zIndex;
            const position = styles.position;
            const tag = topEl.tagName.toLowerCase();
            const id = topEl.id || '';
            const cls = topEl.className ?
                (typeof topEl.className === 'string' ? topEl.className : '') : '';
            const text = (topEl.textContent || '').trim();

            // Check if this looks like a tracking/interception overlay
            const isOverlay = (
                // Transparent/invisible overlays
                (opacity < 0.1 && position !== 'static') ||
                // Background-only overlays (no visible content)
                (bgColor === 'rgba(0, 0, 0, 0)' && position === 'fixed' &&
                 topEl.children.length === 0) ||
                // High z-index transparent divs
                (tag === 'div' && parseInt(zIndex) > 999 && opacity < 0.5) ||
                // Known overlay patterns
                (cls.includes('overlay') || cls.includes('backdrop') ||
                 cls.includes('modal-bg') || id.includes('overlay')) ||
                // Pointer-events interceptor (empty div sitting on top)
                (tag === 'div' && !text &&
                 position !== 'static' && parseInt(zIndex) > 0)
            );

            // Check if this is a legitimate interactive element
            const isInteractive = (
                ['a', 'button', 'input', 'textarea', 'select', 'label'].includes(tag) ||
                topEl.getAttribute('role') === 'button' ||
                topEl.getAttribute('role') === 'link' ||
                topEl.onclick !== null ||
                topEl.hasAttribute('data-action')
            );

            return {
                clear: !isOverlay || isInteractive,
                overlay_tag: isOverlay ? tag : '',
                overlay_id: isOverlay ? id : '',
                overlay_class: isOverlay ? cls.slice(0, 100) : '',
                target_tag: tag,
                is_transparent: opacity < 0.1,
                z_index: zIndex,
                pointer_events: pointerEvents,
            };
        }
        """, {"x": x, "y": y}), timeout=timeout)

        return result

    except Exception as e:
        logger.warning("Overlay check failed at (%d, %d): %s", x, y, e)
        # Assume clear on error (don't block clicks on detection failure)
        return {
            "clear": True, "overlay_tag": "", "overlay_id": "",
            "overlay_class": "", "target_tag": "",
            "is_transparent": False, "z_index": "auto",
        }


async def bypass_overlay(
    page: Page,
    x: float,
    y: float,
    timeout: float = 2.0,
) -> bool:
    """Temporarily disable pointer-events on any overlay at (x, y).

    This runs in a single synchronous JS block to prevent the overlay's
    telemetry from reacting. Returns True if an overlay was bypassed.
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        (params) => {
            const { x, y } = params;
            const topEl = document.elementFromPoint(x, y);
            if (!topEl) return { bypassed: false, reason: 'no_element' };

            const tag = topEl.tagName.toLowerCase();
            const styles = window.getComputedStyle(topEl);
            const opacity = parseFloat(styles.opacity);
            const zIndex = parseInt(styles.zIndex) || 0;
            const position = styles.position;
            const text = (topEl.textContent || '').trim();

            // Only bypass if it looks like a non-interactive overlay
            const isInteractive = (
                ['a', 'button', 'input', 'textarea', 'select'].includes(tag) ||
                topEl.getAttribute('role') === 'button' ||
                topEl.onclick !== null
            );

            if (isInteractive) {
                return { bypassed: false, reason: 'interactive_element' };
            }

            const isSuspicious = (
                (opacity < 0.1 && position !== 'static') ||
                (tag === 'div' && zIndex > 100 && !text) ||
                (tag === 'div' && position === 'fixed' && !text)
            );

            if (isSuspicious) {
                // Disable pointer events on the overlay
                const original = topEl.style.pointerEvents;
                topEl.style.pointerEvents = 'none';

                // Schedule restoration after 500ms
                setTimeout(() => {
                    topEl.style.pointerEvents = original || '';
                }, 500);

                return {
                    bypassed: true,
                    reason: 'overlay_disabled',
                    overlay_tag: tag,
                    overlay_z: zIndex,
                    overlay_opacity: opacity,
                };
            }

            return { bypassed: false, reason: 'element_looks_normal' };
        }
        """, {"x": x, "y": y}), timeout=timeout)

        if result.get("bypassed"):
            logger.info(
                "🎯 Overlay bypassed at (%d, %d): %s z=%s opacity=%.2f",
                x, y,
                result.get("overlay_tag", "?"),
                result.get("overlay_z", "?"),
                result.get("overlay_opacity", 0),
            )
            return True
        else:
            logger.debug(
                "No overlay to bypass at (%d, %d): %s",
                x, y, result.get("reason", "unknown"),
            )
            return False

    except Exception as e:
        logger.warning("Overlay bypass failed at (%d, %d): %s", x, y, e)
        return False


async def smart_click_with_penetration(
    page: Page,
    x: float,
    y: float,
) -> dict:
    """Pre-flight overlay check + bypass + click verification.

    This is the recommended entry point for all clicks.

    Returns:
        {
            'clicked': bool,
            'overlay_bypassed': bool,
            'method': str,  # 'direct' or 'penetrated'
        }
    """
    # Step 1: Check for overlays
    check = await check_click_target(page, x, y)

    if check.get("clear"):
        # No overlay — direct click
        return {
            "clicked": True,
            "overlay_bypassed": False,
            "method": "direct",
        }

    # Step 2: Overlay detected — attempt bypass
    logger.warning(
        "⚠️ Overlay detected at (%d, %d): %s#%s .%s (z=%s)",
        x, y,
        check.get("overlay_tag", "?"),
        check.get("overlay_id", "?"),
        check.get("overlay_class", "?")[:40],
        check.get("z_index", "?"),
    )

    bypassed = await bypass_overlay(page, x, y)

    return {
        "clicked": True,
        "overlay_bypassed": bypassed,
        "method": "penetrated" if bypassed else "direct_through_overlay",
    }
