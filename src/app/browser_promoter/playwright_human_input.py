"""Playwright-native human-like input driver for browser automation.

Replaces the WSL -> PowerShell -> user32.dll pipeline with Playwright's
own high-level API.  All interactions go through the browser engine
directly -- no OS-level focus required, works on Windows / WSL / Linux.

Architecture:
    +----------------+    +-----------------------+    +--------------+
    | Python          |--->| Playwright Engine      |--->| Browser DOM  |
    | (any OS)        |    | (CDP over WebSocket)   |    | (real events)|
    +----------------+    +-----------------------+    +--------------+

Key advantages over PhysicalInputDriver:
  - No OS-level focus required -- works in background
  - No PowerShell subprocess overhead (was ~300-500ms per action)
  - Cross-platform -- identical behavior on Windows, Linux, macOS
  - Playwright's built-in actionability checks (auto-wait, scroll-into-view)
  - Human-like timing preserved via configurable random delays

v2.1 improvements:
  - Realistic scroll physics with ease-in/ease-out curve
  - Auto-scroll-into-view during long typing sessions
  - Smooth deceleration at scroll boundaries
  - Natural reading pauses during scrolling
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import Any

from playwright.async_api import Page

from app.logger import get_logger

logger = get_logger(__name__)


# ===========================================================================
#  Timing Constants — tuned for realistic human behavior
# ===========================================================================

# Typing delay range in milliseconds (per character)
TYPING_DELAY_MIN_MS = 28
TYPING_DELAY_MAX_MS = 115

# Longer pause between words/sentences (triggered on space / newline)
TYPING_WORD_PAUSE_MIN_MS = 60
TYPING_WORD_PAUSE_MAX_MS = 180
TYPING_NEWLINE_PAUSE_MIN_MS = 200
TYPING_NEWLINE_PAUSE_MAX_MS = 500

# Pause before click (to mimic human aiming hesitation)
PRE_CLICK_PAUSE_MIN_MS = 40
PRE_CLICK_PAUSE_MAX_MS = 180

# Pause after click (to mimic human reaction time)
POST_CLICK_PAUSE_MIN_MS = 80
POST_CLICK_PAUSE_MAX_MS = 260

# Typo probability on alphabetic characters (8% matches old driver)
TYPO_PROBABILITY = 0.08

# Scroll physics — tuned for natural feel
SCROLL_NOTCH_PX = 45          # Small notch per wheel tick (was 120 — way too big!)
SCROLL_INTER_NOTCH_MIN_MS = 50     # 50ms min between notches (was 14ms!)
SCROLL_INTER_NOTCH_MAX_MS = 120    # 120ms max between notches (was 38ms!)
SCROLL_READING_PAUSE_PROBABILITY = 0.15   # 15% chance of a reading pause
SCROLL_READING_PAUSE_MIN_MS = 300         # Reading pause duration
SCROLL_READING_PAUSE_MAX_MS = 800

# Auto-scroll during typing: ensure cursor stays visible
TYPING_AUTO_SCROLL_INTERVAL = 40  # Check every N characters


class PlaywrightHumanInput:
    """Human-like input driver using Playwright's native API.

    Delivers click, type, scroll, and clear actions through Playwright's
    built-in methods.  All interactions include randomized timing to
    mimic human behavior patterns.

    This driver works without OS-level window focus because Playwright
    communicates directly with the browser's rendering engine via CDP.

    v2.1 Features:
      - Realistic scroll with ease-in/ease-out acceleration curve
      - Auto-scroll-into-view keeps cursor visible during long typing
      - Natural reading pauses during scrolling
      - Word/sentence boundary awareness for typing rhythm
    """

    def __init__(
        self,
        *,
        typing_delay_min_ms: int = TYPING_DELAY_MIN_MS,
        typing_delay_max_ms: int = TYPING_DELAY_MAX_MS,
        enable_bezier_movement: bool = True,
        enable_typo_simulation: bool = True,
    ) -> None:
        self._typing_delay_min = typing_delay_min_ms
        self._typing_delay_max = typing_delay_max_ms
        self._enable_bezier = enable_bezier_movement
        self._enable_typos = enable_typo_simulation

    # ── Click ─────────────────────────────────────────────────────────────

    async def click(self, page: Page, x: float, y: float) -> dict[str, Any]:
        """Click at viewport coordinates (x, y) with human-like movement.

        Performs optional Bezier mouse movement to the target, then clicks.
        Includes pre-click and post-click pauses for natural timing.

        Args:
            page: Playwright page.
            x: Viewport X coordinate.
            y: Viewport Y coordinate.

        Returns:
            Dict with action details including coordinates and method.
        """
        # Pre-click pause (human aiming hesitation)
        await asyncio.sleep(
            random.randint(PRE_CLICK_PAUSE_MIN_MS, PRE_CLICK_PAUSE_MAX_MS) / 1000.0
        )

        if self._enable_bezier:
            await self._bezier_move(page, x, y)

        await page.mouse.click(x, y)

        # Post-click pause (human reaction time)
        await asyncio.sleep(
            random.randint(POST_CLICK_PAUSE_MIN_MS, POST_CLICK_PAUSE_MAX_MS) / 1000.0
        )

        logger.info("Playwright click at (%.0f, %.0f)", x, y)
        return {"action": "click", "x": x, "y": y, "method": "playwright-mouse"}

    async def click_selector(
        self,
        page: Page,
        selector: str,
        *,
        timeout_ms: int = 15000,
    ) -> dict[str, Any]:
        """Click an element by CSS selector using Playwright's native click.

        Uses Playwright's built-in actionability checks:
        - Waits for element to be attached to DOM
        - Waits for element to be visible
        - Waits for element to be enabled
        - Scrolls element into view
        - Retries if element is moving/animating

        Args:
            page: Playwright page.
            selector: CSS selector string.
            timeout_ms: Maximum time to wait for element.

        Returns:
            Dict with action details including selector used.
        """
        await asyncio.sleep(
            random.randint(PRE_CLICK_PAUSE_MIN_MS, PRE_CLICK_PAUSE_MAX_MS) / 1000.0
        )

        await page.click(selector, timeout=timeout_ms)

        await asyncio.sleep(
            random.randint(POST_CLICK_PAUSE_MIN_MS, POST_CLICK_PAUSE_MAX_MS) / 1000.0
        )

        logger.info("Playwright click on selector: %s", selector[:80])
        return {
            "action": "click",
            "selector": selector,
            "method": "playwright-selector",
        }

    # ── Type ──────────────────────────────────────────────────────────────

    async def type_text(
        self,
        page: Page,
        text: str,
    ) -> dict[str, int]:
        """Type text character-by-character with human-like timing.

        Features:
        - Optional typo simulation (8% on alphabetic characters)
        - Natural pauses at word boundaries (spaces) and line breaks
        - Auto-scroll-into-view every N characters to keep cursor visible
        - Adaptive timing: faster in mid-word, slower at boundaries

        Args:
            page: Playwright page.
            text: Text to type.

        Returns:
            Dict with ``typed`` (character count) and ``corrections`` (typo count).
        """
        if not text:
            return {"typed": 0, "corrections": 0}

        corrections = 0
        chars_since_scroll = 0

        for i, char in enumerate(text):
            # ── Auto-scroll to keep cursor visible during long typing ──
            chars_since_scroll += 1
            if chars_since_scroll >= TYPING_AUTO_SCROLL_INTERVAL:
                await self._ensure_cursor_visible(page)
                chars_since_scroll = 0

            # ── Simulate occasional typos on alphabetic characters ──
            if self._enable_typos and char.isalpha() and random.random() < TYPO_PROBABILITY:
                typo_char = random.choice("abcdefghijklmnopqrstuvwxyz")
                await page.keyboard.type(typo_char, delay=0)
                await asyncio.sleep(
                    random.randint(self._typing_delay_min, self._typing_delay_max) / 1000.0
                )
                await page.keyboard.press("Backspace")
                await asyncio.sleep(
                    random.randint(30, 120) / 1000.0
                )
                corrections += 1

            # ── Type the actual character ──
            await page.keyboard.type(char, delay=0)

            # ── Adaptive delay based on character type ──
            if char == "\n":
                # Newline: longer pause (human thinking about next line)
                await asyncio.sleep(
                    random.randint(TYPING_NEWLINE_PAUSE_MIN_MS, TYPING_NEWLINE_PAUSE_MAX_MS) / 1000.0
                )
                # Also auto-scroll after every newline to keep visible
                await self._ensure_cursor_visible(page)
                chars_since_scroll = 0
            elif char == " ":
                # Space between words: slightly longer pause
                await asyncio.sleep(
                    random.randint(TYPING_WORD_PAUSE_MIN_MS, TYPING_WORD_PAUSE_MAX_MS) / 1000.0
                )
            elif char in ".,;:!?":
                # Punctuation: brief pause (natural sentence rhythm)
                await asyncio.sleep(
                    random.randint(60, 150) / 1000.0
                )
            else:
                # Normal character
                await asyncio.sleep(
                    random.randint(self._typing_delay_min, self._typing_delay_max) / 1000.0
                )

        # Final scroll-into-view after typing completes
        await self._ensure_cursor_visible(page)

        logger.info(
            "Playwright type: %d chars, %d corrections, text=%r",
            len(text), corrections, text[:60],
        )
        return {"typed": len(text), "corrections": corrections}

    async def type_into_selector(
        self,
        page: Page,
        selector: str,
        text: str,
        *,
        clear_first: bool = True,
        timeout_ms: int = 15000,
    ) -> dict[str, Any]:
        """Click a field by selector, optionally clear it, then type text.

        Args:
            page: Playwright page.
            selector: CSS selector of the input field.
            text: Text to type.
            clear_first: Whether to clear existing content first.
            timeout_ms: Maximum time to wait for element.

        Returns:
            Combined dict with click and typing details.
        """
        # Click to focus the field
        await self.click_selector(page, selector, timeout_ms=timeout_ms)

        if clear_first:
            await self.clear_field(page)

        typed = await self.type_text(page, text)

        return {
            "action": "type",
            "selector": selector,
            "method": "playwright-selector",
            **typed,
        }

    # ── Clear Field ───────────────────────────────────────────────────────

    async def clear_field(self, page: Page) -> None:
        """Select all text in the focused field and delete it.

        Uses Ctrl+A (or Meta+A on Mac) -> Backspace, which works
        universally across all platforms and input types.
        """
        await page.keyboard.press("Control+a")
        await asyncio.sleep(random.randint(40, 80) / 1000.0)
        await page.keyboard.press("Backspace")
        await asyncio.sleep(random.randint(30, 60) / 1000.0)

        logger.debug("Field cleared via Ctrl+A -> Backspace")

    # ── Scroll (v2.1 — realistic physics) ─────────────────────────────────

    async def scroll(
        self,
        page: Page,
        delta_y: float,
    ) -> dict[str, float]:
        """Scroll the page with realistic human-like physics.

        Uses an ease-in/ease-out acceleration curve:
        - Starts slow (finger just touched the wheel)
        - Accelerates to cruise speed
        - Decelerates at the end (finger lifting off)
        - Includes random "reading pauses" (15% chance per chunk)
        - Tiny overshoot + correction for natural feel

        Positive ``delta_y`` = scroll DOWN, negative = scroll UP.

        Args:
            page: Playwright page.
            delta_y: Total pixels to scroll (positive=down, negative=up).

        Returns:
            Dict with ``moved`` and ``overshoot`` values.
        """
        if abs(delta_y) < 1.0:
            return {"moved": 0.0, "overshoot": 0.0}

        direction = 1 if delta_y >= 0 else -1
        total_px = abs(delta_y)

        # Calculate number of notches (small increments for smooth feel)
        notches = max(3, int(total_px / SCROLL_NOTCH_PX))
        overshoot_notches = random.randint(1, 2)

        # ── Main scroll with ease-in/ease-out ──
        total_notches = notches + overshoot_notches
        scrolled_px = 0.0

        for i in range(total_notches):
            # Ease-in/ease-out: compute speed factor using sine curve
            # t goes from 0.0 to 1.0 across the scroll
            t = i / max(1, total_notches - 1)
            # Sine easing: slow at start, fast in middle, slow at end
            speed_factor = math.sin(t * math.pi)
            # Clamp minimum speed so scroll never completely stops
            speed_factor = max(0.25, speed_factor)

            # Variable chunk size based on speed
            chunk_px = SCROLL_NOTCH_PX * speed_factor * random.uniform(0.8, 1.2)
            chunk = chunk_px * direction
            scrolled_px += chunk_px

            await page.mouse.wheel(0, chunk)

            # Variable delay: slower at edges, faster in middle
            base_delay = random.randint(SCROLL_INTER_NOTCH_MIN_MS, SCROLL_INTER_NOTCH_MAX_MS)
            # Invert speed_factor for delay (slow movement = long delay, fast = short)
            delay_factor = 1.0 + (1.0 - speed_factor) * 0.6
            delay_ms = base_delay * delay_factor
            await asyncio.sleep(delay_ms / 1000.0)

            # ── Random "reading pause" ──
            # Humans sometimes pause to read what scrolled into view
            if random.random() < SCROLL_READING_PAUSE_PROBABILITY:
                pause_ms = random.randint(SCROLL_READING_PAUSE_MIN_MS, SCROLL_READING_PAUSE_MAX_MS)
                logger.debug("Scroll reading pause: %dms", pause_ms)
                await asyncio.sleep(pause_ms / 1000.0)

        # ── Gentle correction (reverse the overshoot) ──
        overshoot_px = scrolled_px - total_px
        if abs(overshoot_px) > 5.0:
            correction_steps = random.randint(2, 4)
            for step in range(correction_steps):
                t = step / max(1, correction_steps - 1)
                # Decelerate the correction
                factor = 1.0 - (t * 0.6)
                corr_chunk = (overshoot_px / correction_steps) * factor * (-direction)
                await page.mouse.wheel(0, corr_chunk)
                await asyncio.sleep(random.randint(60, 140) / 1000.0)

        overshoot_report = overshoot_notches * SCROLL_NOTCH_PX * direction

        logger.info(
            "Playwright scroll: %.0fpx in %d notches (overshoot=%.0fpx)",
            delta_y, total_notches, overshoot_report,
        )
        return {"moved": float(delta_y), "overshoot": float(overshoot_report)}

    # ── Key Press ─────────────────────────────────────────────────────────

    async def press_key(self, page: Page, key: str) -> None:
        """Press a keyboard key (e.g., 'Enter', 'Tab', 'Escape').

        Args:
            page: Playwright page.
            key: Key name (Playwright key string format).
        """
        await asyncio.sleep(random.randint(30, 80) / 1000.0)
        await page.keyboard.press(key)
        await asyncio.sleep(random.randint(40, 100) / 1000.0)

        logger.debug("Playwright key press: %s", key)

    # ── Auto-Scroll Into View ─────────────────────────────────────────────

    async def _ensure_cursor_visible(self, page: Page) -> None:
        """Scroll the page/container so the active cursor remains visible.

        Uses JavaScript to find the active element and scroll it into
        view if it has gone below the visible viewport.  This prevents
        the "typing below the fold" problem where typed text disappears
        off-screen.
        """
        try:
            await page.evaluate("""
                () => {
                    const el = document.activeElement;
                    if (!el) return;

                    // For textareas and inputs, scroll the element itself
                    if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
                        el.scrollTop = el.scrollHeight;
                    }

                    // Also scroll the element into the viewport
                    el.scrollIntoView({
                        behavior: 'smooth',
                        block: 'nearest',
                        inline: 'nearest'
                    });
                }
            """)
        except Exception:
            # Non-critical — don't let scroll-into-view failure break typing
            pass

    # ── Bezier Mouse Movement ─────────────────────────────────────────────

    async def _bezier_move(
        self,
        page: Page,
        target_x: float,
        target_y: float,
    ) -> None:
        """Move the mouse along a cubic Bezier curve for natural trajectory.

        Uses the same physics model as the original CDPHumanBehavior to
        maintain identical trajectory characteristics, but delivered via
        Playwright's ``page.mouse.move()``.
        """
        # Get current mouse position (default to center-ish if unknown)
        # Playwright doesn't expose current mouse pos, so we track it
        # by just starting from a reasonable position near the target
        start_x = target_x + random.uniform(-120, 120)
        start_y = target_y + random.uniform(-80, 80)

        distance = math.hypot(target_x - start_x, target_y - start_y)
        steps = max(8, min(30, int(distance / 20.0)))

        # Random control points for natural curvature
        ctrl1_x = start_x + (target_x - start_x) * random.uniform(0.2, 0.45) + random.uniform(-30, 30)
        ctrl1_y = start_y + (target_y - start_y) * random.uniform(0.2, 0.45) + random.uniform(-25, 25)
        ctrl2_x = start_x + (target_x - start_x) * random.uniform(0.55, 0.8) + random.uniform(-25, 25)
        ctrl2_y = start_y + (target_y - start_y) * random.uniform(0.55, 0.8) + random.uniform(-20, 20)

        for step in range(1, steps + 1):
            t = step / steps
            px = self._cubic_bezier(t, start_x, ctrl1_x, ctrl2_x, target_x)
            py = self._cubic_bezier(t, start_y, ctrl1_y, ctrl2_y, target_y)

            await page.mouse.move(px, py)
            await asyncio.sleep(random.uniform(0.005, 0.016))

    @staticmethod
    def _cubic_bezier(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
        """Evaluate cubic Bezier at parameter t in [0, 1]."""
        u = 1.0 - t
        return (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3
