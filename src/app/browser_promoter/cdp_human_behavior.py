"""Human-like browser interaction using Playwright's native API.

Refactored from raw CDP ``Input.dispatchMouseEvent`` / ``Input.dispatchKeyEvent``
to Playwright's high-level ``page.mouse`` and ``page.keyboard`` APIs.

Previous CDP-based approach failed because:
  - ``Input.dispatchMouseEvent`` requires OS-level window focus to be received
  - WSL → Windows CDP connections lack this focus guarantee
  - Raw CDP events bypass Playwright's actionability checks

Current Playwright-based approach:
  - ``page.mouse.move/click()`` — delivered through the rendering engine
  - ``page.keyboard.type/press()`` — works without OS focus
  - All actions include human-like timing (random delays, Bézier curves)

.. note::
    For production use, prefer ``PlaywrightHumanInput`` from
    ``playwright_human_input.py`` which provides a cleaner API.
    This module is retained for backward compatibility with code
    that directly imports ``CDPHumanBehavior``.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass

from playwright.async_api import Page


@dataclass(slots=True)
class PointerState:
    x: float = 720.0
    y: float = 460.0


class CDPHumanBehavior:
    """Human-like interaction using Playwright's native mouse/keyboard API.

    Provides Bézier curve mouse movement, natural typing delays, and
    physics-based scrolling.  All interactions go through Playwright's
    engine — no CDP session management or OS-level focus required.

    .. deprecated::
        Prefer ``PlaywrightHumanInput`` for new code.  This class is
        maintained for backward compatibility.
    """

    def __init__(self) -> None:
        self._pointer_by_page: dict[int, PointerState] = {}

    async def human_scroll(self, page: Page, delta_y: float) -> dict[str, float]:
        """Scroll with accelerate-cruise-decelerate profile using cubic Bezier math.

        Includes slight overshoot and correction to mimic human wheel behavior.
        Uses ``page.mouse.wheel()`` which works without OS focus.
        """
        pointer = self._pointer(page)

        if abs(delta_y) < 1.0:
            return {"moved": 0.0, "overshoot": 0.0}

        direction = 1.0 if delta_y >= 0 else -1.0
        target = abs(delta_y)
        overshoot = random.uniform(12.0, 60.0)
        overshoot_target = target + overshoot

        steps = max(22, min(52, int(target / 28.0)))
        moved = 0.0

        for step in range(1, steps + 1):
            t0 = (step - 1) / steps
            t1 = step / steps
            p0 = self._cubic_bezier(t0, 0.0, 0.02, 0.98, 1.0)
            p1 = self._cubic_bezier(t1, 0.0, 0.02, 0.98, 1.0)
            chunk = (p1 - p0) * overshoot_target * direction
            moved += chunk

            await page.mouse.wheel(0, chunk)
            await asyncio.sleep(random.uniform(0.012, 0.026))

        # Correction scroll (reverse the overshoot)
        correction_total = (target * direction) - moved
        correction_steps = random.randint(3, 6)
        for _ in range(correction_steps):
            chunk = correction_total / correction_steps
            await page.mouse.wheel(0, chunk)
            await asyncio.sleep(random.uniform(0.014, 0.03))

        return {
            "moved": float(delta_y),
            "overshoot": float(overshoot * direction),
        }

    async def human_click(self, page: Page, x: float, y: float) -> None:
        """Move using curved trajectory and click via Playwright mouse API.

        Uses ``page.mouse.move()`` for Bézier movement and
        ``page.mouse.click()`` for the actual click event.
        """
        pointer = self._pointer(page)

        await self._move_pointer(page, x=x, y=y)
        pointer.x = x
        pointer.y = y

        await page.mouse.click(x, y)

    async def human_type(self, page: Page, text: str) -> dict[str, int]:
        """Type with natural pauses and occasional correction via backspace.

        Uses ``page.keyboard.type()`` and ``page.keyboard.press()``
        which work without OS focus.
        """
        typed = 0
        corrections = 0

        for char in text:
            if char.isalpha() and random.random() < 0.08:
                typo = random.choice("abcdefghijklmnopqrstuvwxyz")
                await page.keyboard.type(typo, delay=0)
                await asyncio.sleep(random.uniform(0.03, 0.12))
                await page.keyboard.press("Backspace")
                corrections += 1
                await asyncio.sleep(random.uniform(0.03, 0.12))

            await page.keyboard.type(char, delay=0)
            typed += 1
            await asyncio.sleep(random.uniform(0.03, 0.12))

        return {"typed": typed, "corrections": corrections}

    async def _move_pointer(self, page: Page, *, x: float, y: float) -> None:
        """Move the mouse along a cubic Bézier curve via page.mouse.move()."""
        pointer = self._pointer(page)

        start_x = pointer.x
        start_y = pointer.y
        distance = math.hypot(x - start_x, y - start_y)
        steps = max(10, min(42, int(distance / 18.0)))

        ctrl1_x = start_x + (x - start_x) * random.uniform(0.2, 0.45) + random.uniform(-40, 40)
        ctrl1_y = start_y + (y - start_y) * random.uniform(0.2, 0.45) + random.uniform(-32, 32)
        ctrl2_x = start_x + (x - start_x) * random.uniform(0.55, 0.8) + random.uniform(-35, 35)
        ctrl2_y = start_y + (y - start_y) * random.uniform(0.55, 0.8) + random.uniform(-28, 28)

        for step in range(1, steps + 1):
            t = step / steps
            px = self._cubic_bezier(t, start_x, ctrl1_x, ctrl2_x, x)
            py = self._cubic_bezier(t, start_y, ctrl1_y, ctrl2_y, y)

            await page.mouse.move(px, py)
            await asyncio.sleep(random.uniform(0.004, 0.018))

    def _pointer(self, page: Page) -> PointerState:
        key = id(page)
        state = self._pointer_by_page.get(key)
        if state is None:
            state = PointerState()
            self._pointer_by_page[key] = state
        return state

    @staticmethod
    def _cubic_bezier(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
        u = 1.0 - t
        return (
            (u ** 3) * p0
            + 3 * (u ** 2) * t * p1
            + 3 * u * (t ** 2) * p2
            + (t ** 3) * p3
        )
