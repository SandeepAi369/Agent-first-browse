"""Interact Skill — Click, Type, Scroll using ghost_input.

Wraps the humanized Bézier-curve mouse movement, variable-speed
typing, and entropy-injected scrolling from ghost_input.py into
a standardized Skill interface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from playwright.async_api import Page

from skills.base import Skill, SkillResult

logger = logging.getLogger("skills.interact")


class InteractSkill(Skill):
    """Humanized browser interaction: click, type, scroll, press keys."""

    name = "interact"

    def __init__(self, page: Page) -> None:
        super().__init__(page)
        # Lazy import to avoid circular deps at module level
        from ghost_input import ghost_click, ghost_type, ghost_scroll
        self._ghost_click = ghost_click
        self._ghost_type = ghost_type
        self._ghost_scroll = ghost_scroll

    async def run(self, params: dict[str, Any]) -> SkillResult:
        """Execute a browser interaction.

        Params:
            interaction (str): "click", "type", "scroll", "press_key", "clear_and_type"
            x (int): X coordinate for click/type target
            y (int): Y coordinate for click/type target
            text (str): Text to type (for "type" and "clear_and_type")
            key (str): Key to press (for "press_key", e.g., "Enter", "Tab")
            delta (int): Scroll amount in pixels (for "scroll")
            selector (str): Optional CSS selector as alternative to x/y
        """
        interaction = params.get("interaction", "click")

        if interaction == "click":
            return await self._do_click(params)
        elif interaction == "type":
            return await self._do_type(params)
        elif interaction == "clear_and_type":
            return await self._do_clear_and_type(params)
        elif interaction == "scroll":
            return await self._do_scroll(params)
        elif interaction == "press_key":
            return await self._do_press_key(params)
        else:
            return SkillResult(
                success=False,
                summary=f"Unknown interaction type: {interaction}",
                error=f"Unsupported: {interaction}",
            )

    # ── Click ─────────────────────────────────────────────────────────────

    async def _do_click(self, params: dict) -> SkillResult:
        x, y = params.get("x"), params.get("y")
        selector = params.get("selector")

        try:
            if selector:
                # Selector-based click (fallback for complex elements)
                el = self.page.locator(selector).first
                await el.click(timeout=5000)
                logger.info("Clicked selector: %s", selector)
                return SkillResult(
                    success=True,
                    summary=f"Clicked element: {selector}",
                    data={"selector": selector},
                )
            elif x is not None and y is not None:
                await self._ghost_click(self.page, int(x), int(y))
                logger.info("Clicked at (%d, %d)", x, y)
                return SkillResult(
                    success=True,
                    summary=f"Clicked at ({x}, {y})",
                    data={"x": x, "y": y},
                )
            else:
                return SkillResult(
                    success=False,
                    summary="Click requires x/y coordinates or selector",
                    error="Missing coordinates",
                )
        except Exception as e:
            logger.warning("Click failed: %s", e)
            return SkillResult(
                success=False,
                summary="Click failed",
                error=str(e),
            )

    # ── Type ──────────────────────────────────────────────────────────────

    async def _do_type(self, params: dict) -> SkillResult:
        text = params.get("text", "")
        x, y = params.get("x"), params.get("y")

        if not text:
            return SkillResult(
                success=False,
                summary="No text provided for typing",
                error="Missing param: text",
            )

        try:
            # Click target first if coordinates provided
            if x is not None and y is not None:
                await self._ghost_click(self.page, int(x), int(y))
                await asyncio.sleep(0.3)

            await self._ghost_type(self.page, text)
            logger.info("Typed %d chars", len(text))
            return SkillResult(
                success=True,
                summary=f"Typed {len(text)} chars",
                data={"text_length": len(text), "preview": text[:80]},
            )
        except Exception as e:
            logger.warning("Type failed: %s", e)
            return SkillResult(
                success=False,
                summary="Typing failed",
                error=str(e),
            )

    # ── Clear + Type (for form fields) ────────────────────────────────────

    async def _do_clear_and_type(self, params: dict) -> SkillResult:
        text = params.get("text", "")
        x, y = params.get("x"), params.get("y")

        if not text:
            return SkillResult(
                success=False,
                summary="No text for clear_and_type",
                error="Missing param: text",
            )

        try:
            if x is not None and y is not None:
                await self._ghost_click(self.page, int(x), int(y))
                await asyncio.sleep(0.3)

            # Select all + delete to clear
            await self.page.keyboard.press("Control+A")
            await asyncio.sleep(0.1)
            await self.page.keyboard.press("Delete")
            await asyncio.sleep(0.2)

            # Type the new text
            await self._ghost_type(self.page, text)

            logger.info("Cleared and typed %d chars", len(text))
            return SkillResult(
                success=True,
                summary=f"Cleared field and typed {len(text)} chars",
                data={"text_length": len(text), "preview": text[:80]},
            )
        except Exception as e:
            logger.warning("Clear+Type failed: %s", e)
            return SkillResult(
                success=False,
                summary="Clear and type failed",
                error=str(e),
            )

    # ── Scroll ────────────────────────────────────────────────────────────

    async def _do_scroll(self, params: dict) -> SkillResult:
        delta = params.get("delta", 500)
        direction = params.get("direction", "down")

        try:
            pixels = int(delta) if direction == "down" else -int(delta)
            await self._ghost_scroll(self.page, pixels)
            logger.info("Scrolled %s by %d px", direction, abs(pixels))
            return SkillResult(
                success=True,
                summary=f"Scrolled {direction} by {abs(pixels)}px",
                data={"direction": direction, "delta": abs(pixels)},
            )
        except Exception as e:
            logger.warning("Scroll failed: %s", e)
            return SkillResult(
                success=False,
                summary="Scroll failed",
                error=str(e),
            )

    # ── Press Key ─────────────────────────────────────────────────────────

    async def _do_press_key(self, params: dict) -> SkillResult:
        key = params.get("key", "Enter")

        try:
            await self.page.keyboard.press(key)
            logger.info("Pressed key: %s", key)
            return SkillResult(
                success=True,
                summary=f"Pressed key: {key}",
                data={"key": key},
            )
        except Exception as e:
            logger.warning("Key press failed: %s", e)
            return SkillResult(
                success=False,
                summary=f"Key press '{key}' failed",
                error=str(e),
            )
