from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass

from playwright.async_api import CDPSession, Page


@dataclass(slots=True)
class PointerState:
    x: float = 720.0
    y: float = 460.0


class CDPHumanBehavior:
    """Physics-based human interaction using raw CDP input events."""

    def __init__(self) -> None:
        self._cdp_by_page: dict[int, CDPSession] = {}
        self._pointer_by_page: dict[int, PointerState] = {}

    async def human_scroll(self, page: Page, delta_y: float) -> dict[str, float]:
        """Scroll with accelerate-cruise-decelerate profile using cubic Bezier math.

        Includes slight overshoot and correction to mimic human wheel behavior.
        """
        cdp = await self._cdp(page)
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
            # Cubic Bezier ease for accelerate-cruise-decelerate shape.
            p0 = self._cubic_bezier(t0, 0.0, 0.02, 0.98, 1.0)
            p1 = self._cubic_bezier(t1, 0.0, 0.02, 0.98, 1.0)
            chunk = (p1 - p0) * overshoot_target * direction
            moved += chunk

            await cdp.send(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseWheel",
                    "x": pointer.x,
                    "y": pointer.y,
                    "deltaX": random.uniform(-0.35, 0.35),
                    "deltaY": chunk,
                    "pointerType": "mouse",
                },
            )
            await asyncio.sleep(random.uniform(0.012, 0.026))

        correction_total = (target * direction) - moved
        # Small corrective wheel ticks after overshoot.
        correction_steps = random.randint(3, 6)
        for _ in range(correction_steps):
            chunk = correction_total / correction_steps
            await cdp.send(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseWheel",
                    "x": pointer.x,
                    "y": pointer.y,
                    "deltaX": random.uniform(-0.2, 0.2),
                    "deltaY": chunk,
                    "pointerType": "mouse",
                },
            )
            await asyncio.sleep(random.uniform(0.014, 0.03))

        return {
            "moved": float(delta_y),
            "overshoot": float(overshoot * direction),
        }

    async def human_click(self, page: Page, x: float, y: float) -> None:
        """Move using curved trajectory and click via CDP mouse events."""
        cdp = await self._cdp(page)
        pointer = self._pointer(page)

        await self._move_pointer(page, x=x, y=y)
        pointer.x = x
        pointer.y = y

        await cdp.send(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed",
                "x": x,
                "y": y,
                "button": "left",
                "clickCount": 1,
            },
        )
        await asyncio.sleep(random.uniform(0.045, 0.11))
        await cdp.send(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": x,
                "y": y,
                "button": "left",
                "clickCount": 1,
            },
        )

    async def human_type(self, page: Page, text: str) -> dict[str, int]:
        """Type with natural pauses and occasional correction via backspace."""
        cdp = await self._cdp(page)
        typed = 0
        corrections = 0

        for char in text:
            if char.isalpha() and random.random() < 0.08:
                typo = random.choice("abcdefghijklmnopqrstuvwxyz")
                await self._dispatch_text_char(cdp, typo)
                await asyncio.sleep(random.uniform(0.03, 0.12))
                await self._dispatch_backspace(cdp)
                corrections += 1
                await asyncio.sleep(random.uniform(0.03, 0.12))

            await self._dispatch_text_char(cdp, char)
            typed += 1
            await asyncio.sleep(random.uniform(0.03, 0.12))

        return {"typed": typed, "corrections": corrections}

    async def _move_pointer(self, page: Page, *, x: float, y: float) -> None:
        cdp = await self._cdp(page)
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
            await cdp.send(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseMoved",
                    "x": px,
                    "y": py,
                    "buttons": 1,
                },
            )
            await asyncio.sleep(random.uniform(0.004, 0.018))

    async def _dispatch_text_char(self, cdp: CDPSession, char: str) -> None:
        code = self._code_for_char(char)
        vk = ord(char.upper()) if len(char) == 1 and char.isprintable() else 0

        await cdp.send(
            "Input.dispatchKeyEvent",
            {
                "type": "keyDown",
                "text": char,
                "unmodifiedText": char,
                "key": char,
                "code": code,
                "windowsVirtualKeyCode": vk,
            },
        )
        await cdp.send(
            "Input.dispatchKeyEvent",
            {
                "type": "keyUp",
                "key": char,
                "code": code,
                "windowsVirtualKeyCode": vk,
            },
        )

    async def _dispatch_backspace(self, cdp: CDPSession) -> None:
        payload = {
            "key": "Backspace",
            "code": "Backspace",
            "windowsVirtualKeyCode": 8,
        }
        await cdp.send("Input.dispatchKeyEvent", {"type": "keyDown", **payload})
        await cdp.send("Input.dispatchKeyEvent", {"type": "keyUp", **payload})

    async def _cdp(self, page: Page) -> CDPSession:
        key = id(page)
        session = self._cdp_by_page.get(key)
        if session is None:
            session = await page.context.new_cdp_session(page)
            self._cdp_by_page[key] = session
        return session

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

    @staticmethod
    def _code_for_char(char: str) -> str:
        if char == " ":
            return "Space"
        if char == "\n":
            return "Enter"
        if len(char) == 1 and char.isalpha():
            return f"Key{char.upper()}"
        if len(char) == 1 and char.isdigit():
            return f"Digit{char}"
        return "Unidentified"
