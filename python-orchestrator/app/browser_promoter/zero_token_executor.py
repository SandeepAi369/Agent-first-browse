from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from playwright.async_api import Page

from app.browser_promoter.state import BrowserAction
from app.logger import get_logger
from ghost_input import ghost_click, ghost_scroll, ghost_type

logger = get_logger(__name__)


@dataclass(slots=True)
class ActionExecutionResult:
    details: dict[str, Any]
    vision_calls_used: int = 0


class ActionExecutionError(RuntimeError):
    def __init__(self, message: str, *, vision_calls_used: int = 0) -> None:
        super().__init__(message)
        self.vision_calls_used = vision_calls_used


class ZeroTokenActionExecutor:
    """Coordinate-only executor for reasoning actions.

    Uses ghost_click/ghost_type for all targeted actions. No DOM parsing
    or selector-based fallbacks are allowed in this pipeline.
    """

    async def execute_action(self, *, page: Page, action: BrowserAction) -> ActionExecutionResult:
        if action.action == "manual_intervention_required":
            return ActionExecutionResult(details={
                "action": "manual_intervention_required",
                "message": "Manual step flagged by planner.",
            })

        if action.action == "goto":
            response = await page.goto(
                _normalize_url(action.url),
                timeout=action.timeout_ms,
                wait_until="domcontentloaded",
            )
            return ActionExecutionResult(details={
                "action": "goto",
                "status_code": response.status if response is not None else None,
                "final_url": page.url,
            })

        if action.action == "wait":
            await page.wait_for_timeout(action.wait_ms)
            return ActionExecutionResult(details={
                "action": "wait",
                "wait_ms": action.wait_ms,
            })

        if action.action == "screenshot":
            return ActionExecutionResult(details={"action": "screenshot"})

        if action.action == "scroll":
            await ghost_scroll(page, action.delta_y)
            return ActionExecutionResult(details={
                "action": "scroll",
                "delta_y": action.delta_y,
            })

        if action.action in {"click", "type", "type_and_enter"}:
            if action.x is None or action.y is None:
                raise ActionExecutionError(
                    f"Missing coordinates for action '{action.action}'."
                )

            if action.action == "click":
                await ghost_click(page, action.x, action.y)
                return ActionExecutionResult(details={
                    "action": "click",
                    "x": action.x,
                    "y": action.y,
                })

            if action.action in {"type", "type_and_enter"}:
                await ghost_click(page, action.x, action.y)
                if action.clear_before_type:
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Backspace")
                await ghost_type(page, action.text)
                if action.action == "type_and_enter":
                    await page.keyboard.press("Enter")
                return ActionExecutionResult(details={
                    "action": action.action,
                    "x": action.x,
                    "y": action.y,
                    "typed_characters": len(action.text or ""),
                })

        raise ActionExecutionError(f"Unsupported action: {action.action}")


def _normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"https://{value}"
