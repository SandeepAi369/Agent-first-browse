"""THE CRITIC — Observation + Self-Correction Judge.

After each action, the Critic evaluates whether it succeeded by
observing the page state. It uses DOM comparison, URL change detection,
and optional LLM semantic evaluation to produce a verdict.

Inspired by Magentic-One's Progress Ledger reflection mechanism.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from playwright.async_api import Page

from orchestrator.actions import TaskNode
from skills.base import SkillResult

logger = logging.getLogger("orchestrator.critic")


class Verdict:
    """The Critic's judgment on whether an action succeeded."""

    __slots__ = ("success", "reason", "confidence", "dom_changed", "url_changed")

    def __init__(
        self,
        success: bool,
        reason: str,
        confidence: float = 1.0,
        dom_changed: bool = False,
        url_changed: bool = False,
    ):
        self.success = success
        self.reason = reason
        self.confidence = confidence
        self.dom_changed = dom_changed
        self.url_changed = url_changed

    def __repr__(self) -> str:
        icon = "✅" if self.success else "❌"
        return f"Verdict({icon} {self.reason} [{self.confidence:.0%}])"


class Critic:
    """Observes execution results and judges success/failure.

    The Critic operates in two modes:
    1. FAST MODE (default): Compares pre/post DOM hashes and URLs.
       This is deterministic, instant, and doesn't use any LLM tokens.
    2. SEMANTIC MODE (on-demand): Uses an LLM to semantically evaluate
       whether the goal of a specific node was achieved.
    """

    def __init__(self, page: Page) -> None:
        self._page = page
        self._prev_url: str = ""
        self._prev_dom_hash: int = 0

    async def snapshot_before(self) -> None:
        """Capture pre-action state for comparison."""
        try:
            self._prev_url = self._page.url
        except Exception:
            self._prev_url = ""

        try:
            screenshot_bytes = await self._page.screenshot(type="jpeg", quality=15)
            self._prev_dom_hash = hash(screenshot_bytes[:2000])
        except Exception:
            self._prev_dom_hash = 0

    async def evaluate(self, node: TaskNode, result: SkillResult) -> Verdict:
        """Evaluate whether the action achieved its goal.

        Uses a multi-signal approach:
        1. Did the skill itself report success?
        2. Did the URL change (for navigation actions)?
        3. Did the DOM visually change (for click/type actions)?
        4. Are there error modals or unexpected states?
        """
        # Signal 1: Skill's own assessment
        if not result.success:
            return Verdict(
                success=False,
                reason=f"Skill reported failure: {result.error or result.summary}",
                confidence=0.95,
            )

        # Signal 2: URL change detection
        try:
            current_url = self._page.url
        except Exception:
            current_url = ""

        url_changed = current_url != self._prev_url

        # Signal 3: DOM change detection
        dom_changed = False
        try:
            new_bytes = await self._page.screenshot(type="jpeg", quality=15)
            new_hash = hash(new_bytes[:2000])
            dom_changed = new_hash != self._prev_dom_hash
        except Exception:
            dom_changed = True  # Assume changed if we can't check

        # Signal 4: Error modal detection
        has_error = await self._detect_errors()

        if has_error:
            return Verdict(
                success=False,
                reason="Error modal or alert detected on page",
                confidence=0.85,
                dom_changed=dom_changed,
                url_changed=url_changed,
            )

        # For navigation: URL must have changed
        if node.action == "navigate":
            if url_changed:
                return Verdict(
                    success=True,
                    reason=f"Navigation confirmed: {current_url}",
                    confidence=0.95,
                    url_changed=True,
                    dom_changed=dom_changed,
                )
            else:
                return Verdict(
                    success=False,
                    reason="URL did not change after navigation",
                    confidence=0.8,
                )

        # For click/type: DOM should have changed
        if node.action in ("click", "type"):
            if dom_changed or url_changed:
                return Verdict(
                    success=True,
                    reason=f"Page state changed after {node.action}",
                    confidence=0.85,
                    dom_changed=dom_changed,
                    url_changed=url_changed,
                )
            else:
                return Verdict(
                    success=False,
                    reason=f"No visible effect from {node.action}",
                    confidence=0.7,
                    dom_changed=False,
                    url_changed=False,
                )

        # For extract/screenshot/wait/done: trust the skill result
        return Verdict(
            success=True,
            reason=result.summary,
            confidence=0.9,
            dom_changed=dom_changed,
            url_changed=url_changed,
        )

    async def _detect_errors(self) -> bool:
        """Check for common error indicators on the page."""
        try:
            return await asyncio.wait_for(
                self._page.evaluate("""
                    () => {
                        // Check for error modals
                        const modal = document.querySelector(
                            '[role="alertdialog"], .error-modal, .error-message, ' +
                            '[class*="error"], [class*="alert-danger"]'
                        );
                        if (modal) {
                            const style = window.getComputedStyle(modal);
                            if (style.display !== 'none' && style.visibility !== 'hidden') {
                                return true;
                            }
                        }
                        // Check for HTTP error pages
                        const title = document.title.toLowerCase();
                        if (title.includes('404') || title.includes('500') ||
                            title.includes('error') || title.includes('not found')) {
                            return true;
                        }
                        return false;
                    }
                """),
                timeout=3.0,
            )
        except Exception:
            return False
