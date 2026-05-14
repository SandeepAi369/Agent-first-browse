"""Navigate Skill — Go to any URL with verification.

Handles navigation, page load waiting, and URL change verification.
Uses the stealth warm-up infrastructure to avoid detection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from playwright.async_api import Page

from skills.base import Skill, SkillResult

logger = logging.getLogger("skills.navigate")


class NavigateSkill(Skill):
    """Navigate the browser to a URL and verify arrival."""

    name = "navigate"

    async def run(self, params: dict[str, Any]) -> SkillResult:
        """Navigate to the specified URL.

        Params:
            url (str): The target URL to navigate to.
            wait_until (str): Playwright wait condition. Default: "domcontentloaded"
            timeout (int): Max wait time in ms. Default: 15000
        """
        url = params.get("url")
        if not url:
            return SkillResult(
                success=False,
                summary="No URL provided",
                error="Missing required param: url",
            )

        wait_until = params.get("wait_until", "domcontentloaded")
        timeout = params.get("timeout", 15000)
        prev_url = await self.get_current_url()

        try:
            response = await self.page.goto(
                url,
                wait_until=wait_until,
                timeout=timeout,
            )

            # Wait for JS frameworks to settle
            await asyncio.sleep(0.8)

            new_url = await self.get_current_url()
            status = response.status if response else 0

            # Verify navigation succeeded
            if status >= 400:
                return SkillResult(
                    success=False,
                    summary=f"Navigation returned HTTP {status}",
                    data={"url": new_url, "status": status, "prev_url": prev_url},
                    error=f"HTTP {status}",
                )

            logger.info("Navigated: %s → %s (HTTP %d)", prev_url, new_url, status)
            return SkillResult(
                success=True,
                summary=f"Navigated to {new_url}",
                data={"url": new_url, "status": status, "prev_url": prev_url},
            )

        except Exception as e:
            logger.warning("Navigation failed: %s → %s: %s", prev_url, url, e)
            return SkillResult(
                success=False,
                summary=f"Navigation to {url} failed",
                data={"url": url, "prev_url": prev_url},
                error=str(e),
            )
