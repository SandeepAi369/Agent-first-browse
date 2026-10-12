"""Abstract Skill Base Class.

Every skill must implement `run()` which receives parameters
and returns a structured result. Skills have access to the
Playwright page, DOM parser, and ghost input layer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from playwright.async_api import Page

logger = logging.getLogger("skills")


class SkillResult:
    """Standardized result from a skill execution."""

    __slots__ = ("success", "data", "error", "summary")

    def __init__(
        self,
        success: bool,
        summary: str,
        data: Any = None,
        error: str | None = None,
    ):
        self.success = success
        self.summary = summary
        self.data = data
        self.error = error

    def __repr__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"SkillResult({status} {self.summary})"


class Skill(ABC):
    """Base class for all skills.

    Skills are stateless workers. They receive a Playwright Page
    and action parameters, perform their work, and return a SkillResult.
    """

    name: str = "base_skill"

    def __init__(self, page: Page) -> None:
        self.page = page

    @abstractmethod
    async def run(self, params: dict[str, Any]) -> SkillResult:
        """Execute the skill with the given parameters.

        Args:
            params: Action-specific parameters from the TaskNode.

        Returns:
            SkillResult with success status, data, and error info.
        """
        ...

    async def get_current_url(self) -> str:
        """Helper: get the current page URL safely."""
        try:
            return self.page.url
        except Exception:
            return "unknown"
