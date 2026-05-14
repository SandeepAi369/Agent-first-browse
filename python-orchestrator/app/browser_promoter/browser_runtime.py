"""Native Playwright browser runtime for Agent First IDE.

Architecture (v4.0 — Persistent Context):
    +─────────────────+    +────────────────────────+    +──────────────+
    │ Python Agent     │───>│ Playwright Engine       │───>│ Chromium DOM  │
    │ (LangGraph)      │    │ (persistent context)    │    │ (real events) │
    +─────────────────+    +────────────────────────+    +──────────────+

Key design:
  - The agent OWNS its browser via `launch_persistent_context`.
  - All sessions/cookies survive restarts (saved to `./persistence/browser_sessions/`).
  - Stealth hardening applied automatically (webdriver masking, canvas noise, etc.).
  - No CDP attachment, no external browser process, no WSL bridge.
  - The human can see and interact with the browser window (headless=False default).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from app import config
from app.browser_promoter.cdp_stealth_launcher import (
    STEALTH_INIT_SCRIPT,
    STEALTH_LAUNCH_ARGS,
    STEALTH_USER_AGENT,
    apply_page_stealth,
    get_random_viewport,
    VISUAL_CURSOR_INIT_SCRIPT,
)
from app.browser_promoter.state import BrowserConfig
from app.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Browser Runtime — Singleton Persistent Context
# ═══════════════════════════════════════════════════════════════════════════════

class BrowserRuntime:
    """Persistent Async Playwright runtime shared across all graph nodes.

    Uses `launch_persistent_context` to own a Chromium browser with saved
    sessions. The browser window stays open between graph cycles and even
    survives full agent restarts (cookies/localStorage persisted on disk).
    """

    _lock = asyncio.Lock()
    _playwright: Playwright | None = None
    _context: BrowserContext | None = None
    _page: Page | None = None
    _active_session_key: str | None = None

    @classmethod
    async def ensure_page(
        cls,
        *,
        browser_config: BrowserConfig,
        platform_name: str = "default",
        thread_id: str = "default",
    ) -> Page:
        """Return a live page, creating the persistent context if needed."""
        normalized_platform = _normalize_name(platform_name)
        normalized_thread = _normalize_name(thread_id)
        session_key = f"{normalized_platform}|{normalized_thread}|{browser_config.headless}"

        async with cls._lock:
            # If session key changed, tear down and rebuild
            if cls._active_session_key and cls._active_session_key != session_key:
                await cls._dispose()

            # Return existing page if still alive
            if cls._page is not None and not cls._page.is_closed():
                return cls._page

            # Bootstrap Playwright engine
            if cls._playwright is None:
                cls._playwright = await async_playwright().start()

            # Launch persistent context (browser + session storage)
            if cls._context is None:
                profile_dir = _persistent_profile_dir(
                    platform_name=normalized_platform,
                    thread_id=normalized_thread,
                )
                logger.info(
                    "Launching Playwright Chromium with persistent profile: %s",
                    profile_dir,
                )
                cls._context = await cls._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(profile_dir),
                    headless=browser_config.headless,
                    viewport={
                        "width": browser_config.viewport_width,
                        "height": browser_config.viewport_height,
                    },
                    locale="en-US",
                    timezone_id="UTC",
                    user_agent=STEALTH_USER_AGENT,
                    java_script_enabled=True,
                    device_scale_factor=1,
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                    args=STEALTH_LAUNCH_ARGS,
                    ignore_default_args=["--enable-automation"],
                )
                await cls._context.add_init_script(STEALTH_INIT_SCRIPT)
                await cls._context.add_init_script(VISUAL_CURSOR_INIT_SCRIPT)
                cls._active_session_key = session_key

            # Get or create page
            if cls._context.pages:
                cls._page = cls._context.pages[0]
            else:
                cls._page = await cls._context.new_page()

            # Apply page-level stealth (playwright_stealth plugin if installed)
            await apply_page_stealth(cls._page)

            # Bring to front for human visibility
            try:
                await cls._page.bring_to_front()
            except Exception:
                pass

            return cls._page

    @classmethod
    async def _dispose(cls) -> None:
        """Tear down the entire browser context and Playwright engine."""
        if cls._page is not None and not cls._page.is_closed():
            try:
                await cls._page.close()
            except Exception:
                pass
            cls._page = None

        if cls._context is not None:
            try:
                await cls._context.close()
            except Exception:
                pass
            cls._context = None

        cls._active_session_key = None

    @classmethod
    async def shutdown(cls) -> None:
        """Gracefully close all browser resources when application exits."""
        async with cls._lock:
            await cls._dispose()
            if cls._playwright is not None:
                try:
                    await cls._playwright.stop()
                except Exception:
                    pass
                cls._playwright = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    """Normalize a name for safe filesystem path usage."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip().lower())
    return cleaned or "default"


def _workspace_persistence_dir() -> Path:
    """Return shared persistence root under project/persistence."""
    workspace_root = Path(__file__).resolve().parents[3]
    persistence_dir = workspace_root / "persistence"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    return persistence_dir


def _persistent_profile_dir(*, platform_name: str, thread_id: str) -> Path:
    """Return the persistent browser profile directory for a given session."""
    base = _workspace_persistence_dir() / "browser_sessions" / platform_name / thread_id
    base.mkdir(parents=True, exist_ok=True)
    return base
