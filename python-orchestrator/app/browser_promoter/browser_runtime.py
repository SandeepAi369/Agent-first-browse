from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

try:
    from playwright_stealth import stealth_async
except Exception:  # pragma: no cover
    stealth_async = None


STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
window.chrome = window.chrome || { runtime: {} };
"""

STEALTH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class BrowserRuntime:
    """Persistent Async Playwright runtime shared across graph node executions."""

    _lock = asyncio.Lock()
    _playwright: Playwright | None = None
    _context: BrowserContext | None = None
    _page: Page | None = None

    @classmethod
    async def ensure_page(cls) -> Page:
        """Return a live page, creating persistent context only once."""
        async with cls._lock:
            if cls._page is not None and not cls._page.is_closed():
                return cls._page

            if cls._playwright is None:
                cls._playwright = await async_playwright().start()

            if cls._context is None:
                cls._context = await cls._create_persistent_context(cls._playwright)

            if cls._context.pages:
                cls._page = cls._context.pages[0]
            else:
                cls._page = await cls._context.new_page()

            await cls._apply_stealth(cls._page)
            return cls._page

    @classmethod
    async def _create_persistent_context(cls, playwright: Playwright) -> BrowserContext:
        root_dir = Path(__file__).resolve().parents[2]
        profile_dir = root_dir / ".browser_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=True,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="UTC",
            user_agent=STEALTH_USER_AGENT,
            java_script_enabled=True,
            device_scale_factor=1,
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )
        await context.add_init_script(STEALTH_INIT_SCRIPT)
        return context

    @classmethod
    async def _apply_stealth(cls, page: Page) -> None:
        if stealth_async is not None:
            await stealth_async(page)

    @classmethod
    async def shutdown(cls) -> None:
        """Gracefully close browser resources when application exits."""
        async with cls._lock:
            if cls._context is not None:
                await cls._context.close()
                cls._context = None
                cls._page = None
            if cls._playwright is not None:
                await cls._playwright.stop()
                cls._playwright = None
