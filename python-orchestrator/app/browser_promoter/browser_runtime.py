from __future__ import annotations

import asyncio
from pathlib import Path
import re

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from app.browser_promoter.state import BrowserConfig, BrowserMode

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
    _browser: Browser | None = None
    _context: BrowserContext | None = None
    _page: Page | None = None
    _active_session_key: str | None = None
    _always_create_agent_page: bool = False

    @classmethod
    async def ensure_page(
        cls,
        *,
        browser_config: BrowserConfig,
        platform_name: str = "default",
        thread_id: str = "default",
    ) -> Page:
        """Return a live page for the requested browser mode and session."""
        normalized_platform = _normalize_platform_name(platform_name)
        normalized_thread = _normalize_thread_id(thread_id)
        requested_session_key = _build_session_key(
            browser_config=browser_config,
            platform_name=normalized_platform,
            thread_id=normalized_thread,
        )

        async with cls._lock:
            if cls._active_session_key and cls._active_session_key != requested_session_key:
                await cls._dispose_current_context()

            if cls._page is not None and not cls._page.is_closed():
                return cls._page

            if cls._playwright is None:
                cls._playwright = await async_playwright().start()

            if cls._context is None:
                context, browser, always_create_agent_page = await cls._create_context_for_mode(
                    playwright=cls._playwright,
                    browser_config=browser_config,
                    platform_name=normalized_platform,
                    thread_id=normalized_thread,
                )
                cls._context = context
                cls._browser = browser
                cls._active_session_key = requested_session_key
                cls._always_create_agent_page = always_create_agent_page

            if cls._context.pages and not cls._always_create_agent_page:
                cls._page = cls._context.pages[0]
            else:
                cls._page = await cls._context.new_page()

            await cls._apply_stealth(cls._page)
            return cls._page

    @classmethod
    async def _create_context_for_mode(
        cls,
        playwright: Playwright,
        browser_config: BrowserConfig,
        platform_name: str,
        thread_id: str,
    ) -> tuple[BrowserContext, Browser | None, bool]:
        if browser_config.mode == BrowserMode.ISOLATED_SERVER:
            profile_dir = _persistent_profile_dir(platform_name=platform_name, thread_id=thread_id)
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=browser_config.headless,
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
            return context, None, False

        endpoint = _resolve_cdp_endpoint(browser_config)
        browser = await playwright.chromium.connect_over_cdp(
            endpoint_url=endpoint,
            timeout=browser_config.cdp_timeout_ms,
        )

        # Safety guard: never attach to user-visible existing tabs.
        # Prefer isolated context; if unavailable, create a dedicated new page.
        try:
            context = await browser.new_context(
                viewport={"width": 1440, "height": 900},
                locale="en-US",
                timezone_id="UTC",
                user_agent=STEALTH_USER_AGENT,
                java_script_enabled=True,
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
        except Exception:
            if browser.contexts:
                context = browser.contexts[0]
            else:
                raise

        await context.add_init_script(STEALTH_INIT_SCRIPT)
        return context, browser, True

    @classmethod
    async def _apply_stealth(cls, page: Page) -> None:
        if stealth_async is not None:
            await stealth_async(page)

    @classmethod
    async def _dispose_current_context(cls) -> None:
        """Close active context/browser while preserving playwright service."""
        if cls._context is not None:
            await cls._context.close()
            cls._context = None
            cls._page = None

        if cls._browser is not None:
            await cls._browser.close()
            cls._browser = None

        cls._active_session_key = None
        cls._always_create_agent_page = False

    @classmethod
    async def shutdown(cls) -> None:
        """Gracefully close browser resources when application exits."""
        async with cls._lock:
            await cls._dispose_current_context()
            if cls._playwright is not None:
                await cls._playwright.stop()
                cls._playwright = None


def _normalize_platform_name(platform_name: str) -> str:
    """Normalize platform segment for safe filesystem path usage."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", platform_name.strip().lower())
    return cleaned or "default"


def _normalize_thread_id(thread_id: str) -> str:
    """Normalize thread identifier for filesystem-safe persistence paths."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", thread_id.strip().lower())
    return cleaned or "default"


def _workspace_persistence_dir() -> Path:
    """Return shared persistence root under ~/agent first IDE/persistence."""
    workspace_root = Path(__file__).resolve().parents[3]
    persistence_dir = workspace_root / "persistence"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    return persistence_dir


def _persistent_profile_dir(*, platform_name: str, thread_id: str) -> Path:
    base = _workspace_persistence_dir() / "browser_sessions" / platform_name / thread_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_cdp_endpoint(browser_config: BrowserConfig) -> str:
    """Resolve CDP endpoint according to selected BrowserMode."""
    if browser_config.mode == BrowserMode.LOCAL_NATIVE_CDP:
        endpoint = browser_config.local_cdp_endpoint.strip() or "http://localhost:9222"
    elif browser_config.mode == BrowserMode.REMOTE_WEBSOCKET:
        endpoint = browser_config.remote_websocket_url.strip()
    else:
        raise ValueError("CDP endpoint resolution requested for non-CDP mode")

    if not endpoint:
        raise ValueError("BrowserConfig remote_websocket_url is required for REMOTE_WEBSOCKET mode")

    return endpoint


def _build_session_key(
    *,
    browser_config: BrowserConfig,
    platform_name: str,
    thread_id: str,
) -> str:
    """Build deterministic key used to decide runtime context reuse boundaries."""
    endpoint = ""
    if browser_config.mode == BrowserMode.LOCAL_NATIVE_CDP:
        endpoint = browser_config.local_cdp_endpoint.strip() or "http://localhost:9222"
    elif browser_config.mode == BrowserMode.REMOTE_WEBSOCKET:
        endpoint = browser_config.remote_websocket_url.strip()

    return "|".join(
        [
            browser_config.mode.value,
            platform_name,
            thread_id,
            endpoint,
            str(browser_config.headless),
        ]
    )
