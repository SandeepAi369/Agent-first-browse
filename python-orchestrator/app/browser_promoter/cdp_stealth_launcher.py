from __future__ import annotations

import inspect
import os
import platform
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page, Playwright

from app.browser_promoter.state import BrowserConfig
from app.logger import get_logger

try:  # Optional 2026 stealth browser wrapper.
    import cloakbrowser  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    cloakbrowser = None

try:
    from playwright_stealth import stealth_async
except Exception:  # pragma: no cover - optional dependency.
    stealth_async = None

logger = get_logger(__name__)

# Platform-specific user agents — matched to actual OS to avoid detection
_WINDOWS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.0.0 Safari/537.36"
)
_LINUX_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.0.0 Safari/537.36"
)


def _is_linux_env() -> bool:
    """Return True if running on native Linux or inside WSL."""
    return platform.system() == "Linux" or "WSL_DISTRO_NAME" in os.environ


def get_stealth_user_agent() -> str:
    """Return a user agent that matches the current OS.

    Websites compare OS headers (Accept-Platform, Sec-CH-UA-Platform)
    against the User-Agent string. A mismatch (e.g., Linux headers +
    Windows UA) is a strong automation signal. This function ensures
    they always match.
    """
    if _is_linux_env():
        return _LINUX_USER_AGENT
    return _WINDOWS_USER_AGENT


# Default export for backward compatibility
STEALTH_USER_AGENT = get_stealth_user_agent()

STEALTH_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-popup-blocking",
    "--disable-infobars",
    "--disable-features=IsolateOrigins,SitePerProcess,OptimizeImageLoading,EnableOverscroll",
    "--password-store=basic",
]

# Includes webdriver masking + canvas/webgl noise patch + closed shadow root capture.
STEALTH_INIT_SCRIPT = """
(() => {
  const defineSafe = (obj, key, value) => {
    try {
      Object.defineProperty(obj, key, { get: () => value, configurable: true });
    } catch (_) {}
  };

  defineSafe(navigator, 'webdriver', undefined);
  defineSafe(navigator, 'languages', ['en-US', 'en']);
  defineSafe(navigator, 'platform', '""" + ("Linux x86_64" if _is_linux_env() else "Win32") + """');
  defineSafe(navigator, 'deviceMemory', 8);
  defineSafe(navigator, 'hardwareConcurrency', 12);
  defineSafe(navigator, 'maxTouchPoints', 0);
  
  delete window.cdc_goog_xxx;
  delete window.cdc_goog_;
  
  window.chrome = window.chrome || { runtime: {} };

  // Advanced Plugin spoofing
  if (!navigator.plugins || navigator.plugins.length === 0) {
    defineSafe(navigator, 'plugins', [1, 2, 3]);
  }

  const originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function(type, attrs) {
    const ctx = originalGetContext.call(this, type, attrs);
    if (!ctx) return ctx;

    if (type === '2d') {
      const originalGetImageData = ctx.getImageData.bind(ctx);
      ctx.getImageData = function(x, y, w, h) {
        const imageData = originalGetImageData(x, y, w, h);
        if (imageData && imageData.data && imageData.data.length > 0) {
          imageData.data[0] = imageData.data[0] ^ 1;
          imageData.data[1] = imageData.data[1] ^ 1;
        }
        return imageData;
      };
    }

    return ctx;
  };

  const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
  WebGLRenderingContext.prototype.getParameter = function(param) {
    if (param === 37445) return 'Intel Inc.';
    if (param === 37446) return 'Intel Iris OpenGL Engine';
    return originalGetParameter.call(this, param);
  };

  const originalAttachShadow = Element.prototype.attachShadow;
  Element.prototype.attachShadow = function(init) {
    const root = originalAttachShadow.call(this, init);
    try {
      if (init && init.mode === 'closed') {
        Object.defineProperty(this, '__closedShadowRoot__', {
          value: root,
          configurable: true,
          enumerable: false,
          writable: false,
        });
      }
    } catch (_) {}
    return root;
  };
})();
"""


async def launch_stealth_context(
    *,
    playwright: Playwright,
    browser_config: BrowserConfig,
    user_data_dir: Path,
) -> tuple[BrowserContext, bool]:
    """Launch a stealth browser context with cloakbrowser fallback strategy.

    Returns:
      (context, used_cloakbrowser)
    """
    context = await _try_launch_with_cloakbrowser(
        playwright=playwright,
        browser_config=browser_config,
        user_data_dir=user_data_dir,
    )
    if context is not None:
        await context.add_init_script(STEALTH_INIT_SCRIPT)
        logger.info("Using cloakbrowser launcher with humanize mode.")
        return context, True

    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=str(user_data_dir),
        headless=browser_config.headless,
        viewport={"width": 1440, "height": 900},
        locale="en-US",
        timezone_id="UTC",
        user_agent=STEALTH_USER_AGENT,
        java_script_enabled=True,
        device_scale_factor=1,
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        args=STEALTH_LAUNCH_ARGS,
        ignore_default_args=["--enable-automation"],
    )
    await context.add_init_script(STEALTH_INIT_SCRIPT)
    return context, False


async def apply_page_stealth(page: Page) -> None:
    """Apply page-level stealth hardening when plugin is available."""
    if stealth_async is not None:
        await stealth_async(page)


async def _try_launch_with_cloakbrowser(
    *,
    playwright: Playwright,
    browser_config: BrowserConfig,
    user_data_dir: Path,
) -> BrowserContext | None:
    if cloakbrowser is None:
        return None

    launchers = [
        getattr(cloakbrowser, "launch_persistent_context", None),
        getattr(cloakbrowser, "launch_context", None),
        getattr(cloakbrowser, "launch", None),
        getattr(cloakbrowser, "start", None),
    ]

    candidate_kwargs = {
        "playwright": playwright,
        "headless": browser_config.headless,
        "humanize": True,
        "user_data_dir": str(user_data_dir),
        "args": list(STEALTH_LAUNCH_ARGS),
        "ignore_default_args": ["--enable-automation"],
        "viewport": {"width": 1440, "height": 900},
        "locale": "en-US",
        "timezone_id": "UTC",
        "user_agent": STEALTH_USER_AGENT,
        "extra_http_headers": {"Accept-Language": "en-US,en;q=0.9"},
    }

    for launcher in launchers:
        if launcher is None:
            continue
        try:
            filtered = _filter_kwargs(launcher, candidate_kwargs)
            result = launcher(**filtered)
            if inspect.isawaitable(result):
                result = await result
            context = _extract_context(result)
            if context is not None:
                return context
        except Exception as exc:
            logger.warning("cloakbrowser launcher attempt failed: %s", exc)
            continue

    return None


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except Exception:
        # If signature cannot be inspected, send everything and let runtime decide.
        return kwargs

    filtered: dict[str, Any] = {}
    parameters = signature.parameters
    for key, value in kwargs.items():
        if key in parameters:
            filtered[key] = value
    return filtered


def _extract_context(result: Any) -> BrowserContext | None:
    if isinstance(result, BrowserContext):
        return result

    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, BrowserContext):
                return item

    if isinstance(result, dict):
        for key in ("context", "browser_context"):
            value = result.get(key)
            if isinstance(value, BrowserContext):
                return value

    context_attr = getattr(result, "context", None)
    if isinstance(context_attr, BrowserContext):
        return context_attr

    return None
