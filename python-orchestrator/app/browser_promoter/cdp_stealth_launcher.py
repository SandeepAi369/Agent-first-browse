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

# ── Randomized Viewport Pool ──
# Real-world common resolutions to prevent fingerprint linkage across sessions.
import random as _random

VIEWPORT_POOL: list[dict[str, int]] = [
    {"width": 1920, "height": 1080},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1366, "height": 768},
    {"width": 1600, "height": 900},
    {"width": 1280, "height": 800},
]


def get_random_viewport() -> dict[str, int]:
    """Return a random viewport from the pool for this session."""
    return _random.choice(VIEWPORT_POOL)


STEALTH_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-popup-blocking",
    "--disable-infobars",
    "--disable-features=IsolateOrigins,SitePerProcess,OptimizeImageLoading,EnableOverscroll",
    "--password-store=basic",
    # Prevent WebRTC from leaking local IP via mDNS — a common headless tell
    "--disable-features=WebRtcHideLocalIpsWithMdns",
    # Disable background network service that headless mode handles differently
    "--disable-background-networking",
    # Suppress "Chrome is being controlled by automated software" infobar
    "--disable-client-side-phishing-detection",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  World-Class Stealth Init Script (v3.0)
#
#  Injection layers:
#    1. Proxy-based navigator.webdriver masking (defeats getOwnPropertyDescriptor)
#    2. Canvas 2D fingerprint noise (per-session random seed)
#    3. WebGL1 + WebGL2 vendor/renderer spoofing
#    4. Permissions.query override (blocks headless notification leak)
#    5. chrome.csi / chrome.loadTimes stubs (present in real Chrome, missing in headless)
#    6. Plugin array spoofing (headless has empty plugins)
#    7. AudioContext fingerprint noise
#    8. Closed ShadowRoot capture (for DOM piercing)
#    9. navigator property hardening (platform, deviceMemory, hardwareConcurrency)
# ═══════════════════════════════════════════════════════════════════════════════

_PLATFORM_STRING = "Linux x86_64" if _is_linux_env() else "Win32"

STEALTH_INIT_SCRIPT = """
(() => {
  // ────────────────────────────────────────────────────────
  // 0. Utility: define property that survives introspection
  // ────────────────────────────────────────────────────────
  const defineSafe = (obj, key, value) => {
    try {
      Object.defineProperty(obj, key, {
        get: () => value,
        configurable: true,
        enumerable: true,
      });
    } catch (_) {}
  };

  // ────────────────────────────────────────────────────────
  // 1. navigator.webdriver — Proxy-based deep masking
  //    Defeats: Object.getOwnPropertyDescriptor(navigator, 'webdriver')
  //    Defeats: navigator.webdriver === undefined checks
  //    Defeats: 'webdriver' in navigator checks
  // ────────────────────────────────────────────────────────
  try {
    const originalNavigator = navigator;
    const navigatorProxy = new Proxy(originalNavigator, {
      has: (target, key) => {
        if (key === 'webdriver') return false;
        return key in target;
      },
      get: (target, key) => {
        if (key === 'webdriver') return undefined;
        const value = target[key];
        return typeof value === 'function' ? value.bind(target) : value;
      },
      getOwnPropertyDescriptor: (target, key) => {
        if (key === 'webdriver') return undefined;
        return Object.getOwnPropertyDescriptor(target, key);
      },
    });
    Object.defineProperty(window, 'navigator', {
      get: () => navigatorProxy,
      configurable: true,
    });
  } catch (_) {
    // Fallback to simple override if Proxy fails
    try {
      Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
        configurable: true,
      });
    } catch (_) {}
  }

  // ────────────────────────────────────────────────────────
  // 2. Navigator property hardening
  // ────────────────────────────────────────────────────────
  defineSafe(navigator, 'languages', ['en-US', 'en']);
  defineSafe(navigator, 'platform', '""" + _PLATFORM_STRING + """');
  defineSafe(navigator, 'deviceMemory', 8);
  defineSafe(navigator, 'hardwareConcurrency', """ + str(_random.choice([8, 10, 12, 16])) + """);
  defineSafe(navigator, 'maxTouchPoints', 0);

  // Clean Chromium DevTools protocol traces
  delete window.cdc_goog_xxx;
  delete window.cdc_goog_;
  for (const key of Object.keys(window)) {
    if (key.startsWith('cdc_') || key.startsWith('__webdriver')) {
      try { delete window[key]; } catch (_) {}
    }
  }

  // ────────────────────────────────────────────────────────
  // 3. chrome.* stubs — present in real Chrome, missing in headless
  // ────────────────────────────────────────────────────────
  window.chrome = window.chrome || {};
  window.chrome.runtime = window.chrome.runtime || {};

  if (!window.chrome.csi) {
    window.chrome.csi = function() {
      return {
        onloadT: Date.now(),
        startE: Date.now() - Math.floor(Math.random() * 500 + 200),
        pageT: Math.random() * 2000 + 1000,
        tran: 15,
      };
    };
  }

  if (!window.chrome.loadTimes) {
    window.chrome.loadTimes = function() {
      return {
        commitLoadTime: Date.now() / 1000,
        connectionInfo: 'h2',
        finishDocumentLoadTime: Date.now() / 1000 + Math.random(),
        finishLoadTime: Date.now() / 1000 + Math.random() * 2,
        firstPaintAfterLoadTime: 0,
        firstPaintTime: Date.now() / 1000 + 0.1,
        navigationType: 'Other',
        npnNegotiatedProtocol: 'h2',
        requestTime: Date.now() / 1000 - Math.random(),
        startLoadTime: Date.now() / 1000 - Math.random() * 2,
        wasAlternateProtocolAvailable: false,
        wasFetchedViaSpdy: true,
        wasNpnNegotiated: true,
      };
    };
  }

  // ────────────────────────────────────────────────────────
  // 4. Plugin array spoofing (headless = empty plugins)
  // ────────────────────────────────────────────────────────
  if (!navigator.plugins || navigator.plugins.length === 0) {
    const fakePlugin = (name, desc, filename) => ({
      name, description: desc, filename, length: 1,
      0: { type: 'application/pdf', suffixes: 'pdf', description: desc },
    });
    const fakePlugins = [
      fakePlugin('Chrome PDF Plugin', 'Portable Document Format', 'internal-pdf-viewer'),
      fakePlugin('Chrome PDF Viewer', 'Portable Document Format', 'mhjfbmdgcfjbbpaeojofohoefgiehjai'),
      fakePlugin('Native Client', 'Native Client Executable', 'internal-nacl-plugin'),
    ];
    defineSafe(navigator, 'plugins', {
      length: fakePlugins.length,
      item: (i) => fakePlugins[i] || null,
      namedItem: (n) => fakePlugins.find(p => p.name === n) || null,
      refresh: () => {},
      0: fakePlugins[0], 1: fakePlugins[1], 2: fakePlugins[2],
      [Symbol.iterator]: function*() { yield* fakePlugins; },
    });
    defineSafe(navigator, 'mimeTypes', {
      length: 2,
      item: (i) => [{type:'application/pdf'}, {type:'text/pdf'}][i] || null,
      namedItem: () => null,
      0: {type: 'application/pdf', suffixes: 'pdf', description: '', enabledPlugin: fakePlugins[0]},
      1: {type: 'text/pdf', suffixes: 'pdf', description: '', enabledPlugin: fakePlugins[0]},
    });
  }

  // ────────────────────────────────────────────────────────
  // 5. Permissions.query override — block notification headless leak
  // ────────────────────────────────────────────────────────
  try {
    const originalQuery = window.Permissions.prototype.query;
    window.Permissions.prototype.query = function(params) {
      if (params && params.name === 'notifications') {
        return Promise.resolve({ state: Notification.permission });
      }
      return originalQuery.call(this, params);
    };
  } catch (_) {}

  // ────────────────────────────────────────────────────────
  // 6. Canvas 2D fingerprint noise (per-session random seed)
  // ────────────────────────────────────────────────────────
  const canvasSeed = Math.floor(Math.random() * 255);
  const originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function(type, attrs) {
    const ctx = originalGetContext.call(this, type, attrs);
    if (!ctx) return ctx;

    if (type === '2d') {
      const origGetImageData = ctx.getImageData.bind(ctx);
      ctx.getImageData = function(x, y, w, h) {
        const imageData = origGetImageData(x, y, w, h);
        if (imageData && imageData.data && imageData.data.length > 4) {
          // XOR first 4 channels with session-unique seed
          for (let i = 0; i < Math.min(imageData.data.length, 16); i += 4) {
            imageData.data[i] = imageData.data[i] ^ (canvasSeed & 0x03);
            imageData.data[i+1] = imageData.data[i+1] ^ ((canvasSeed >> 2) & 0x03);
          }
        }
        return imageData;
      };

      // Also intercept toDataURL for canvas fingerprinting
      const origToDataURL = this.toDataURL.bind(this);
      this.toDataURL = function(type) {
        // Inject a single invisible pixel mutation before export
        try {
          ctx.fillStyle = `rgba(${canvasSeed % 3}, ${canvasSeed % 5}, ${canvasSeed % 7}, 0.004)`;
          ctx.fillRect(0, 0, 1, 1);
        } catch (_) {}
        return origToDataURL(type);
      };
    }
    return ctx;
  };

  // ────────────────────────────────────────────────────────
  // 7. WebGL1 + WebGL2 vendor/renderer spoofing
  // ────────────────────────────────────────────────────────
  const WEBGL_VENDOR = 'Intel Inc.';
  const WEBGL_RENDERER = 'Intel Iris OpenGL Engine';
  const UNMASKED_VENDOR_EXT = 37445;
  const UNMASKED_RENDERER_EXT = 37446;

  const patchWebGL = (proto) => {
    if (!proto) return;
    const origGetParam = proto.getParameter;
    proto.getParameter = function(param) {
      if (param === UNMASKED_VENDOR_EXT) return WEBGL_VENDOR;
      if (param === UNMASKED_RENDERER_EXT) return WEBGL_RENDERER;
      return origGetParam.call(this, param);
    };
  };

  try { patchWebGL(WebGLRenderingContext.prototype); } catch (_) {}
  try { patchWebGL(WebGL2RenderingContext.prototype); } catch (_) {}

  // ────────────────────────────────────────────────────────
  // 8. AudioContext fingerprint noise
  // ────────────────────────────────────────────────────────
  try {
    const origCreateOscillator = AudioContext.prototype.createOscillator;
    AudioContext.prototype.createOscillator = function() {
      const osc = origCreateOscillator.call(this);
      const origConnect = osc.connect.bind(osc);
      osc.connect = function(dest) {
        // Inject imperceptible frequency drift to defeat audio fingerprinting
        try { osc.detune.value = (Math.random() - 0.5) * 0.2; } catch (_) {}
        return origConnect(dest);
      };
      return osc;
    };
  } catch (_) {}

  // ────────────────────────────────────────────────────────
  // 9. Closed ShadowRoot capture (for DOM piercing)
  // ────────────────────────────────────────────────────────
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
