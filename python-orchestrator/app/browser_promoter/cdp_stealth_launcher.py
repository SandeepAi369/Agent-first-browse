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

# ── Fixed Viewport ──
# V16.1: Consistent viewport that fills ~75% of a 1920×1080 screen,
# top-to-bottom from the left corner. Right 25% remains free for the user.
# Large enough for vision models to work properly and user to see everything.
_FIXED_VIEWPORT: dict[str, int] = {"width": 1440, "height": 1080}


def get_random_viewport() -> dict[str, int]:
    """Return the fixed viewport. Name kept for backward compatibility."""
    return dict(_FIXED_VIEWPORT)


STEALTH_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-popup-blocking",
    "--disable-infobars",
    # Suppress the "Restore pages / Chrome didn't shut down correctly" crash
    # bubble — it renders top-right and overlaps page controls (e.g. GitHub's
    # Star button), blinding the agent. We hard-terminate sessions (pkill/Ctrl-C)
    # so Chrome thinks it crashed. Both flag spellings for cross-version coverage.
    "--hide-crash-restore-bubble",
    "--disable-session-crashed-bubble",
    "--disable-features=IsolateOrigins,SitePerProcess,OptimizeImageLoading,EnableOverscroll",
    "--password-store=basic",
    # Prevent WebRTC from leaking local IP via mDNS — a common headless tell
    "--disable-features=WebRtcHideLocalIpsWithMdns",
    # Disable background network service that headless mode handles differently
    "--disable-background-networking",
    # Suppress "Chrome is being controlled by automated software" infobar
    "--disable-client-side-phishing-detection",
    # V16.1: Pin window to top-left corner, full height, 75% width
    "--window-position=0,0",
    f"--window-size={_FIXED_VIEWPORT['width']},{_FIXED_VIEWPORT['height']}",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  World-Class Stealth Init Script (v4.0)
#
#  Injection layers:
#    0. Utility (defineSafe) + Session seed (canvasSeed + mulberry32 PRNG)
#    1. Proxy-based navigator.webdriver masking (defeats getOwnPropertyDescriptor)
#    2. Navigator property hardening (platform, deviceMemory, hwConcurrency — per-session)
#    3. chrome.csi / chrome.loadTimes stubs (present in real Chrome, missing in headless)
#    4. Plugin array spoofing (headless has empty plugins)
#    5. Permissions.query override (blocks headless notification leak)
#    6. Canvas 2D fingerprint noise (full-buffer, session-stable via seeded PRNG)
#    7. WebGL1+WebGL2 correlated GPU profile spoofing
#    8. AudioContext session-stable fingerprint perturbation
#    9. Closed ShadowRoot capture (for DOM piercing)
#   10. WebRTC ICE candidate sanitization (strips private IPs without disabling)
#   11. Font enumeration hardening (OS-matched allowlist)
# ═══════════════════════════════════════════════════════════════════════════════

_PLATFORM_STRING = "Linux x86_64" if _is_linux_env() else "Win32"

STEALTH_INIT_SCRIPT = r"""
(() => {
  // ────────────────────────────────────────────────────────
  // 0. Utility + Session Seed
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

  // Session-stable seed: unique per browser session, consistent within
  const canvasSeed = Math.floor(Math.random() * 0xFFFF);

  // Deterministic PRNG (Mulberry32) — same seed always produces same sequence
  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

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
  defineSafe(navigator, 'platform', '""" + _PLATFORM_STRING + r"""');
  // V-18 fix: per-session hardware profile via canvasSeed (not baked at Python import)
  defineSafe(navigator, 'deviceMemory', [4, 8, 8, 16][canvasSeed % 4]);
  defineSafe(navigator, 'hardwareConcurrency', [8, 10, 12, 16][canvasSeed % 4]);
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
  // 6. Canvas 2D — Full-buffer session-stable noise (V-10 fix)
  //    Research: detectors render multiple times and compare.
  //    Random noise fails. Session-stable via mulberry32 passes.
  // ────────────────────────────────────────────────────────
  const originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function(type, attrs) {
    const ctx = originalGetContext.call(this, type, attrs);
    if (!ctx || type !== '2d') return ctx;

    const origGetImageData = ctx.getImageData.bind(ctx);
    ctx.getImageData = function(x, y, w, h) {
      const imageData = origGetImageData(x, y, w, h);
      if (imageData && imageData.data && imageData.data.length > 16) {
        // Session-stable noise: same seed → same noise across renders
        const rng = mulberry32(canvasSeed ^ (w * 7 + h * 13));
        const stride = Math.max(1, Math.floor(imageData.data.length / 400));
        for (let i = 0; i < imageData.data.length; i += stride * 4) {
          const noise = (rng() * 4 - 2) | 0;  // ±2 per channel
          imageData.data[i]   = Math.max(0, Math.min(255, imageData.data[i] + noise));
          imageData.data[i+1] = Math.max(0, Math.min(255, imageData.data[i+1] + (noise ^ 1)));
        }
      }
      return imageData;
    };

    // toDataURL: inject session-stable pixel before export
    const origToDataURL = this.toDataURL.bind(this);
    this.toDataURL = function(fmt) {
      try {
        const rng = mulberry32(canvasSeed);
        ctx.fillStyle = `rgba(${(rng()*3)|0},${(rng()*5)|0},${(rng()*7)|0},0.003)`;
        ctx.fillRect(0, 0, 1, 1);
      } catch (_) {}
      return origToDataURL(fmt);
    };

    return ctx;
  };

  // ────────────────────────────────────────────────────────
  // 7. WebGL — Correlated GPU profile spoofing
  //    Research: detectors cross-check vendor/renderer vs
  //    MAX_TEXTURE_SIZE, MAX_VIEWPORT_DIMS, MAX_RENDERBUFFER_SIZE.
  //    Must be a matched set from real hardware.
  // ────────────────────────────────────────────────────────
  // Direct3D11 renderers (Windows) vs Mesa/OpenGL renderers (Linux). The set
  // MUST match the spoofed navigator.platform — a Linux UA reporting a D3D11
  // renderer is an instant inconsistency tell.
  const _WIN_GPUS = [
    { vendor: 'Google Inc. (Intel)',
      renderer: 'ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11)',
      maxTex: 16384, maxVP: [16384,16384], maxRB: 16384 },
    { vendor: 'Google Inc. (NVIDIA)',
      renderer: 'ANGLE (NVIDIA, NVIDIA GeForce GTX 1660 SUPER Direct3D11 vs_5_0 ps_5_0, D3D11)',
      maxTex: 32768, maxVP: [32768,32768], maxRB: 32768 },
    { vendor: 'Google Inc. (AMD)',
      renderer: 'ANGLE (AMD, AMD Radeon RX 580 Direct3D11 vs_5_0 ps_5_0, D3D11)',
      maxTex: 16384, maxVP: [16384,16384], maxRB: 16384 },
  ];
  const _LIN_GPUS = [
    { vendor: 'Google Inc. (Intel)',
      renderer: 'ANGLE (Intel, Mesa Intel(R) UHD Graphics 630 (CFL GT2), OpenGL 4.6 (Core Profile) Mesa 23.2.1-1ubuntu3.1)',
      maxTex: 16384, maxVP: [16384,16384], maxRB: 16384 },
    { vendor: 'Google Inc. (NVIDIA Corporation)',
      renderer: 'ANGLE (NVIDIA Corporation, NVIDIA GeForce GTX 1660 SUPER/PCIe/SSE2, OpenGL 4.6.0)',
      maxTex: 32768, maxVP: [32768,32768], maxRB: 32768 },
    { vendor: 'Google Inc. (AMD)',
      renderer: 'ANGLE (AMD, AMD Radeon RX 580 Series (polaris10, LLVM 15.0.7, DRM 3.49, 6.5.0), OpenGL 4.6 (Core Profile) Mesa 23.2.1)',
      maxTex: 16384, maxVP: [16384,16384], maxRB: 16384 },
  ];
  const GPU_PROFILES = /Linux/.test(navigator.platform || '') ? _LIN_GPUS : _WIN_GPUS;
  const gpu = GPU_PROFILES[canvasSeed % GPU_PROFILES.length];

  const patchWebGL = (proto) => {
    if (!proto) return;
    const origGetParam = proto.getParameter;
    proto.getParameter = function(p) {
      if (p === 37445) return gpu.vendor;             // UNMASKED_VENDOR
      if (p === 37446) return gpu.renderer;           // UNMASKED_RENDERER
      if (p === 3379)  return gpu.maxTex;             // MAX_TEXTURE_SIZE
      if (p === 3386)  return new Int32Array(gpu.maxVP); // MAX_VIEWPORT_DIMS
      if (p === 34024) return gpu.maxRB;              // MAX_RENDERBUFFER_SIZE
      return origGetParam.call(this, p);
    };
  };

  try { patchWebGL(WebGLRenderingContext.prototype); } catch (_) {}
  try { patchWebGL(WebGL2RenderingContext.prototype); } catch (_) {}

  // ────────────────────────────────────────────────────────
  // 8. AudioContext — Session-stable fingerprint perturbation
  //    Research: random detune per-oscillator is detectable.
  //    Fixed per-session offset from canvasSeed for consistency.
  // ────────────────────────────────────────────────────────
  try {
    const audioOffset = ((canvasSeed % 100) - 50) * 0.001;  // ±0.05 fixed per session

    const origCreateOscillator = AudioContext.prototype.createOscillator;
    AudioContext.prototype.createOscillator = function() {
      const osc = origCreateOscillator.call(this);
      const origConnect = osc.connect.bind(osc);
      osc.connect = function(dest) {
        try { osc.detune.value = audioOffset; } catch (_) {}
        return origConnect(dest);
      };
      return osc;
    };

    // Also intercept createDynamicsCompressor (used by audio fingerprinters)
    const origCreateDC = AudioContext.prototype.createDynamicsCompressor;
    if (origCreateDC) {
      AudioContext.prototype.createDynamicsCompressor = function() {
        const dc = origCreateDC.call(this);
        try { dc.knee.value = dc.knee.value + audioOffset * 10; } catch (_) {}
        return dc;
      };
    }
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

  // ────────────────────────────────────────────────────────
  // 10. WebRTC — Prevent local IP leak without disabling
  //     Research: full disable is detectable. Instead,
  //     intercept ICE candidates and strip private IPs.
  // ────────────────────────────────────────────────────────
  try {
    const origCreateOffer = RTCPeerConnection.prototype.createOffer;
    RTCPeerConnection.prototype.createOffer = function(opts) {
      return origCreateOffer.call(this, opts).then(offer => {
        if (offer && offer.sdp) {
          // Strip private IP candidates (192.168.x, 10.x, 172.16-31.x)
          offer.sdp = offer.sdp.replace(
            /a=candidate:.*?(?:192\.168\.|10\.|172\.(?:1[6-9]|2\d|3[01])\.).*?\r\n/g, ''
          );
        }
        return offer;
      });
    };
  } catch (_) {}

  // ────────────────────────────────────────────────────────
  // 11. Font enumeration — Consistent per-platform allowlist
  //     Research: font list must match OS claim in user-agent.
  //     Unknown fonts → report not available.
  // ────────────────────────────────────────────────────────
  try {
    if (document.fonts && document.fonts.check) {
      const origFontCheck = document.fonts.check.bind(document.fonts);
      const KNOWN_FONTS = new Set([
        'Arial', 'Verdana', 'Helvetica', 'Times New Roman', 'Georgia',
        'Courier New', 'Trebuchet MS', 'Impact', 'Comic Sans MS',
        'Segoe UI', 'Tahoma', 'Calibri', 'Cambria', 'Consolas',
        'Lucida Console', 'Microsoft Sans Serif', 'Palatino Linotype',
        'Roboto', 'Open Sans', 'Lato', 'Noto Sans',
      ]);
      document.fonts.check = function(font, text) {
        // Extract family name from font shorthand (e.g. '16px Arial')
        const name = font.replace(/["']/g, '').split(',')[0].trim();
        const familyName = name.replace(/^\d+(?:\.\d+)?(?:px|pt|em|rem|%)\s*/, '').trim();
        if (KNOWN_FONTS.has(familyName)) return origFontCheck(font, text);
        return false;  // Unknown fonts → not available
      };
    }
  } catch (_) {}
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
        viewport=_FIXED_VIEWPORT,
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
        "viewport": dict(_FIXED_VIEWPORT),
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

# ═══════════════════════════════════════════════════════════════════════════════
#  Visual Cursor Script
# ═══════════════════════════════════════════════════════════════════════════════

VISUAL_CURSOR_INIT_SCRIPT = r"""
(() => {
    // Inject only once per document.
    if (window.__visualCursorInitialized) return;
    window.__visualCursorInitialized = true;

    // ── Realistic native arrow cursor (white fill + black outline, like the OS
    //    pointer) — NOT the old red blob. The hotspot is the TOP-LEFT tip, so the
    //    arrow tip sits exactly on the (x,y) we position it at. ──
    const SVG = 'data:image/svg+xml;utf8,' + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="30" viewBox="0 0 24 30">' +
        '<path d="M3 2 L3 23 L9 17.5 L12.7 25.5 L16 24 L12.3 16 L20 16 Z" ' +
        'fill="#ffffff" stroke="#1a1a1a" stroke-width="1.6" stroke-linejoin="round"/></svg>'
    );

    const cur = document.createElement('div');
    cur.id = 'agent-visual-cursor';
    Object.assign(cur.style, {
        position: 'fixed',          // viewport-relative → correct regardless of scroll
        top: '0px', left: '0px',
        width: '24px', height: '30px',
        zIndex: '2147483647',
        pointerEvents: 'none',
        backgroundImage: 'url("' + SVG + '")',
        backgroundRepeat: 'no-repeat',
        transformOrigin: '3px 2px', // the arrow tip
        transform: 'scale(1)',
        transition: 'transform 0.05s ease-out',
        opacity: '0',               // HIDDEN until we know a real position (no (0,0) flash)
        filter: 'drop-shadow(0 1px 1px rgba(0,0,0,0.45))',
    });

    const attach = () => {
        if (document.body) document.body.appendChild(cur);
        else setTimeout(attach, 30);
    };
    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', attach);
    else attach();

    // Single source of truth for where the visual cursor is, on this page.
    window.__cursorPos = window.__cursorPos || { x: -1, y: -1 };
    const place = (x, y) => {
        if (x == null || y == null || x < 0 || y < 0) return;
        window.__cursorPos = { x: x, y: y };
        cur.style.left = x + 'px';
        cur.style.top = y + 'px';
        cur.style.opacity = '1';    // reveal only once positioned
    };

    // Programmatic positioning — used to RE-SYNC the cursor after a navigation
    // (a fresh page would otherwise leave it hidden / at the corner).
    window.__setCursorPos = (x, y) => place(x, y);

    // Track real movement. clientX/clientY (viewport coords) pair with position:
    // fixed so the on-screen arrow always matches the click coordinates we use.
    document.addEventListener('mousemove', e => place(e.clientX, e.clientY), true);
    document.addEventListener('mousedown', () => { cur.style.transform = 'scale(0.82)'; }, true);
    document.addEventListener('mouseup',   () => { cur.style.transform = 'scale(1)'; }, true);
})();
"""

