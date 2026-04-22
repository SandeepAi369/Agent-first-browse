"""Persistent Async Playwright browser runtime for Agent First IDE.

Manages browser lifecycle, CDP connections, and session reuse.
Supports automatic environment detection:

  - Windows Native  → LOCAL_NATIVE_CDP (user's Edge/Chrome via CDP)
  - WSL             → Try Windows CDP first, fall back to Playwright Chromium
  - Linux Native    → ISOLATED_SERVER (Playwright's own Chromium)

All modes use persistent profile directories so logins survive restarts.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from app import config
from app.browser_promoter.cdp_stealth_launcher import (
    STEALTH_INIT_SCRIPT,
    STEALTH_USER_AGENT,
    apply_page_stealth,
    launch_stealth_context,
)
from app.logger import get_logger
from app.browser_promoter.state import BrowserConfig, BrowserMode

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment Detection
# ═══════════════════════════════════════════════════════════════════════════════

def _is_wsl() -> bool:
    """Detect if running inside Windows Subsystem for Linux."""
    return "WSL_DISTRO_NAME" in os.environ


def _is_windows_native() -> bool:
    """Detect if running natively on Windows (not WSL)."""
    return platform.system() == "Windows"


def _is_linux_native() -> bool:
    """Detect if running on native Linux (not WSL)."""
    return platform.system() == "Linux" and not _is_wsl()


def _detect_environment() -> str:
    """Return a human-readable environment label."""
    if _is_windows_native():
        return "Windows Native"
    if _is_wsl():
        return f"WSL ({os.environ.get('WSL_DISTRO_NAME', 'unknown')})"
    if _is_linux_native():
        return f"Linux Native ({platform.release()})"
    return platform.system()


# ═══════════════════════════════════════════════════════════════════════════════
#  Smart Browser Mode Resolution
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_resolve_browser_mode(requested_mode: BrowserMode) -> BrowserMode:
    """Resolve the best browser mode for the current environment.

    Strategy precedence (when BROWSER_STRATEGY=auto):
      1. Windows Native  → LOCAL_NATIVE_CDP (uses Edge/Chrome via CDP)
      2. WSL              → Try CDP to Windows host; fall back to ISOLATED_SERVER
      3. Linux Native     → ISOLATED_SERVER (Playwright's own Chromium)

    When BROWSER_STRATEGY is 'cdp' or 'playwright', that choice is forced.
    """
    strategy = config.BROWSER_STRATEGY

    if strategy == "cdp":
        logger.info("BROWSER_STRATEGY=cdp → forcing LOCAL_NATIVE_CDP")
        return BrowserMode.LOCAL_NATIVE_CDP

    if strategy == "playwright":
        logger.info("BROWSER_STRATEGY=playwright → forcing ISOLATED_SERVER (Playwright Chromium)")
        return BrowserMode.ISOLATED_SERVER

    # strategy == "auto" (default)
    if _is_windows_native():
        logger.info("Windows detected → LOCAL_NATIVE_CDP (Edge/Chrome via CDP)")
        return BrowserMode.LOCAL_NATIVE_CDP

    if _is_wsl():
        # Try to reach Windows host browser via CDP
        wsl_host_ip = _get_wsl_host_ip()
        cdp_port = 9222
        cdp_reachable = False

        if wsl_host_ip:
            cdp_reachable = _is_cdp_reachable(f"http://{wsl_host_ip}:{cdp_port}")

        if not cdp_reachable:
            cdp_reachable = _is_cdp_reachable(f"http://localhost:{cdp_port}")

        if cdp_reachable:
            logger.info("WSL detected + Windows CDP reachable → LOCAL_NATIVE_CDP")
            return BrowserMode.LOCAL_NATIVE_CDP
        else:
            logger.info("WSL detected + CDP NOT reachable → ISOLATED_SERVER (Playwright Chromium)")
            return BrowserMode.ISOLATED_SERVER

    if _is_linux_native():
        logger.info("Linux detected → ISOLATED_SERVER (Playwright Chromium with persistent profile)")
        return BrowserMode.ISOLATED_SERVER

    # Unknown platform — use whatever was requested
    logger.warning("Unknown platform '%s' → using requested mode: %s", platform.system(), requested_mode.value)
    return requested_mode


# ═══════════════════════════════════════════════════════════════════════════════
#  Browser Runtime
# ═══════════════════════════════════════════════════════════════════════════════

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

        # Smart auto-detection: resolve the best mode for this environment
        resolved_mode = _auto_resolve_browser_mode(browser_config.mode)
        if resolved_mode != browser_config.mode:
            browser_config = browser_config.model_copy(update={"mode": resolved_mode})

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

            # Bring page to front (best-effort)
            try:
                await cls._page.bring_to_front()
            except Exception:
                pass

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
            logger.info(
                "Launching Playwright Chromium with persistent profile: %s",
                profile_dir,
            )
            context, used_cloak = await launch_stealth_context(
                playwright=playwright,
                browser_config=browser_config,
                user_data_dir=profile_dir,
            )
            if used_cloak:
                logger.info("Isolated session launched with cloakbrowser strategy.")
            return context, None, False

        # CDP modes: LOCAL_NATIVE_CDP or REMOTE_WEBSOCKET
        endpoint = _resolve_cdp_endpoint(browser_config)
        if browser_config.mode == BrowserMode.LOCAL_NATIVE_CDP:
            endpoint = await _ensure_local_cdp_available(endpoint)

        browser = await playwright.chromium.connect_over_cdp(
            endpoint_url=endpoint,
            timeout=browser_config.cdp_timeout_ms,
        )

        # Reuse the user's EXISTING browser context so all logged-in
        # sessions (Reddit, Gmail, etc.) are available to the agent.
        if browser.contexts:
            context = browser.contexts[0]
        else:
            context = await browser.new_context(
                viewport={"width": 1440, "height": 900},
                locale="en-US",
                timezone_id="UTC",
                user_agent=STEALTH_USER_AGENT,
                java_script_enabled=True,
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )

        await context.add_init_script(STEALTH_INIT_SCRIPT)
        # Always create a NEW tab so we don't hijack whatever the user is looking at
        return context, browser, True

    @classmethod
    async def _apply_stealth(cls, page: Page) -> None:
        await apply_page_stealth(page)

    @classmethod
    async def _dispose_current_context(cls) -> None:
        """Close agent resources while preserving the user's real browser.

        When connected via CDP to the user's real browser, we must NOT call
        context.close() (that would kill their tabs) or browser.close() (that
        would shut down Chrome/Edge entirely).  We only close the agent's own
        page and disconnect the CDP transport.
        """
        if cls._page is not None and not cls._page.is_closed():
            try:
                await cls._page.close()
            except Exception:
                pass
            cls._page = None

        if cls._browser is not None:
            try:
                await cls._browser.close()
            except Exception:
                pass
            cls._browser = None
        elif cls._context is not None:
            # ISOLATED_SERVER mode only: safe to close the whole context
            try:
                await cls._context.close()
            except Exception:
                pass

        cls._context = None
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_platform_name(platform_name: str) -> str:
    """Normalize platform segment for safe filesystem path usage."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", platform_name.strip().lower())
    return cleaned or "default"


def _normalize_thread_id(thread_id: str) -> str:
    """Normalize thread identifier for filesystem-safe persistence paths."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", thread_id.strip().lower())
    return cleaned or "default"


def _workspace_persistence_dir() -> Path:
    """Return shared persistence root under project/persistence."""
    workspace_root = Path(__file__).resolve().parents[3]
    persistence_dir = workspace_root / "persistence"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    return persistence_dir


def _persistent_profile_dir(*, platform_name: str, thread_id: str) -> Path:
    base = _workspace_persistence_dir() / "browser_sessions" / platform_name / thread_id
    base.mkdir(parents=True, exist_ok=True)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
#  CDP Endpoint Resolution
# ═══════════════════════════════════════════════════════════════════════════════

def _get_wsl_host_ip() -> str | None:
    """Read the Windows Host IP automatically from WSL resolv.conf."""
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.startswith("nameserver"):
                    return line.split()[1].strip()
    except Exception:
        pass
    return None


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


def _is_cdp_reachable(endpoint: str) -> bool:
    """Quick TCP reachability check for CDP host/port."""
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or 9222
    try:
        with socket.create_connection((host, port), timeout=0.8):
            return True
    except OSError:
        return False


async def _ensure_local_cdp_available(endpoint: str) -> str:
    """Ensure LOCAL_NATIVE_CDP target is reachable and return a working endpoint."""
    resolved = await _pick_reachable_endpoint(endpoint)
    if resolved is not None:
        return resolved

    if not config.LOCAL_BROWSER_AUTOSTART:
        # On Linux/WSL, if CDP not available and autostart disabled, suggest fallback
        if _is_linux_native() or _is_wsl():
            raise RuntimeError(
                "LOCAL_NATIVE_CDP endpoint is not reachable and no Windows browser is available. "
                "Set BROWSER_STRATEGY=playwright to use Playwright's own Chromium, or "
                "start a browser with --remote-debugging-port=9222."
            )
        raise RuntimeError(
            "LOCAL_NATIVE_CDP endpoint is not reachable. "
            "Start Edge/Chrome with --remote-debugging-port=9222 and retry."
        )

    launched = await asyncio.to_thread(
        _launch_browser_for_cdp,
        endpoint,
        config.LOCAL_BROWSER_PREFERENCE,
    )
    if launched:
        logger.info("Launched local browser for CDP attach: %s", launched)

    deadline = time.monotonic() + max(2.0, config.LOCAL_BROWSER_STARTUP_TIMEOUT_SECONDS)
    while time.monotonic() < deadline:
        resolved = await _pick_reachable_endpoint(endpoint)
        if resolved is not None:
            return resolved
        await asyncio.sleep(0.4)

    # If auto-start failed on WSL/Linux, give a helpful fallback message
    if _is_linux_native() or _is_wsl():
        raise RuntimeError(
            "Could not start or connect to a browser for CDP. "
            "Set BROWSER_STRATEGY=playwright to use Playwright's built-in Chromium instead."
        )

    raise RuntimeError(
        "LOCAL_NATIVE_CDP endpoint is still unreachable after auto-start. "
        "Open Edge/Chrome manually with --remote-debugging-port=9222."
    )


async def _pick_reachable_endpoint(endpoint: str) -> str | None:
    """Return first reachable CDP endpoint candidate."""
    for candidate in _candidate_cdp_endpoints(endpoint):
        if await asyncio.to_thread(_is_cdp_reachable, candidate):
            return candidate
    return None


def _candidate_cdp_endpoints(endpoint: str) -> list[str]:
    """Generate CDP endpoint variants to handle different networking setups."""
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    scheme = parsed.scheme or "http"
    port = parsed.port or 9222

    candidates = [endpoint]
    if host in {"localhost", "127.0.0.1"}:
        candidates.append(f"{scheme}://127.0.0.1:{port}")
        candidates.append(f"{scheme}://localhost:{port}")
        if _is_wsl():
            wsl_ip = _get_wsl_host_ip()
            if wsl_ip:
                candidates.append(f"{scheme}://{wsl_ip}:{port}")
    elif _is_wsl():
        candidates.append(f"{scheme}://localhost:{port}")
        candidates.append(f"{scheme}://127.0.0.1:{port}")

    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


# ═══════════════════════════════════════════════════════════════════════════════
#  Browser Auto-Launch (per platform)
# ═══════════════════════════════════════════════════════════════════════════════

def _launch_browser_for_cdp(endpoint: str, preference: str) -> str | None:
    """Best-effort launch of a browser for CDP control.

    Detects the platform and uses the appropriate launch strategy:
    - Windows Native: Direct subprocess with Edge/Chrome paths
    - WSL: PowerShell bridge to launch Windows Edge/Chrome
    - Linux Native: Direct launch of google-chrome or chromium
    """
    if _is_windows_native():
        return _launch_browser_windows_native(endpoint, preference)
    if _is_wsl():
        return _launch_browser_from_wsl(endpoint, preference)
    if _is_linux_native():
        return _launch_browser_on_linux(endpoint, preference)
    return None


def _launch_browser_windows_native(endpoint: str, preference: str) -> str | None:
    """Launch browser directly on Windows using subprocess."""
    parsed = urlparse(endpoint)
    port = parsed.port or 9222
    candidates = _windows_browser_candidates(preference)

    for exe_path in candidates:
        if os.path.exists(exe_path):
            profile_dir = os.path.join(os.environ.get("USERPROFILE", ""), ".agent-first-ide-cdp")
            os.makedirs(profile_dir, exist_ok=True)

            args = [
                exe_path,
                f"--remote-debugging-port={port}",
                "--remote-debugging-address=0.0.0.0",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,SitePerProcess,OptimizeImageLoading",
                "--password-store=basic",
                "--start-maximized",
                f"--user-data-dir={profile_dir}",
            ]

            try:
                subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
                return exe_path
            except Exception as exc:
                logger.warning("Failed to launch %s: %s", exe_path, exc)

    return None


def _launch_browser_from_wsl(endpoint: str, preference: str) -> str | None:
    """Launch Windows browser from WSL using powershell.exe."""
    if shutil.which("powershell.exe") is None:
        return None

    parsed = urlparse(endpoint)
    port = parsed.port or 9222
    candidates = _windows_browser_candidates(preference)
    if not candidates:
        return None

    candidate_list = ",".join(f"'{path}'" for path in candidates)
    ps_script = (
        "$ErrorActionPreference='Stop';"
        f"$candidates=@({candidate_list});"
        "$exe=$candidates | Where-Object { Test-Path $_ } | Select-Object -First 1;"
        "if(-not $exe){ exit 3 };"
        "$profileDir=Join-Path $env:USERPROFILE '.agent-first-ide-cdp';"
        "New-Item -ItemType Directory -Force -Path $profileDir | Out-Null;"
        f"$args=@('--remote-debugging-port={port}','--remote-debugging-address=0.0.0.0','--no-first-run','--no-default-browser-check','--disable-blink-features=AutomationControlled','--disable-features=IsolateOrigins,SitePerProcess,OptimizeImageLoading','--password-store=basic','--start-maximized',\"--user-data-dir=$profileDir\");"
        "Start-Process -FilePath $exe -ArgumentList $args | Out-Null;"
        "Write-Output $exe;"
    )

    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        logger.warning("Windows browser auto-start failed: %s", completed.stderr.strip() or completed.stdout.strip())
        return None

    output = completed.stdout.strip().splitlines()
    return output[-1].strip() if output else "windows-browser"


def _launch_browser_on_linux(endpoint: str, preference: str) -> str | None:
    """Launch Chrome/Chromium on native Linux for CDP control.

    Searches for installed browsers in standard Linux paths and launches
    with CDP debugging enabled and a persistent profile directory.
    """
    parsed = urlparse(endpoint)
    port = parsed.port or 9222
    candidates = _linux_browser_candidates(preference)

    for exe_path in candidates:
        if shutil.which(exe_path) is not None or os.path.isfile(exe_path):
            profile_dir = os.path.expanduser("~/.agent-first-ide-cdp")
            os.makedirs(profile_dir, exist_ok=True)

            args = [
                exe_path,
                f"--remote-debugging-port={port}",
                "--remote-debugging-address=127.0.0.1",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,SitePerProcess,OptimizeImageLoading",
                "--password-store=basic",
                f"--user-data-dir={profile_dir}",
            ]

            # Check if DISPLAY or WAYLAND_DISPLAY is set (needed for headed mode)
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            if not has_display:
                logger.info("No DISPLAY set — launching browser headless on Linux")
                args.append("--headless=new")

            try:
                subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                logger.info("Launched Linux browser: %s (port %d)", exe_path, port)
                return exe_path
            except Exception as exc:
                logger.warning("Failed to launch %s: %s", exe_path, exc)

    return None


def _windows_browser_candidates(preference: str) -> list[str]:
    """Return ordered list of Windows browser executables."""
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    pref = preference.strip().lower()
    if pref == "chrome":
        return chrome_paths + edge_paths
    if pref == "edge":
        return edge_paths + chrome_paths
    return edge_paths + chrome_paths


def _linux_browser_candidates(preference: str) -> list[str]:
    """Return ordered list of Linux browser executables.

    Checks standard system paths for Chrome and Chromium installations.
    Preference can be 'chrome' or 'chromium'.
    """
    chrome_paths = [
        "google-chrome",
        "google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
    ]
    chromium_paths = [
        "chromium-browser",
        "chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/snap/bin/chromium",
    ]

    pref = preference.strip().lower()
    if pref == "chromium":
        return chromium_paths + chrome_paths
    # Default: prefer Chrome, then Chromium
    return chrome_paths + chromium_paths
