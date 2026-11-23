"""Shared browser session bootstrap, signal guard, and failover invocation bridge.

Extracted from legacy `src/advanced_agent.py` so `src/brain_graph.py` and modern
entry points have zero runtime dependency on the legacy monolithic loop, while
preserving `SessionGuard` and persistent-profile launch behavior identically.
`src/advanced_agent.py` re-exports these symbols for 100% backward compatibility.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import logging
import signal
from pathlib import Path
from typing import Any

from paths import PERSISTENCE_DIR as PERSISTENCE_ROOT

logger = logging.getLogger("core.session_bootstrap")

PROFILE_DIR: Path = PERSISTENCE_ROOT / "browser_sessions" / "agent_main"


def ensure_profile_dirs() -> None:
    """Create persistent browser profile directories if missing."""
    PERSISTENCE_ROOT.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def mark_profile_clean_exit() -> None:
    """Patch Chrome's Preferences file so crash-restore bubbles do not appear."""
    prefs_path = PROFILE_DIR / "Default" / "Preferences"
    try:
        if prefs_path.exists():
            data = json.loads(prefs_path.read_text(encoding="utf-8"))
        else:
            prefs_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
        profile = data.setdefault("profile", {})
        profile["exit_type"] = "Normal"
        profile["exited_cleanly"] = True
        prefs_path.write_text(json.dumps(data), encoding="utf-8")
        logger.info("Profile marked clean-exit (crash-restore bubble suppressed)")
    except Exception as exc:
        logger.warning("Could not mark profile clean-exit (non-fatal): %s", exc)


class SessionGuard:
    """Lightweight lifecycle guard ensuring clean browser & Playwright shutdown.

    Preserves the proven §6 contract: `bind_task()` routes SIGINT/SIGTERM to
    cancelling the bound asyncio task so its `finally` awaits `shutdown()`,
    closing the browser context (flushing native `user_data_dir` state) and
    stopping the Playwright driver idempotently.
    """

    _instance: SessionGuard | None = None

    def __init__(self) -> None:
        self._context: Any = None
        self._playwright: Any = None
        self._task: asyncio.Task[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._interrupted: bool = False
        self._installed: bool = False

    @classmethod
    def get(cls) -> SessionGuard:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def attach(self, context: Any, playwright: Any = None) -> None:
        """Attach a live browser context (and optional Playwright driver handle)."""
        self._context = context
        if playwright is not None:
            self._playwright = playwright
        self._ensure_handlers()
        logger.info("SessionGuard: attached to browser context")

    def bind_task(self) -> None:
        """Route SIGINT/SIGTERM to cancelling the calling asyncio task."""
        self._task = asyncio.current_task()
        self._loop = asyncio.get_running_loop()
        self._interrupted = False
        self._ensure_handlers()

    def _ensure_handlers(self) -> None:
        if not self._installed:
            self._install_handlers()
            self._installed = True

    def _install_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, self._signal_handler)
        atexit.register(self._atexit_handler)
        logger.info("SessionGuard: signal + atexit handlers installed")

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        task, loop = self._task, self._loop
        if task is not None and loop is not None and not task.done() and not self._interrupted:
            self._interrupted = True
            logger.warning(
                "SessionGuard: caught %s — shutting down the browser cleanly "
                "(repeat to force quit)...",
                sig_name,
            )
            task.cancel()
            loop.call_soon_threadsafe(lambda: None)
            return
        logger.warning("SessionGuard: caught %s — exiting", sig_name)
        raise KeyboardInterrupt

    async def shutdown(self) -> None:
        """Close the browser context and stop the Playwright driver (idempotent)."""
        context, pw = self._context, self._playwright
        self._context = None
        self._playwright = None
        self._task = None
        self._loop = None
        if context is not None:
            try:
                await context.close()
                logger.info("Browser closed. Profile state persisted.")
            except Exception as exc:
                logger.warning("Browser close failed: %s", exc)
        if pw is not None:
            try:
                await pw.stop()
                logger.info("Playwright driver stopped.")
            except Exception as exc:
                logger.warning("Playwright driver stop failed: %s", exc)

    def _atexit_handler(self) -> None:
        if self._context is not None:
            logger.info("SessionGuard: atexit — detaching context")
            self.detach()

    def detach(self) -> None:
        """Detach the context reference after close."""
        self._context = None


async def launch_browser(*, headless: bool = False) -> tuple[Any, Any]:
    """Launch Playwright Chromium with native persistent profile & §6 stealth wiring."""
    from playwright.async_api import async_playwright

    import dom_parser
    from app.browser_promoter.cdp_stealth_launcher import (
        STEALTH_INIT_SCRIPT,
        STEALTH_LAUNCH_ARGS,
        STEALTH_USER_AGENT,
        VISUAL_CURSOR_INIT_SCRIPT,
        apply_page_stealth,
        get_random_viewport,
    )

    ensure_profile_dirs()
    mark_profile_clean_exit()
    pw = await async_playwright().start()

    logger.info("Launching Playwright Chromium (native profile: %s)", PROFILE_DIR)
    session_viewport = get_random_viewport()
    logger.info(
        "Session viewport: %dx%d",
        session_viewport["width"],
        session_viewport["height"],
    )

    context = await pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=headless,
        viewport={
            "width": int(session_viewport["width"]),
            "height": int(session_viewport["height"]),
        },
        locale="en-US",
        timezone_id="America/New_York",
        user_agent=STEALTH_USER_AGENT,
        java_script_enabled=True,
        device_scale_factor=1,
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        args=STEALTH_LAUNCH_ARGS + dom_parser.TLS_STEALTH_ARGS,
        ignore_default_args=["--enable-automation"],
    )
    await context.add_init_script(STEALTH_INIT_SCRIPT)
    await context.add_init_script(VISUAL_CURSOR_INIT_SCRIPT)
    await dom_parser.install_shadow_piercer(context)

    SessionGuard.get().attach(context, playwright=pw)

    page = context.pages[0] if context.pages else await context.new_page()
    await apply_page_stealth(page)
    with contextlib.suppress(Exception):
        await page.bring_to_front()

    return context, page


async def invoke_with_failover_compat(
    failover_chain: list[Any],
    messages: list[Any],
    schema: Any,
    circuit_breaker: Any = None,
    base64_image: str | None = None,
    health_tracker: Any = None,
) -> tuple[Any, str]:
    """Positional-breaker compatibility wrapper around `model_registry.invoke_with_failover`."""
    from model_registry import invoke_with_failover

    result: tuple[Any, str] = await invoke_with_failover(
        failover_chain,
        messages,
        schema,
        breaker=circuit_breaker,
        health_tracker=health_tracker,
        base64_image=base64_image,
    )
    return result


__all__ = [
    "PERSISTENCE_ROOT",
    "PROFILE_DIR",
    "SessionGuard",
    "ensure_profile_dirs",
    "invoke_with_failover_compat",
    "launch_browser",
    "mark_profile_clean_exit",
]
