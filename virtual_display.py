"""Virtual display — run the browser HEADED on a display-less machine (V22 anti-bot).

WHY
═══
A real headed Chrome and a `--headless` Chrome differ in dozens of ways anti-bot
vendors fingerprint: the User-Agent ("HeadlessChrome"), missing window/screen
geometry, different GPU/`navigator` surfaces, no real compositor, subtle
rendering and timing tells. Our JS stealth layer hides many; the cleanest fix is
to NOT be headless at all.

On WSL2 / servers there's no physical display, so headed normally can't launch.
Xvfb (X virtual framebuffer) gives a real, in-memory X server the browser draws
into — genuinely headed and invisible to the user, but presenting as a normal
headed Chrome to the page. The single biggest realism win.

We drive Xvfb directly via subprocess (pyvirtualdisplay's Display.start() hangs
in some WSL2 setups). Everything degrades gracefully if Xvfb is missing — no
crash, just `--headless`.

USAGE
═════
Call `start_virtual_display()` before launching the browser: True → a display is
available, launch HEADED; False → fall back to `--headless`. `stop_virtual_display()`
tears it down (also auto-registered with atexit). Needs `xvfb`
(`sudo apt-get install -y xvfb`).
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import time

try:
    from app.logger import get_logger
    logger = get_logger("virtual_display")
except ImportError:
    logger = logging.getLogger("virtual_display")

_PROC: subprocess.Popen | None = None      # the Xvfb process we own
_OWNED_DISPLAY: str | None = None           # the ":N" we set (None if real display)

XVFB_INSTALL_HINT = "install it with:  sudo apt-get install -y xvfb"


def xvfb_available() -> bool:
    """True if the Xvfb binary is on PATH."""
    return shutil.which("Xvfb") is not None


def _pick_display_number() -> int:
    """Find a free X display number (avoids clashing with an existing Xvfb)."""
    for n in range(99, 120):
        if not os.path.exists(f"/tmp/.X{n}-lock"):
            return n
    return 99


def start_virtual_display(width: int = 1920, height: int = 1080) -> bool:
    """Ensure a usable X display exists so the browser can run HEADED.

    Returns:
        True  → a display is available (real, or a virtual one we just started)
                → caller should launch the browser headed (headless=False).
        False → no display and no/failed Xvfb → caller must use --headless.
    """
    global _PROC, _OWNED_DISPLAY

    # 1. A real display (WSLg, X server, VNC) is already present — use it.
    real = os.environ.get("DISPLAY", "").strip()
    if real:
        logger.info("🖥️ Using existing display %s — browser runs HEADED", real)
        return True

    # 2. No display — need the Xvfb binary to fabricate one.
    if not xvfb_available():
        logger.warning(
            "🖥️ No display and Xvfb not installed — running --headless "
            "(more bot-detectable). For stealth, %s", XVFB_INSTALL_HINT)
        return False

    # 3. Start Xvfb directly and point DISPLAY at it.
    n = _pick_display_number()
    disp = f":{n}"
    try:
        _PROC = subprocess.Popen(
            ["Xvfb", disp, "-screen", "0", f"{width}x{height}x24",
             "-nolisten", "tcp", "-ac"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.warning("Xvfb launch failed (%s) — falling back to --headless", e)
        _PROC = None
        return False

    # Wait for the server to come up (lock file appears) — bounded, no hang.
    lock = f"/tmp/.X{n}-lock"
    for _ in range(60):  # up to ~6s
        if _PROC.poll() is not None:
            logger.warning("Xvfb exited early (rc=%s) — falling back to --headless",
                           _PROC.returncode)
            _PROC = None
            return False
        if os.path.exists(lock):
            break
        time.sleep(0.1)
    else:
        logger.warning("Xvfb did not become ready in time — falling back to --headless")
        _stop_safely()
        return False

    os.environ["DISPLAY"] = disp
    _OWNED_DISPLAY = disp
    atexit.register(stop_virtual_display)
    logger.info("🖥️ Virtual display %s started (%dx%d) — browser runs HEADED (stealth)",
                disp, width, height)
    return True


def stop_virtual_display() -> None:
    """Tear down the virtual display we own (no-op for a real display)."""
    _stop_safely()


def _stop_safely() -> None:
    global _PROC, _OWNED_DISPLAY
    if _PROC is not None:
        try:
            _PROC.terminate()
            try:
                _PROC.wait(timeout=5)
            except Exception:
                _PROC.kill()
            logger.info("🖥️ Virtual display stopped")
        except Exception as e:
            logger.debug("Virtual display stop failed (non-fatal): %s", e)
        _PROC = None
    if _OWNED_DISPLAY and os.environ.get("DISPLAY") == _OWNED_DISPLAY:
        os.environ.pop("DISPLAY", None)
    _OWNED_DISPLAY = None
