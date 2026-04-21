"""Physical OS-level input driver for WSL → Windows browser automation.

.. deprecated:: 2.0.0
    This module is DEPRECATED.  Use ``PlaywrightHumanInput`` from
    ``playwright_human_input.py`` instead.  The Playwright-native driver
    works without OS-level focus, has no PowerShell subprocess overhead,
    and is cross-platform.  Set ``INPUT_DRIVER=playwright`` (default).

    This module is retained ONLY for backward compatibility.
    Set ``INPUT_DRIVER=physical`` to re-enable if needed.

Replaces CDP-based Input.dispatchKeyEvent / dispatchMouseEvent with real
Windows input events delivered via a PowerShell bridge using user32.dll
and System.Windows.Forms.  This ensures the browser receives authentic
OS-level keystrokes and mouse events regardless of CDP focus state.

Design:
    ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
    │ Python (WSL)  │───▶│ powershell.exe   │───▶│ user32.dll /     │
    │ compute path  │    │ -EncodedCommand  │    │ SendKeys         │
    │ + Bézier      │    │ (Base64 script)  │    │ (real OS input)  │
    └──────────────┘    └─────────────────┘    └──────────────────┘

Safety guarantees:
  · FAILSAFE = True equivalent: aborts if cursor is pushed to any screen corner
  · Zero use of BlockInput / SetForegroundWindow locks
  · User retains 100% physical control of mouse and keyboard at all times
"""

from __future__ import annotations

import asyncio
import base64
import math
import random
import subprocess
from dataclasses import dataclass

from playwright.async_api import Page

from app.logger import get_logger

logger = get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

FAILSAFE_CORNER_MARGIN_PX = 5


# ── Exceptions ───────────────────────────────────────────────────────────────

class FailsafeTriggered(RuntimeError):
    """Raised when the cursor is in a screen corner, aborting the action.

    Equivalent to pyautogui.FailSafeException.  Push the mouse to any
    screen corner to instantly abort the agent's physical input.
    """


# ── Internal state ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class _ScreenPointer:
    """Tracks the last known physical cursor position on the Windows desktop."""

    x: int | None = None
    y: int | None = None


# ── SendKeys escaping map ────────────────────────────────────────────────────

_SENDKEYS_SPECIAL: dict[str, str] = {
    "+": "{+}",   # Shift modifier
    "^": "{^}",   # Ctrl modifier
    "%": "{%}",   # Alt modifier
    "~": "{~}",   # Enter
    "(": "{(}",
    ")": "{)}",
    "{": "{{}",   # Literal left brace
    "}": "{}}",   # Literal right brace
}


# ── PowerShell preamble (compiled once per PS process) ───────────────────────
# Uses -MemberDefinition to avoid here-string quoting issues across WSL.

_PS_MOUSE_PREAMBLE = (
    "Add-Type -AssemblyName System.Windows.Forms\n"
    "Add-Type -MemberDefinition "
    "'[DllImport(\"user32.dll\")] "
    "public static extern void mouse_event("
    "int dwFlags, int dx, int dy, int dwData, int dwExtraInfo);' "
    "-Name 'WinMouse' -Namespace 'AgentInput' "
    "-ErrorAction SilentlyContinue\n"
)


# ── Driver ───────────────────────────────────────────────────────────────────

class PhysicalInputDriver:
    """Delivers real OS-level mouse/keyboard events to the Windows desktop.

    Designed for WSL → Windows automation where CDP input events fail because
    the browser doesn't hold OS-level focus.  Uses PowerShell subprocess calls
    to invoke ``user32.dll mouse_event`` and ``System.Windows.Forms.SendKeys``.

    All mouse movement uses cubic Bézier curves for human-like trajectories.
    A failsafe mechanism checks whether the cursor is in any screen corner
    before every action — equivalent to ``pyautogui.FAILSAFE = True``.

    Args:
        chrome_height_override: Manual override for browser chrome height in
            CSS pixels.  Set to 0 (default) for auto-detection.
        dpi_multiplier: Coordinate scale factor.  Use 1.0 for standard setups.
            Only change if coordinates are consistently off on HiDPI screens.
    """

    def __init__(
        self,
        *,
        chrome_height_override: int = 0,
        dpi_multiplier: float = 1.0,
    ) -> None:
        self._pointer = _ScreenPointer()
        self._chrome_height_override = chrome_height_override
        self._dpi_multiplier = dpi_multiplier
        self._cached_metrics: dict | None = None
        self._cached_metrics_url: str = ""

    # ── Public API ────────────────────────────────────────────────────────

    async def physical_click(self, page: Page, x: float, y: float) -> None:
        """Move cursor with Bézier trajectory and left-click at viewport (x, y).

        Args:
            page: Playwright page for coordinate reference.
            x: Viewport X coordinate (CSS pixels from left edge of content area).
            y: Viewport Y coordinate (CSS pixels from top edge of content area).

        Raises:
            FailsafeTriggered: If cursor is in a screen corner before action.
            RuntimeError: If the PowerShell bridge is unavailable.
        """
        # Bring the browser to front (best-effort via CDP, doesn't lock anything)
        await self._ensure_browser_foreground(page)

        screen_x, screen_y = await self._viewport_to_screen(page, x, y)
        start_x, start_y = await self._get_or_track_cursor()
        path_points = self._build_bezier_path(start_x, start_y, screen_x, screen_y)

        await self._execute_move_and_click(path_points, screen_x, screen_y)

        self._pointer.x = screen_x
        self._pointer.y = screen_y

        logger.info(
            "Physical click: viewport (%.0f, %.0f) → screen (%d, %d)",
            x, y, screen_x, screen_y,
        )

    async def physical_type(self, page: Page, text: str) -> dict[str, int]:
        """Type text character-by-character with human-like timing.

        Uses ``System.Windows.Forms.SendKeys.SendWait()`` for each character.
        Includes natural timing variation and occasional typo + correction
        to mimic real human typing.

        Args:
            page: Playwright page (used for failsafe context only).
            text: The text to type.

        Returns:
            Dict with ``typed`` (character count) and ``corrections`` (typo count).

        Raises:
            FailsafeTriggered: If cursor is in a screen corner before action.
        """
        if not text:
            return {"typed": 0, "corrections": 0}

        await self._failsafe_check()

        # Build the sequence with occasional typos (8% chance on alphabetic chars)
        sequence: list[tuple[str, str | None]] = []
        corrections = 0

        for char in text:
            if char.isalpha() and random.random() < 0.08:
                typo = random.choice("abcdefghijklmnopqrstuvwxyz")
                sequence.append(("char", typo))
                sequence.append(("backspace", None))
                corrections += 1
            sequence.append(("char", char))

        await self._execute_type(sequence)

        logger.info(
            "Physical type: %d chars, %d corrections, text=%r",
            len(text), corrections, text[:60],
        )
        return {"typed": len(text), "corrections": corrections}

    async def physical_clear_field(self, page: Page) -> None:
        """Select all text and delete (Ctrl+A → Backspace).

        Replaces the CDP-based ``_clear_with_cdp_shortcut``.
        """
        await self._failsafe_check()

        lines = [
            "Add-Type -AssemblyName System.Windows.Forms",
            self._ps_failsafe_block(),
            "# Select all",
            '[System.Windows.Forms.SendKeys]::SendWait("^a")',
            "Start-Sleep -Milliseconds 60",
            "# Delete selection",
            '[System.Windows.Forms.SendKeys]::SendWait("{BACKSPACE}")',
            "Start-Sleep -Milliseconds 40",
            'Write-Output "CLEAR_OK"',
        ]
        result = await self._run_ps("\n".join(lines))
        if "FAILSAFE_TRIGGERED" in result:
            raise FailsafeTriggered("Cursor in screen corner during clear_field")

        logger.debug("Physical clear field completed")

    async def physical_scroll(self, page: Page, delta_y: float) -> dict[str, float]:
        """Scroll using authentic mouse wheel events via user32.dll.

        Includes overshoot and correction for human-like feel.
        Positive ``delta_y`` = scroll DOWN, negative = scroll UP.
        """
        if abs(delta_y) < 1.0:
            return {"moved": 0.0, "overshoot": 0.0}

        await self._failsafe_check()

        direction = 1 if delta_y >= 0 else -1
        # Windows WHEEL_DELTA = 120 per notch; ~40px per notch on screen
        notches = max(1, int(abs(delta_y) / 40.0))
        overshoot_notches = random.randint(1, 3)
        total = notches + overshoot_notches

        # Windows wheel convention: +120 = scroll UP, -120 = scroll DOWN
        # Web convention: +delta_y = scroll DOWN
        wheel_sign = -direction  # Invert for Windows

        lines = [_PS_MOUSE_PREAMBLE, self._ps_failsafe_block()]

        # Main scroll
        for _ in range(total):
            wheel_data = wheel_sign * 120
            delay = random.randint(18, 40)
            lines.append(
                f"[AgentInput.WinMouse]::mouse_event(0x0800, 0, 0, {wheel_data}, 0)"
            )
            lines.append(f"Start-Sleep -Milliseconds {delay}")

        # Correction (reverse overshoot)
        for _ in range(overshoot_notches):
            wheel_data = -wheel_sign * 120
            delay = random.randint(22, 48)
            lines.append(
                f"[AgentInput.WinMouse]::mouse_event(0x0800, 0, 0, {wheel_data}, 0)"
            )
            lines.append(f"Start-Sleep -Milliseconds {delay}")

        lines.append('Write-Output "SCROLL_OK"')

        result = await self._run_ps("\n".join(lines))
        if "FAILSAFE_TRIGGERED" in result:
            raise FailsafeTriggered("Cursor in screen corner during scroll")

        return {
            "moved": float(delta_y),
            "overshoot": float(overshoot_notches * 40 * direction),
        }

    # ── Coordinate Translation ────────────────────────────────────────────

    async def _viewport_to_screen(
        self, page: Page, vp_x: float, vp_y: float,
    ) -> tuple[int, int]:
        """Translate Playwright viewport coordinates → Windows screen coordinates.

        Uses ``window.screenX/Y`` for window position and computes the browser
        chrome height (tabs + address bar) from ``outerHeight − innerHeight``.

        Chrome CSS pixel coordinates align with ``SetCursorPos`` logical
        coordinates on Windows single-monitor setups at any DPI scale.
        """
        metrics = await self._get_viewport_metrics(page)

        chrome_h = self._chrome_height_override or (
            metrics["outerHeight"] - metrics["innerHeight"]
        )

        screen_x = metrics["screenX"] + vp_x
        screen_y = metrics["screenY"] + chrome_h + vp_y

        # Optional DPI scaling override (1.0 = no change)
        screen_x = int(screen_x * self._dpi_multiplier)
        screen_y = int(screen_y * self._dpi_multiplier)

        return screen_x, screen_y

    async def _get_viewport_metrics(self, page: Page) -> dict:
        """Fetch browser window geometry from the page, with URL-based caching."""
        current_url = page.url
        if self._cached_metrics and self._cached_metrics_url == current_url:
            return self._cached_metrics

        metrics = await page.evaluate(
            """() => ({
                screenX: window.screenX,
                screenY: window.screenY,
                outerWidth: window.outerWidth,
                outerHeight: window.outerHeight,
                innerWidth: window.innerWidth,
                innerHeight: window.innerHeight,
                dpr: window.devicePixelRatio
            })"""
        )

        self._cached_metrics = metrics
        self._cached_metrics_url = current_url

        chrome_h = metrics["outerHeight"] - metrics["innerHeight"]
        logger.debug(
            "Viewport metrics: window(%d,%d) outer(%dx%d) inner(%dx%d) "
            "dpr=%.1f chrome_h=%dpx",
            metrics["screenX"], metrics["screenY"],
            metrics["outerWidth"], metrics["outerHeight"],
            metrics["innerWidth"], metrics["innerHeight"],
            metrics["dpr"], chrome_h,
        )

        return metrics

    # ── Bézier Path ───────────────────────────────────────────────────────

    def _build_bezier_path(
        self,
        start_x: int, start_y: int,
        end_x: int, end_y: int,
    ) -> list[tuple[int, int]]:
        """Compute cubic Bézier curve points for natural mouse movement.

        Uses the same physics model as the original CDPHumanBehavior to
        maintain identical trajectory characteristics.
        """
        distance = math.hypot(end_x - start_x, end_y - start_y)
        steps = max(12, min(40, int(distance / 18.0)))

        # Random control points produce natural curvature
        ctrl1_x = start_x + (end_x - start_x) * random.uniform(0.2, 0.45) + random.uniform(-35, 35)
        ctrl1_y = start_y + (end_y - start_y) * random.uniform(0.2, 0.45) + random.uniform(-30, 30)
        ctrl2_x = start_x + (end_x - start_x) * random.uniform(0.55, 0.8) + random.uniform(-30, 30)
        ctrl2_y = start_y + (end_y - start_y) * random.uniform(0.55, 0.8) + random.uniform(-25, 25)

        points: list[tuple[int, int]] = []
        for step in range(1, steps + 1):
            t = step / steps
            px = self._cubic_bezier(t, start_x, ctrl1_x, ctrl2_x, end_x)
            py = self._cubic_bezier(t, start_y, ctrl1_y, ctrl2_y, end_y)
            points.append((int(px), int(py)))

        return points

    @staticmethod
    def _cubic_bezier(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
        """Evaluate cubic Bézier at parameter t ∈ [0, 1]."""
        u = 1.0 - t
        return (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3

    # ── Failsafe ──────────────────────────────────────────────────────────

    async def _failsafe_check(self) -> None:
        """Check if cursor is in a screen corner.  If so, raise FailsafeTriggered.

        Equivalent to ``pyautogui.FAILSAFE = True``.  The user can abort any
        agent action instantly by pushing the mouse to any screen corner.
        """
        lines = [
            "Add-Type -AssemblyName System.Windows.Forms",
            "$pos = [System.Windows.Forms.Cursor]::Position",
            "$scr = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds",
            f"$m = {FAILSAFE_CORNER_MARGIN_PX}",
            "if (($pos.X -le $m -and $pos.Y -le $m) -or",
            "    ($pos.X -ge ($scr.Width-$m) -and $pos.Y -le $m) -or",
            "    ($pos.X -le $m -and $pos.Y -ge ($scr.Height-$m)) -or",
            "    ($pos.X -ge ($scr.Width-$m) -and $pos.Y -ge ($scr.Height-$m))) {",
            '    Write-Output "FAILSAFE_TRIGGERED"',
            "    exit 0",
            "}",
            'Write-Output "SAFE $($pos.X) $($pos.Y)"',
        ]
        result = await self._run_ps("\n".join(lines))

        if "FAILSAFE_TRIGGERED" in result:
            raise FailsafeTriggered(
                "FAILSAFE: Cursor is in a screen corner. "
                "Agent action aborted. Move cursor away from corners to resume."
            )

        # Update tracked position from the response
        parts = result.strip().split()
        if len(parts) >= 3 and parts[0] == "SAFE":
            try:
                self._pointer.x = int(parts[1])
                self._pointer.y = int(parts[2])
            except (ValueError, IndexError):
                pass

    async def _get_or_track_cursor(self) -> tuple[int, int]:
        """Return the current cursor position, querying Windows if not tracked."""
        if self._pointer.x is not None and self._pointer.y is not None:
            return self._pointer.x, self._pointer.y

        lines = [
            "Add-Type -AssemblyName System.Windows.Forms",
            "$pos = [System.Windows.Forms.Cursor]::Position",
            'Write-Output "$($pos.X) $($pos.Y)"',
        ]
        result = await self._run_ps("\n".join(lines))
        parts = result.strip().split()
        x = int(parts[0]) if len(parts) >= 1 else 960
        y = int(parts[1]) if len(parts) >= 2 else 540
        self._pointer.x = x
        self._pointer.y = y
        return x, y

    # ── PowerShell Script Builders ────────────────────────────────────────

    async def _execute_move_and_click(
        self,
        path_points: list[tuple[int, int]],
        target_x: int,
        target_y: int,
    ) -> None:
        """Build and execute a single PowerShell script for smooth move + click."""
        points_str = ",".join(f"@({x},{y})" for x, y in path_points)
        hold_ms = random.randint(45, 110)

        lines = [
            _PS_MOUSE_PREAMBLE,
            self._ps_failsafe_block(),
            "# ── Smooth Bézier movement ──",
            "$rng = New-Object System.Random",
            f"$points = @({points_str})",
            "foreach ($p in $points) {",
            "    [System.Windows.Forms.Cursor]::Position = "
            "New-Object System.Drawing.Point($p[0], $p[1])",
            "    Start-Sleep -Milliseconds ($rng.Next(8, 20))",
            "}",
            "",
            "# ── Physical click ──",
            "[AgentInput.WinMouse]::mouse_event(0x0002, 0, 0, 0, 0)",
            f"Start-Sleep -Milliseconds {hold_ms}",
            "[AgentInput.WinMouse]::mouse_event(0x0004, 0, 0, 0, 0)",
            f'Write-Output "CLICK_OK {target_x} {target_y}"',
        ]

        result = await self._run_ps("\n".join(lines))
        if "FAILSAFE_TRIGGERED" in result:
            raise FailsafeTriggered(
                "FAILSAFE: Cursor in screen corner before click. Action aborted."
            )

    async def _execute_type(self, sequence: list[tuple[str, str | None]]) -> None:
        """Build and execute a single PowerShell script for character-by-character typing."""
        type_lines: list[str] = []

        for kind, char in sequence:
            if kind == "backspace":
                delay = random.randint(30, 120)
                type_lines.append(
                    "[System.Windows.Forms.SendKeys]::SendWait('{BACKSPACE}')"
                )
                type_lines.append(f"Start-Sleep -Milliseconds {delay}")
            elif kind == "char" and char is not None:
                escaped = self._escape_for_sendkeys(char)
                if not escaped:
                    continue  # Skip null/CR characters

                # Make safe for PowerShell single-quoted string
                ps_safe = escaped.replace("'", "''")
                delay = random.randint(30, 120)

                type_lines.append(
                    f"[System.Windows.Forms.SendKeys]::SendWait('{ps_safe}')"
                )
                type_lines.append(f"Start-Sleep -Milliseconds {delay}")

        lines = [
            "Add-Type -AssemblyName System.Windows.Forms",
            self._ps_failsafe_block(),
            "# ── Type characters ──",
            *type_lines,
            'Write-Output "TYPE_OK"',
        ]

        result = await self._run_ps("\n".join(lines))
        if "FAILSAFE_TRIGGERED" in result:
            raise FailsafeTriggered(
                "FAILSAFE: Cursor in screen corner before typing. Action aborted."
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _escape_for_sendkeys(char: str) -> str:
        """Escape a character for ``System.Windows.Forms.SendKeys.SendWait()``.

        Handles modifier keys (+^%~), braces, parentheses, and control
        characters (newline → ENTER, tab → TAB).
        """
        if char == "\n":
            return "{ENTER}"
        if char == "\t":
            return "{TAB}"
        if char == "\r":
            return ""  # Skip carriage return
        return _SENDKEYS_SPECIAL.get(char, char)

    @staticmethod
    def _ps_failsafe_block() -> str:
        """Return a PowerShell code block that aborts if cursor is in a corner."""
        m = FAILSAFE_CORNER_MARGIN_PX
        return "\n".join([
            "$pos = [System.Windows.Forms.Cursor]::Position",
            "$scr = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds",
            f"$m = {m}",
            "if (($pos.X -le $m -and $pos.Y -le $m) -or",
            "    ($pos.X -ge ($scr.Width-$m) -and $pos.Y -le $m) -or",
            "    ($pos.X -le $m -and $pos.Y -ge ($scr.Height-$m)) -or",
            f"    ($pos.X -ge ($scr.Width-$m) -and $pos.Y -ge ($scr.Height-$m))) {{",
            '    Write-Output "FAILSAFE_TRIGGERED"',
            "    exit 0",
            "}",
        ])

    @staticmethod
    async def _ensure_browser_foreground(page: Page) -> None:
        """Best-effort attempt to bring the browser tab to the OS foreground.

        Uses Playwright's ``bring_to_front()`` and raw CDP ``Page.bringToFront``.
        This does NOT lock any input — it's a polite window-manager request.
        """
        try:
            await page.bring_to_front()
        except Exception:
            pass

        try:
            cdp = await page.context.new_cdp_session(page)
            await cdp.send("Page.bringToFront")
            await cdp.detach()
        except Exception:
            pass

    async def _run_ps(self, script: str) -> str:
        """Execute a PowerShell script from WSL using ``-EncodedCommand``.

        The script is Base64-encoded as UTF-16LE to avoid all quoting and
        escaping issues when crossing the WSL → Windows boundary.

        Returns:
            The captured stdout from the PowerShell process.

        Raises:
            RuntimeError: If ``powershell.exe`` is not found or times out.
        """
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy", "Bypass",
                    "-EncodedCommand", encoded,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "powershell.exe not found. Physical input requires WSL with "
                "Windows interop enabled (ensure /proc/sys/fs/binfmt_misc/ is active)."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "PowerShell input command timed out after 30s. "
                "Check if the Windows desktop is responsive."
            )

        if result.returncode != 0 and "FAILSAFE_TRIGGERED" not in result.stdout:
            stderr = result.stderr.strip()
            if stderr:
                logger.warning("PowerShell stderr: %s", stderr[:500])

        return result.stdout.strip()
