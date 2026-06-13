"""GhostInput — Humanized browser interaction layer.

Replaces raw Playwright mouse.click() / keyboard.type() calls with
Bézier-curve mouse paths, variable-speed typing with occasional typos,
and entropy-injected scrolling to bypass behavioral bot detection.

Drop-in replacement:
  - ghost_click(page, x, y)   → replaces page.mouse.click(x, y)
  - ghost_type(page, text)    → replaces page.keyboard.type(text, delay=30)
  - ghost_scroll(page, delta) → replaces page.mouse.wheel(0, delta)
"""

from __future__ import annotations

import asyncio
import math
import random
import string
from typing import Sequence

from playwright.async_api import Page

from app.logger import get_logger

logger = get_logger("ghost_input")

# ═══════════════════════════════════════════════════════════════════════════════
#  Bézier Curve Math
# ═══════════════════════════════════════════════════════════════════════════════

def _cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate a cubic Bézier curve at parameter t ∈ [0, 1]."""
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def _generate_bezier_path(
    start: tuple[float, float],
    end: tuple[float, float],
    num_points: int = 20,
    overshoot: float = 0.15,
) -> list[tuple[float, float]]:
    """Generate a human-like Bézier mouse path from start to end.
    
    The control points are randomized to create natural-looking curves.
    An optional overshoot factor makes the cursor slightly overshoot
    the target before settling (mimicking human motor correction).
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.sqrt(dx**2 + dy**2)

    # Randomize control points with perpendicular offset
    perp_x, perp_y = -dy, dx  # Perpendicular vector
    if distance > 0:
        perp_x /= distance
        perp_y /= distance

    # Control point 1: ~30% along the path + random perpendicular offset
    offset1 = random.uniform(-0.3, 0.3) * distance
    cp1 = (
        start[0] + dx * 0.3 + perp_x * offset1,
        start[1] + dy * 0.3 + perp_y * offset1,
    )

    # Control point 2: ~70% along the path + random perpendicular offset
    offset2 = random.uniform(-0.2, 0.2) * distance
    cp2 = (
        start[0] + dx * 0.7 + perp_x * offset2,
        start[1] + dy * 0.7 + perp_y * offset2,
    )

    # Optionally overshoot the target slightly
    if random.random() < 0.3 and distance > 50:
        overshoot_end = (
            end[0] + dx * overshoot * random.uniform(0.5, 1.0) / max(distance, 1),
            end[1] + dy * overshoot * random.uniform(0.5, 1.0) / max(distance, 1),
        )
    else:
        overshoot_end = end

    # Generate points along the Bézier curve
    path = []
    for i in range(num_points):
        t = i / (num_points - 1)
        # Ease-in-out timing (slow at start and end, fast in middle)
        t_eased = 0.5 - 0.5 * math.cos(t * math.pi)
        point = _cubic_bezier(start, cp1, cp2, overshoot_end, t_eased)
        # Add micro-jitter (±1-3 pixels) to prevent perfectly smooth paths
        jitter_x = random.gauss(0, 1.5)
        jitter_y = random.gauss(0, 1.5)
        path.append((point[0] + jitter_x, point[1] + jitter_y))

    # If we overshot, add correction points back to actual target
    if overshoot_end != end:
        correction_points = random.randint(3, 5)
        for i in range(1, correction_points + 1):
            t = i / correction_points
            x = overshoot_end[0] + (end[0] - overshoot_end[0]) * t
            y = overshoot_end[1] + (end[1] - overshoot_end[1]) * t
            path.append((x + random.gauss(0, 0.5), y + random.gauss(0, 0.5)))

    # Ensure the final point is exactly the target
    path.append(end)
    return path


def _movement_delay(distance: float) -> float:
    """Calculate delay between mouse movement steps based on distance.
    
    Shorter distances = slower (human precision), longer = faster (ballistic).
    Returns delay in seconds.
    """
    base_delay = 0.008  # 8ms base
    if distance < 100:
        return base_delay * random.uniform(1.5, 3.0)
    elif distance < 400:
        return base_delay * random.uniform(0.8, 1.5)
    else:
        return base_delay * random.uniform(0.5, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ghost Click — Bézier mouse path + click
# ═══════════════════════════════════════════════════════════════════════════════

# Track cursor position across calls — the SINGLE source of truth for where the
# mouse is. Every mouse op (move/click/scroll) keeps this current so the agent
# always has accurate, continuous awareness of the exact coordinates.
_last_mouse_pos: tuple[float, float] = (400.0, 300.0)


def get_mouse_pos() -> tuple[float, float]:
    """Current tracked mouse position (the exact coords of the last mouse op)."""
    return _last_mouse_pos


def set_mouse_pos(x: float, y: float) -> None:
    """Record a new mouse position (used by callers that move the mouse directly,
    e.g. the CDP click engine, so awareness never drifts)."""
    global _last_mouse_pos
    _last_mouse_pos = (float(x), float(y))


async def resync_visual_cursor(page: Page, x: float | None = None, y: float | None = None) -> None:
    """Place the on-screen arrow at the tracked position (or an explicit one).

    A fresh page (after navigation) starts with the real mouse at (0,0) and the
    visual cursor hidden. Calling this right after a navigation moves the arrow to
    where the agent believes the cursor is — so it never flashes/jumps to the
    top-left corner, and any screenshot the vision model takes shows a truthful
    cursor. Best-effort; never raises.
    """
    px, py = (x, y) if x is not None and y is not None else _last_mouse_pos
    try:
        await page.evaluate(
            "([x,y]) => { if (window.__setCursorPos) window.__setCursorPos(x, y); }",
            [float(px), float(py)],
        )
    except Exception:
        pass


async def ghost_click(page: Page, x: float, y: float) -> None:
    """Move the mouse along a Bézier curve to (x, y) and click.
    
    Replaces: await page.mouse.click(x, y)
    """
    global _last_mouse_pos

    start = _last_mouse_pos
    end = (float(x), float(y))
    distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

    # Scale path complexity with distance
    num_points = max(8, min(30, int(distance / 15)))
    path = _generate_bezier_path(start, end, num_points=num_points)

    step_delay = _movement_delay(distance)

    for point in path:
        await page.mouse.move(point[0], point[1])
        await asyncio.sleep(step_delay + random.uniform(0, 0.004))

    # Small pre-click pause (humans hesitate slightly before clicking)
    await asyncio.sleep(random.uniform(0.05, 0.15))

    await page.mouse.click(end[0], end[1])
    _last_mouse_pos = end

    # Small post-click pause
    await asyncio.sleep(random.uniform(0.1, 0.25))


async def ghost_move_to(page: Page, x: float, y: float) -> None:
    """Move the mouse along a Bézier curve to (x, y) WITHOUT clicking.
    
    Used by the CDP native click engine: ghost_move_to() handles the
    human-like mouse path, then cdp_click dispatches the actual click
    via CDP Input.dispatchMouseEvent for isTrusted=true events.
    """
    global _last_mouse_pos

    start = _last_mouse_pos
    end = (float(x), float(y))
    distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

    # Scale path complexity with distance
    num_points = max(8, min(30, int(distance / 15)))
    path = _generate_bezier_path(start, end, num_points=num_points)

    step_delay = _movement_delay(distance)

    for point in path:
        await page.mouse.move(point[0], point[1])
        await asyncio.sleep(step_delay + random.uniform(0, 0.004))

    _last_mouse_pos = end

    # Small pre-click pause (humans hesitate slightly before clicking)
    await asyncio.sleep(random.uniform(0.05, 0.15))


# ═══════════════════════════════════════════════════════════════════════════════
#  Ghost Type — Variable-speed typing with occasional corrections
# ═══════════════════════════════════════════════════════════════════════════════

def _typing_delay() -> float:
    """Return a human-like delay between keystrokes (seconds).
    
    Uses a gaussian distribution centered at 70ms with std dev of 25ms.
    """
    delay = random.gauss(0.070, 0.025)
    return max(0.025, min(0.180, delay))  # Clamp between 25ms-180ms


def _word_boundary_pause() -> float:
    """Return a longer pause at word boundaries (space, period, newline)."""
    return random.uniform(0.08, 0.20)


def _should_make_typo() -> bool:
    """~1.5% chance of making a typo per character."""
    return random.random() < 0.015


def _nearby_key(char: str) -> str:
    """Return a plausible adjacent key for a typo."""
    keyboard_neighbors: dict[str, str] = {
        "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr",
        "f": "dg", "g": "fh", "h": "gj", "i": "uo", "j": "hk",
        "k": "jl", "l": "k;", "m": "n,", "n": "bm", "o": "ip",
        "p": "o[", "q": "wa", "r": "et", "s": "ad", "t": "ry",
        "u": "yi", "v": "cb", "w": "qe", "x": "zc", "y": "tu",
        "z": "xa",
    }
    lower = char.lower()
    neighbors = keyboard_neighbors.get(lower, "")
    if neighbors:
        typo = random.choice(neighbors)
        return typo.upper() if char.isupper() else typo
    return char


# ── Safety Constants ──
GHOST_TYPE_MAX_LENGTH = 8000
GHOST_TYPE_FAST_THRESHOLD = 500
GHOST_TYPE_NO_TYPO_THRESHOLD = 100
GHOST_TYPE_ABORT_CHECK_INTERVAL = 50


def _fast_typing_delay() -> float:
    """Faster delay for long text (15ms avg instead of 70ms)."""
    delay = random.gauss(0.015, 0.005)
    return max(0.008, min(0.035, delay))


async def ghost_type(page: Page, text: str) -> None:
    """Type text with human-like variable speed and occasional typo corrections.
    
    Safety guards:
      - Truncates to GHOST_TYPE_MAX_LENGTH (2000) chars
      - Switches to fast mode for text > 500 chars
      - Disables typo simulation for text > 100 chars
      - Checks page.is_closed() every 50 chars to abort if page died
    
    Replaces: await page.keyboard.type(text, delay=30)
    """
    # ── Length guard ──
    original_len = len(text)
    if original_len > GHOST_TYPE_MAX_LENGTH:
        logger.warning(
            "Ghost type truncated: %d → %d chars", original_len, GHOST_TYPE_MAX_LENGTH
        )
        text = text[:GHOST_TYPE_MAX_LENGTH]

    fast_mode = len(text) > GHOST_TYPE_FAST_THRESHOLD
    typos_enabled = len(text) <= GHOST_TYPE_NO_TYPO_THRESHOLD

    if fast_mode:
        logger.info("Ghost type: fast mode enabled (%d chars)", len(text))

    i = 0
    while i < len(text):
        # ── Abort check: stop if page closed mid-type ──
        if i > 0 and i % GHOST_TYPE_ABORT_CHECK_INTERVAL == 0:
            try:
                if page.is_closed():
                    logger.warning("Ghost type aborted: page closed at char %d/%d", i, len(text))
                    return
            except Exception:
                return

        char = text[i]

        # Occasional typo (only for short text)
        if typos_enabled and char.isalpha() and _should_make_typo():
            wrong = _nearby_key(char)
            await page.keyboard.press(wrong)
            await asyncio.sleep(random.uniform(0.15, 0.40))
            await page.keyboard.press("Backspace")
            await asyncio.sleep(random.uniform(0.05, 0.12))
            await page.keyboard.press(char)
        else:
            if char == "\n":
                await page.keyboard.press("Enter")
            else:
                try:
                    await page.keyboard.press(char)
                except Exception:
                    # Fallback for unicode characters like non-breaking hyphens
                    await page.keyboard.insert_text(char)

        # Delay after this character
        if fast_mode:
            await asyncio.sleep(_fast_typing_delay())
        elif char in " \n\t":
            await asyncio.sleep(_word_boundary_pause())
        elif char in ".,!?;:":
            await asyncio.sleep(random.uniform(0.10, 0.30))
        else:
            await asyncio.sleep(_typing_delay())

        i += 1

    mode_label = "fast" if fast_mode else "humanized"
    logger.info("Ghost typed %d chars with %s cadence", len(text), mode_label)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ghost Scroll — Chunked scrolling with entropy
# ═══════════════════════════════════════════════════════════════════════════════

# Read the current vertical scroll offset (window, or the document scrolling element).
_SCROLL_Y_JS = """
() => {
  const se = document.scrollingElement || document.documentElement || document.body;
  return Math.round(window.scrollY || (se && se.scrollTop) || 0);
}
"""

# Deterministic fallback when the wheel is swallowed: scroll the window, and if
# the window itself does not move (content lives in a nested scroll container, or
# a sticky region ate the wheel), scroll the largest scrollable container instead.
_FORCE_SCROLL_JS = """
(dy) => {
  const se = document.scrollingElement || document.documentElement || document.body;
  const before = window.scrollY || (se && se.scrollTop) || 0;
  window.scrollBy(0, dy);
  const after = window.scrollY || (se && se.scrollTop) || 0;
  if (Math.abs(after - before) >= 5) return after - before;
  // Window did not budge — find the dominant scrollable element and scroll it.
  let best = null, bestArea = 0;
  for (const el of document.querySelectorAll('div, main, section, ul, article, [style*=overflow]')) {
    if (el.scrollHeight - el.clientHeight <= 20) continue;
    const oy = getComputedStyle(el).overflowY;
    if (oy !== 'auto' && oy !== 'scroll') continue;
    const r = el.getBoundingClientRect();
    const area = r.width * r.height;
    if (area > bestArea) { bestArea = area; best = el; }
  }
  if (best) { const b = best.scrollTop; best.scrollBy(0, dy); return best.scrollTop - b; }
  return 0;
}
"""


async def ghost_scroll(page: Page, delta_y: int = 600) -> None:
    """Scroll in human-like chunks, then GUARANTEE the viewport actually moved.

    page.mouse.wheel() dispatches at the current cursor position, so after a
    click (cursor left hovering a product tile or a sticky column) the wheel can
    be swallowed and the viewport never moves — the action "succeeds" yet the
    screen is unchanged. That is the #1 reason the agent never reaches off-screen
    controls like "Add to Cart" and just re-scrolls forever. We park the cursor
    over the main content first, perform the humanized wheel, then verify the
    scroll offset changed and fall back to a deterministic programmatic scroll if
    it did not.

    Replaces: await page.mouse.wheel(0, delta)
    """
    remaining = abs(delta_y)
    direction = 1 if delta_y > 0 else -1
    chunks = random.randint(2, 4)

    # Park the cursor over the centre of the viewport so the wheel targets the
    # main scroller, not whatever element the previous click left it hovering.
    try:
        vp = page.viewport_size or {"width": 1280, "height": 800}
        cx, cy = vp["width"] // 2, vp["height"] // 2
        await page.mouse.move(cx, cy)
        global _last_mouse_pos
        _last_mouse_pos = (float(cx), float(cy))  # keep position-awareness in sync
    except Exception:
        pass

    try:
        before = await page.evaluate(_SCROLL_Y_JS)
    except Exception:
        before = 0

    for i in range(chunks):
        if remaining <= 0:
            break

        # Each chunk is a random portion of the remaining distance
        if i == chunks - 1:
            chunk = remaining  # Last chunk takes whatever's left
        else:
            chunk = int(remaining * random.uniform(0.25, 0.55))

        await page.mouse.wheel(0, chunk * direction)
        remaining -= chunk

        # Random pause between scroll chunks
        await asyncio.sleep(random.uniform(0.15, 0.45))

    # Small settling pause after scrolling
    await asyncio.sleep(random.uniform(0.3, 0.7))

    # Verify the wheel actually moved the page; if not, force it deterministically
    # so "scroll" always changes what perception sees.
    try:
        after = await page.evaluate(_SCROLL_Y_JS)
    except Exception:
        after = before

    if abs(after - before) < max(20, int(abs(delta_y) * 0.25)):
        try:
            moved = await page.evaluate(_FORCE_SCROLL_JS, delta_y)
            logger.info("Ghost scroll: wheel ineffective (Δ%dpx) → forced scroll Δ%dpx",
                        after - before, moved)
        except Exception as e:
            logger.warning("Ghost scroll force-fallback failed: %s", e)
    else:
        logger.info("Ghost scrolled %dpx in %d chunks (Δ%dpx)", abs(delta_y), chunks, after - before)
