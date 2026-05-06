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

# Track cursor position across calls
_last_mouse_pos: tuple[float, float] = (400.0, 300.0)


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


async def ghost_type(page: Page, text: str) -> None:
    """Type text with human-like variable speed and occasional typo corrections.
    
    Replaces: await page.keyboard.type(text, delay=30)
    """
    i = 0
    while i < len(text):
        char = text[i]

        # Occasional typo (type wrong key, pause, backspace, retype correct)
        if char.isalpha() and _should_make_typo():
            wrong = _nearby_key(char)
            await page.keyboard.press(wrong)
            await asyncio.sleep(random.uniform(0.15, 0.40))  # "realize mistake" pause
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
        if char in " \n\t":
            await asyncio.sleep(_word_boundary_pause())
        elif char in ".,!?;:":
            await asyncio.sleep(random.uniform(0.10, 0.30))  # Punctuation pause
        else:
            await asyncio.sleep(_typing_delay())

        i += 1

    logger.info("Ghost typed %d chars with humanized cadence", len(text))


# ═══════════════════════════════════════════════════════════════════════════════
#  Ghost Scroll — Chunked scrolling with entropy
# ═══════════════════════════════════════════════════════════════════════════════

async def ghost_scroll(page: Page, delta_y: int = 600) -> None:
    """Scroll in human-like incremental chunks instead of one instant jump.
    
    Replaces: await page.mouse.wheel(0, delta)
    """
    remaining = abs(delta_y)
    direction = 1 if delta_y > 0 else -1
    chunks = random.randint(2, 4)

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
    logger.info("Ghost scrolled %dpx in %d chunks", abs(delta_y), chunks)
