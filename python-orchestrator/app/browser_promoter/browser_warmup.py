"""Browser Warm-Up Routine — Human Behavior Simulation Layer.

Provides a pre-navigation warm-up sequence that mimics real human browser
behavior to defeat WAF/bot-detection systems. Runs ONCE after browser launch
and BEFORE the agent enters its action loop.

The warm-up establishes:
  - A realistic browsing "origin story" (human opened browser → homepage → target)
  - Mouse telemetry that passes behavioral analysis
  - Natural cookie/JS warm-up on the target domain
  - Credible referrer chain in the browser history

Drop-in usage:
    from app.browser_promoter.browser_warmup import run_warmup
    await run_warmup(page, target_url="https://dev.to/new")
"""

from __future__ import annotations

import asyncio
import random
from urllib.parse import urlparse

from playwright.async_api import Page

from app.logger import get_logger

logger = get_logger("browser_warmup")

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Neutral high-trust sites the "user" might visit as a homepage.
# These are fast-loading, universally accessible, and never WAF-blocked.
_HOMEPAGE_POOL = [
    "https://www.google.com/",
    "https://www.google.com/search?q=weather",
    "https://en.wikipedia.org/wiki/Main_Page",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Core Warm-Up Sequence
# ═══════════════════════════════════════════════════════════════════════════════

async def run_warmup(page: Page, *, target_url: str = "") -> None:
    """Execute the full human warm-up routine.

    Sequence:
      1. Homepage visit (Google/Wikipedia)
      2. Random reading delay
      3. Fake idle scroll (down, pause, up)
      4. Mouse wiggle across viewport
      5. Page refresh jitter
      6. Target domain priming (base domain visit)
      7. Cookie/JS warm-up wait

    Args:
        page: Live Playwright page object.
        target_url: The deep URL the agent will eventually navigate to.
                    Used to extract the base domain for priming.
    """
    logger.info("🔥 Starting human warm-up routine...")

    # ── Step 1: Homepage visit ──
    homepage = random.choice(_HOMEPAGE_POOL)
    logger.info("  [1/7] Homepage warm-up → %s", homepage)
    try:
        await page.goto(homepage, wait_until="domcontentloaded", timeout=15000)
    except Exception as e:
        logger.warning("  Homepage navigation failed (%s) — continuing...", e)

    # ── Step 2: Random reading delay ──
    reading_delay = random.uniform(2.5, 5.2)
    logger.info("  [2/7] Reading delay: %.1fs", reading_delay)
    await asyncio.sleep(reading_delay)

    # ── Step 3: Fake idle scroll ──
    logger.info("  [3/7] Idle scroll simulation")
    await _fake_scroll(page)

    # ── Step 4: Mouse wiggle ──
    logger.info("  [4/7] Mouse wiggle (generating telemetry)")
    await _mouse_wiggle(page)

    # ── Step 5: Page refresh jitter ──
    logger.info("  [5/7] Refresh jitter")
    try:
        await page.reload(wait_until="networkidle", timeout=12000)
    except Exception:
        pass
    await asyncio.sleep(random.uniform(1.0, 2.5))

    # ── Step 6: Target domain priming ──
    base_domain = _extract_base_url(target_url)
    if base_domain:
        logger.info("  [6/7] Target domain priming → %s", base_domain)
        try:
            await page.goto(base_domain, wait_until="domcontentloaded", timeout=15000)
        except Exception as e:
            logger.warning("  Domain priming failed (%s) — continuing...", e)

        # Wait for cookies, tracking JS, and session restoration
        prime_delay = random.uniform(2.0, 4.0)
        await asyncio.sleep(prime_delay)

        # Quick mouse movement on the target domain to register interaction
        await _mouse_wiggle(page, movements=2)
    else:
        logger.info("  [6/7] No target URL provided — skipping domain priming")

    # ── Step 7: Cookie/JS warm-up wait ──
    logger.info("  [7/7] Final warm-up wait (networkidle + JS settle)")
    try:
        await page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
    await asyncio.sleep(random.uniform(1.0, 2.0))

    logger.info("🔥 Warm-up complete. Browser is hot and ready.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Warm-Up Primitives
# ═══════════════════════════════════════════════════════════════════════════════

async def _fake_scroll(page: Page) -> None:
    """Simulate idle browsing: scroll down, pause, scroll back up."""
    try:
        scroll_down = random.randint(200, 500)
        await page.mouse.wheel(0, scroll_down)
        await asyncio.sleep(random.uniform(0.8, 1.5))

        scroll_up = random.randint(100, scroll_down)
        await page.mouse.wheel(0, -scroll_up)
        await asyncio.sleep(random.uniform(0.5, 1.0))
    except Exception as e:
        logger.debug("Scroll simulation failed: %s", e)


async def _mouse_wiggle(page: Page, movements: int = 0) -> None:
    """Generate realistic mouse telemetry across the viewport.

    Uses Bézier-like curves with eased timing and micro-jitter
    to defeat behavioral bot detection.
    """
    num_movements = movements if movements > 0 else random.randint(3, 6)

    try:
        viewport = page.viewport_size or {"width": 1440, "height": 900}
        vw, vh = viewport["width"], viewport["height"]

        # Start from a plausible position (center-ish area)
        current_x = random.uniform(vw * 0.3, vw * 0.7)
        current_y = random.uniform(vh * 0.3, vh * 0.7)
        await page.mouse.move(current_x, current_y)

        for _ in range(num_movements):
            # Target: random position within safe viewport margins
            target_x = random.uniform(vw * 0.1, vw * 0.9)
            target_y = random.uniform(vh * 0.1, vh * 0.9)

            # Move in small steps with eased timing
            steps = random.randint(8, 18)
            for s in range(steps):
                t = s / steps
                # Ease-in-out curve
                t_eased = 0.5 - 0.5 * _cos_approx(t * 3.14159)

                ix = current_x + (target_x - current_x) * t_eased
                iy = current_y + (target_y - current_y) * t_eased

                # Micro-jitter (±2px gaussian)
                jx = random.gauss(0, 1.8)
                jy = random.gauss(0, 1.8)

                await page.mouse.move(ix + jx, iy + jy)
                await asyncio.sleep(random.uniform(0.008, 0.025))

            current_x, current_y = target_x, target_y

            # Random pause between movements (human thinking)
            await asyncio.sleep(random.uniform(0.15, 0.6))

    except Exception as e:
        logger.debug("Mouse wiggle failed: %s", e)


def _cos_approx(x: float) -> float:
    """Fast cosine approximation (Bhaskara I) — avoids importing math."""
    import math
    return math.cos(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  URL Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_base_url(url: str) -> str:
    """Extract the base domain URL from a deep link.

    Examples:
        "https://dev.to/user/post-slug/edit" → "https://dev.to/"
        "https://www.reddit.com/r/rust/submit" → "https://www.reddit.com/"
        "" → ""
    """
    if not url or not url.strip():
        return ""
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}/"
    except Exception:
        pass
    return ""


def extract_target_url_from_objective(objective: str) -> str:
    """Heuristically extract the first URL mentioned in the agent objective.

    Scans the objective text for http/https URLs and returns the first one found.
    Used by the integration layer to determine the target domain for priming.

    V17: strip trailing punctuation that natural-language sentences append to
    URLs (e.g., parentheses, periods, commas, semicolons). Previously every
    task hit ERR_NAME_NOT_RESOLVED because "https://www.amazon.in)." was
    passed as-is to Page.goto().
    """
    import re
    urls = re.findall(r'https?://[^\s\'"<>]+', objective)
    if not urls:
        return ""
    url = urls[0].rstrip('.,;:!?)]\'"')
    return url
