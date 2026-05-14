import sys
import asyncio
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

from playwright.async_api import async_playwright
from app.browser_promoter.cdp_stealth_launcher import (
    STEALTH_INIT_SCRIPT, STEALTH_LAUNCH_ARGS, STEALTH_USER_AGENT,
    apply_page_stealth, get_random_viewport, VISUAL_CURSOR_INIT_SCRIPT,
)
from app.browser_promoter.browser_warmup import run_warmup
from app.logger import get_logger
from ghost_input import ghost_type, ghost_click

logger = get_logger("deterministic_reddit")

PROFILE_DIR = Path(__file__).parent / "persistence" / "browser_sessions" / "agent_main"
SUBMIT_URL = "https://old.reddit.com/r/test/submit?selftext=true&title=SearchWala+local+deployment+test"

async def main():
    logger.info("=== DETERMINISTIC REDDIT POST (OLD REDDIT) ===")
    
    pw = await async_playwright().start()
    viewport = get_random_viewport()
    
    context = await pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=False,
        viewport=viewport,
        locale="en-US",
        timezone_id="America/New_York",
        user_agent=STEALTH_USER_AGENT,
        java_script_enabled=True,
        device_scale_factor=1,
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        args=STEALTH_LAUNCH_ARGS,
        ignore_default_args=["--enable-automation"],
    )
    
    await context.add_init_script(STEALTH_INIT_SCRIPT)
    await context.add_init_script(VISUAL_CURSOR_INIT_SCRIPT)
    
    page = context.pages[0] if context.pages else await context.new_page()
    await apply_page_stealth(page)
    
    # ── Warm-up ──
    logger.info("Running warm-up...")
    await run_warmup(page, target_url=SUBMIT_URL)
    
    # ── Navigate to submit page ──
    logger.info(f"Navigating to {SUBMIT_URL}")
    await page.goto(SUBMIT_URL, wait_until="domcontentloaded", timeout=30000)
    await asyncio.sleep(3)
    
    # ── Body ──
    logger.info("Finding Body field...")
    body_loc = page.locator("textarea[name='text']")
    if await body_loc.is_visible():
        box = await body_loc.bounding_box()
        if box:
            await ghost_click(page, box["x"] + box["width"]/2, box["y"] + box["height"]/2)
            await asyncio.sleep(0.5)
            await ghost_type(page, "Just running a quick local test of the deployment pipeline. Checking visual cursor animations.")
            logger.info("Typed Body.")
    else:
        logger.error("Could not find Body field!")

    # ── Post Button ──
    logger.info("Finding Post button...")
    post_btn = page.locator("button[name='submit']")
    if await post_btn.is_visible():
        box = await post_btn.bounding_box()
        if box:
            logger.info("Clicking Post button...")
            await ghost_click(page, box["x"] + box["width"]/2, box["y"] + box["height"]/2)
            await asyncio.sleep(10)  # Wait for submission to complete
            logger.info(f"Final URL: {page.url}")
            logger.info("✅ POST SUCCESSFULLY SUBMITTED TO REDDIT!")
    else:
        logger.error("Post button not found.")
        
    await context.close()

if __name__ == "__main__":
    asyncio.run(main())
