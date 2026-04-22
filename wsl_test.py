#!/usr/bin/env python3
"""
Simple WSL test — no login required.
Searches Wikipedia and types a note in an online notepad.
"""
import asyncio, sys, os, platform
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "python-orchestrator"))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from playwright.async_api import async_playwright
from app.browser_promoter.playwright_human_input import PlaywrightHumanInput
from app.browser_promoter.browser_runtime import _detect_environment


async def main():
    env = _detect_environment()
    print(f"\n{'='*55}")
    print(f"  WSL/LINUX TEST - {env}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    driver = PlaywrightHumanInput(
        typing_delay_min_ms=35,
        typing_delay_max_ms=95,
        enable_bezier_movement=True,
        enable_typo_simulation=True,
    )

    async with async_playwright() as pw:
        # Launch Playwright's own Chromium (works on WSL/Linux without Edge)
        print("[1/6] Launching Playwright Chromium...")
        browser = await pw.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            locale="en-US",
        )
        page = await context.new_page()
        print("  OK - Browser launched\n")

        # Step 2: Go to Wikipedia
        print("[2/6] Opening Wikipedia...")
        await page.goto("https://en.wikipedia.org", wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
        print(f"  OK - {page.url}\n")

        # Step 3: Search for something
        print("[3/6] Searching for 'Artificial Intelligence'...")
        search_input = page.locator("input#searchInput")
        await search_input.click()
        await page.wait_for_timeout(300)
        typed = await driver.type_text(page, "Artificial Intelligence")
        print(f"  OK - Typed {typed['typed']} chars, {typed['corrections']} corrections")
        await driver.press_key(page, "Enter")
        await page.wait_for_timeout(3000)
        print(f"  OK - Landed on: {page.url}\n")

        # Step 4: Scroll through the article
        print("[4/6] Reading the article (scrolling slowly)...")
        for i in range(3):
            scroll = await driver.scroll(page, delta_y=400)
            await page.wait_for_timeout(800)
            print(f"  Scrolled {400*(i+1)}px...")
        print("  OK - Article browsed\n")

        # Step 5: Go to online notepad and type
        print("[5/6] Opening notepad and writing a note...")
        await page.goto("https://notepad.js.org", wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        # Find and click the editor
        for sel in ["textarea", "#editor", "[contenteditable]"]:
            try:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    await el.click()
                    break
            except Exception:
                continue

        await page.wait_for_timeout(300)
        await driver.clear_field(page)

        note = (
            f"Test from {env}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
            f"Python: {sys.version.split()[0]}\n"
            f"OS: {platform.system()} {platform.release()}\n\n"
            "This was typed by the AI agent\n"
            "using Playwright Chromium on WSL!\n"
        )
        typed = await driver.type_text(page, note)
        print(f"  OK - Typed {typed['typed']} chars\n")

        # Step 6: Done
        print("[6/6] Done! Keeping browser open for 5 seconds...")
        await page.wait_for_timeout(5000)
        await browser.close()

    print(f"\n{'='*55}")
    print("  TEST PASSED - Playwright Chromium works on WSL!")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    asyncio.run(main())
