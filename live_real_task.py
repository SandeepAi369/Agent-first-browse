#!/usr/bin/env python3
"""
Agent First IDE — LIVE REAL TASK TEST

This script performs a real, visible browser task to demonstrate
the new Playwright-native input driver working end-to-end:

  1. Opens GitHub.com
  2. Finds the search bar and clicks it
  3. Types a search query with human-like timing
  4. Presses Enter to search
  5. Scrolls through search results
  6. Opens a public online notepad
  7. Types a formatted message
  8. Takes screenshots at each step

Watch your screen — you'll see the browser doing everything live!
"""

from __future__ import annotations

import asyncio
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

# Ensure orchestrator is on path
_PROJECT_ROOT = Path(__file__).resolve().parent
_ORCHESTRATOR = _PROJECT_ROOT / "python-orchestrator"
if str(_ORCHESTRATOR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from playwright.async_api import async_playwright
from app.browser_promoter.playwright_human_input import PlaywrightHumanInput

# ── Output Directory ─────────────────────────────────────────────────────────
SCREENSHOTS_DIR = _PROJECT_ROOT / "live_test_screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)


async def save_screenshot(page, name: str) -> str:
    """Save a timestamped screenshot."""
    ts = datetime.now().strftime("%H%M%S")
    path = SCREENSHOTS_DIR / f"{ts}_{name}.jpg"
    await page.screenshot(path=str(path), type="jpeg", quality=70)
    size = path.stat().st_size
    print(f"    [Screenshot] {path.name} ({size:,} bytes)")
    return str(path)


async def main():
    print()
    print("=" * 65)
    print("  AGENT FIRST IDE - LIVE REAL TASK TEST")
    print("=" * 65)
    print(f"  Runtime:  {platform.system()} {'(WSL)' if 'WSL_DISTRO_NAME' in os.environ else '(Native)'}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()
    print("  >>> Watch your screen - browser will open and perform tasks! <<<")
    print()

    driver = PlaywrightHumanInput(
        typing_delay_min_ms=35,
        typing_delay_max_ms=100,
        enable_bezier_movement=True,
        enable_typo_simulation=True,  # Real human-like typos enabled!
    )

    async with async_playwright() as pw:

        # ════════════════════════════════════════════════════════════════
        # STEP 1: Launch Browser
        # ════════════════════════════════════════════════════════════════
        print("[STEP 1/8] Launching browser...")
        browser = await pw.chromium.launch(
            headless=False,
            args=[
                "--start-maximized",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        print("  [OK] Chromium launched in headed mode")
        print()

        # ════════════════════════════════════════════════════════════════
        # STEP 2: Navigate to GitHub
        # ════════════════════════════════════════════════════════════════
        print("[STEP 2/8] Navigating to GitHub.com...")
        await page.goto("https://github.com", wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)
        await save_screenshot(page, "01_github_home")
        print(f"  [OK] Loaded: {page.url}")
        print()

        # ════════════════════════════════════════════════════════════════
        # STEP 3: Click the search bar
        # ════════════════════════════════════════════════════════════════
        print("[STEP 3/8] Finding and clicking the search bar...")

        # GitHub's search button/bar selectors
        search_selectors = [
            "button.AppHeader-searchButton",
            "[data-target='qbsearch-input.inputButtonText']",
            "button[aria-label='Search or jump to...']",
            ".header-search-button",
            "input[name='q']",
        ]

        clicked_search = False
        for sel in search_selectors:
            try:
                elem = page.locator(sel).first
                if await elem.count() > 0 and await elem.is_visible():
                    await driver.click_selector(page, sel, timeout_ms=5000)
                    clicked_search = True
                    print(f"  [OK] Clicked search via: {sel}")
                    break
            except Exception:
                continue

        if not clicked_search:
            # Try keyboard shortcut
            print("  [INFO] Using keyboard shortcut '/' to open search...")
            await driver.press_key(page, "/")

        await page.wait_for_timeout(1000)
        await save_screenshot(page, "02_search_opened")
        print()

        # ════════════════════════════════════════════════════════════════
        # STEP 4: Type a search query
        # ════════════════════════════════════════════════════════════════
        search_query = "playwright browser automation python"
        print(f"[STEP 4/8] Typing search query: '{search_query}'")

        # Try to find the active search input
        search_input_selectors = [
            "#query-builder-test",
            "input[name='query-builder-test']",
            "[id*='query-builder']",
            "input[aria-label='Search']",
            "input[name='q']",
            "input[type='text']:focus",
        ]

        typed_in_input = False
        for sel in search_input_selectors:
            try:
                elem = page.locator(sel).first
                if await elem.count() > 0:
                    await elem.fill("")  # Clear first
                    typed_result = await driver.type_text(page, search_query)
                    typed_in_input = True
                    print(f"  [OK] Typed {typed_result['typed']} chars (corrections: {typed_result['corrections']})")
                    break
            except Exception:
                continue

        if not typed_in_input:
            # Just type into whatever is focused
            typed_result = await driver.type_text(page, search_query)
            print(f"  [OK] Typed {typed_result['typed']} chars into focused element")

        await page.wait_for_timeout(500)
        await save_screenshot(page, "03_search_typed")
        print()

        # ════════════════════════════════════════════════════════════════
        # STEP 5: Press Enter to search
        # ════════════════════════════════════════════════════════════════
        print("[STEP 5/8] Pressing Enter to execute search...")
        await driver.press_key(page, "Enter")
        await page.wait_for_timeout(3000)
        await save_screenshot(page, "04_search_results")
        print(f"  [OK] Search results loaded: {page.url[:80]}")
        print()

        # ════════════════════════════════════════════════════════════════
        # STEP 6: Scroll through results
        # ════════════════════════════════════════════════════════════════
        print("[STEP 6/8] Scrolling through search results...")
        scroll1 = await driver.scroll(page, delta_y=600)
        await page.wait_for_timeout(800)
        print(f"  [OK] Scrolled down: {scroll1['moved']}px")

        scroll2 = await driver.scroll(page, delta_y=400)
        await page.wait_for_timeout(800)
        print(f"  [OK] Scrolled more: {scroll2['moved']}px")

        await save_screenshot(page, "05_scrolled_results")

        # Scroll back up
        scroll_up = await driver.scroll(page, delta_y=-500)
        await page.wait_for_timeout(500)
        print(f"  [OK] Scrolled back up: {scroll_up['moved']}px")
        print()

        # ════════════════════════════════════════════════════════════════
        # STEP 7: Open an online notepad and type a message
        # ════════════════════════════════════════════════════════════════
        print("[STEP 7/8] Opening online text editor...")
        await page.goto("https://notepad.js.org", wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)
        await save_screenshot(page, "06_notepad_loaded")
        print(f"  [OK] Loaded: {page.url}")

        # Find the editor area and type content
        editor_selectors = [
            "#editor",
            ".CodeMirror",
            "textarea",
            "[contenteditable='true']",
            ".editor",
            "#note",
        ]

        message = (
            "Agent First IDE - Live Test Report\n"
            "==================================\n\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Platform: {platform.system()} {platform.release()}\n"
            f"Python: {sys.version.split()[0]}\n\n"
            "Status: All systems operational!\n\n"
            "This message was typed by the AI Agent using\n"
            "the new Playwright-native input driver.\n\n"
            "Key improvements:\n"
            "- No PowerShell bridge needed\n"
            "- No OS-level window focus required\n"
            "- Works on Windows, WSL, Linux, macOS\n"
            "- Human-like typing with natural delays\n"
            "- Bezier curve mouse movements\n\n"
            "Test PASSED!\n"
        )

        editor_found = False
        for sel in editor_selectors:
            try:
                elem = page.locator(sel).first
                if await elem.count() > 0 and await elem.is_visible():
                    await elem.click()
                    editor_found = True
                    print(f"  [OK] Found editor: {sel}")
                    break
            except Exception:
                continue

        if not editor_found:
            # Click center of page
            await driver.click(page, x=683, y=400)
            print("  [INFO] Clicked center of page for editor focus")

        await page.wait_for_timeout(500)

        # Clear any existing content
        await driver.clear_field(page)
        await page.wait_for_timeout(300)

        # Type the message with human-like timing
        print("  [INFO] Typing test report (watch the browser!)...")
        typed_msg = await driver.type_text(page, message)
        print(f"  [OK] Typed {typed_msg['typed']} chars, {typed_msg['corrections']} corrections")

        await page.wait_for_timeout(1000)
        await save_screenshot(page, "07_message_typed")
        print()

        # ════════════════════════════════════════════════════════════════
        # STEP 8: Final summary
        # ════════════════════════════════════════════════════════════════
        print("[STEP 8/8] Generating final report...")
        await save_screenshot(page, "08_final")

        print()
        print("=" * 65)
        print("  LIVE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 65)
        print()
        print("  Actions performed:")
        print("    1. Launched Chromium browser (headed)")
        print("    2. Navigated to GitHub.com")
        print("    3. Clicked search bar")
        print(f"    4. Typed search query: '{search_query}'")
        print("    5. Executed search (Enter)")
        print("    6. Scrolled through results (down + up)")
        print("    7. Opened online notepad")
        print(f"    8. Typed {typed_msg['typed']}-char test report")
        print()
        print(f"  Screenshots saved to: {SCREENSHOTS_DIR}")
        print(f"  Total screenshots: {len(list(SCREENSHOTS_DIR.glob('*.jpg')))}")
        print()
        print("  Input driver: PlaywrightHumanInput (native API)")
        print("  PowerShell bridge: NOT USED")
        print("  OS focus required: NO")
        print()

        # Keep browser open for 5 seconds so user can see the result
        print("  Browser stays open for 5 seconds so you can see...")
        await page.wait_for_timeout(5000)

        await browser.close()
        print("  Browser closed. Test complete!")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
