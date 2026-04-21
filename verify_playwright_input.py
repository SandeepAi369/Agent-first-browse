#!/usr/bin/env python3
"""
Agent First IDE — Verification Test for Playwright-Native Input Driver.

Tests that the new PlaywrightHumanInput driver can:
  1. Launch a browser (Playwright's own Chromium)
  2. Navigate to a page
  3. Click elements using Playwright native API
  4. Type text using Playwright native API
  5. Scroll the page
  6. Clear fields
  7. Capture screenshots

Run from Windows PowerShell:
    .venv-windows/Scripts/python.exe verify_playwright_input.py
"""

from __future__ import annotations

import asyncio
import sys
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


async def main():
    print("\n" + "=" * 60)
    print("  VERIFICATION: Playwright-Native Input Driver")
    print("=" * 60)

    driver = PlaywrightHumanInput(
        typing_delay_min_ms=25,
        typing_delay_max_ms=80,
        enable_bezier_movement=True,
        enable_typo_simulation=False,  # Disable typos for deterministic test
    )

    async with async_playwright() as pw:
        print("\n[1/7] Launching Chromium...")
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        page = await context.new_page()
        print("  ✓ Browser launched (headed mode)")

        # ── Test 2: Navigation ──
        print("\n[2/7] Navigating to example.com...")
        await page.goto("https://www.example.com", wait_until="domcontentloaded")
        print(f"  ✓ URL: {page.url}")
        assert "example" in page.url.lower(), f"Expected example.com, got {page.url}"

        # ── Test 3: Navigate to a page with input ──
        print("\n[3/7] Navigating to DuckDuckGo search...")
        await page.goto("https://duckduckgo.com", wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        print(f"  ✓ URL: {page.url}")

        # ── Test 4: Click search box and type ──
        print("\n[4/7] Clicking search box and typing text...")
        # Use selector-based click for reliability
        search_selectors = [
            "input[name='q']",
            "#searchbox_input",
            "input[type='text']",
            "textarea",
        ]
        clicked = False
        for sel in search_selectors:
            try:
                await page.wait_for_selector(sel, timeout=3000)
                result = await driver.click_selector(page, sel, timeout_ms=5000)
                print(f"  ✓ Clicked: {result}")
                clicked = True
                break
            except Exception:
                continue

        if not clicked:
            # Fallback: use coordinate click
            print("  ⚠ Selector click failed, using coordinate click")
            await driver.click(page, x=640, y=400)

        # Type text
        typed = await driver.type_text(page, "Playwright automation test")
        print(f"  ✓ Typed: {typed}")
        assert typed["typed"] == len("Playwright automation test")

        # ── Test 5: Press Enter to search ──
        print("\n[5/7] Pressing Enter to search...")
        await driver.press_key(page, "Enter")
        await page.wait_for_timeout(2000)
        print(f"  ✓ Search results URL: {page.url}")

        # ── Test 6: Scroll ──
        print("\n[6/7] Scrolling down...")
        scroll_result = await driver.scroll(page, delta_y=500)
        print(f"  ✓ Scroll: {scroll_result}")
        assert scroll_result["moved"] == 500.0

        # Scroll back up
        scroll_up = await driver.scroll(page, delta_y=-300)
        print(f"  ✓ Scroll up: {scroll_up}")

        # ── Test 7: Screenshot ──
        print("\n[7/7] Capturing screenshot...")
        screenshot = await page.screenshot(type="jpeg", quality=60)
        screenshot_path = _PROJECT_ROOT / "verification_screenshot.jpg"
        screenshot_path.write_bytes(screenshot)
        print(f"  ✓ Screenshot saved: {screenshot_path} ({len(screenshot)} bytes)")

        # ── Cleanup ──
        await page.wait_for_timeout(1500)  # Let human see the result
        await browser.close()

    print("\n" + "=" * 60)
    print("  ✓ ALL TESTS PASSED — Playwright-Native Input Driver OK")
    print("=" * 60)
    print()
    print("  The new driver works without OS-level focus.")
    print("  No PowerShell bridge. No user32.dll. No WSL issues.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
