"""Deterministic Dev.to Article Restore — No LLM Guessing.

This script does NOT use the LLM agent loop for editing. It performs
the content replacement programmatically via Playwright APIs, which is
100% reliable compared to letting an LLM figure out how to edit text.
"""

import sys, asyncio
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

from playwright.async_api import async_playwright
from app.browser_promoter.cdp_stealth_launcher import (
    STEALTH_INIT_SCRIPT, STEALTH_LAUNCH_ARGS, STEALTH_USER_AGENT,
    apply_page_stealth, get_random_viewport, VISUAL_CURSOR_INIT_SCRIPT,
)
from app.browser_promoter.browser_warmup import run_warmup
from app.logger import get_logger

logger = get_logger("restore_article")

PROFILE_DIR = Path(__file__).parent / "persistence" / "browser_sessions" / "agent_main"

# ═══════════════════════════════════════════════════════════════════════════════
#  The Full Article (Markdown)
# ═══════════════════════════════════════════════════════════════════════════════
ARTICLE_TITLE = "Why I rewrote my 90+ Engine Meta-Search in Rust 🦀"

ARTICLE_BODY = """---
title: Why I rewrote my 90+ Engine Meta-Search in Rust 🦀
published: true
tags: rust, python, searchengine, opensource
---

Just testing out my automated dev-log pipeline for **SearchWala**. Moving from Python to Rust dropped my RAM from 512 MB → 38 MB and made cold starts nearly instant. Here's the short version of why and how.

## The Problem

SearchWala aggregates results from **90+ search engines** — Google, Bing, DuckDuckGo, Brave, Mojeek, and dozens of niche/academic sources. The original Python stack (FastAPI + asyncio + BeautifulSoup) worked, but:

- **RAM hungry**: Each worker held parsed DOM trees in memory. Under load, a single instance ate ~512 MB.
- **Cold start pain**: On a fresh container, Python import chains + dependency init took 4-6 seconds.
- **GIL bottleneck**: True parallelism across 90 engines was faked with async I/O, but CPU-bound parsing still serialized.

## The Rust Rewrite

I rewrote the core in Rust using `tokio` for async, `reqwest` for HTTP, and `scraper` for HTML parsing. The results:

| Metric | Python | Rust |
|--------|--------|------|
| RAM (idle) | 512 MB | 38 MB |
| Cold start | 4.2s | 0.3s |
| P95 latency (90 engines) | 2.8s | 0.9s |
| Binary size | ~180 MB (venv) | 12 MB |

The dual-path LLM synthesis pipeline (lite mode for speed, research mode for depth) stayed as a sidecar microservice, but all search orchestration, ranking (BM25 + Reciprocal Rank Fusion), and content extraction now run natively in Rust.

## Key Takeaway

If your I/O-heavy Python service is eating memory and you need predictable latency — Rust with `tokio` is the move. Not everything needs a rewrite, but the hot path absolutely does.

Check out the full source code and drop a star on GitHub: [SearchWala on GitHub](https://github.com/SandeepAi369/SearchWala)
"""

EDIT_URL = "https://dev.to/chidari_sandeep_c8e0478a1/why-i-rewrote-my-90-engine-meta-search-in-rust-41l5/edit"


async def main():
    logger.info("=== DETERMINISTIC ARTICLE RESTORE ===")
    
    pw = await async_playwright().start()
    viewport = get_random_viewport()
    logger.info("Viewport: %dx%d", viewport["width"], viewport["height"])
    
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
    await run_warmup(page, target_url=EDIT_URL)
    
    # ── Navigate to the edit page ──
    logger.info("Navigating to edit page: %s", EDIT_URL)
    try:
        await page.goto(EDIT_URL, wait_until="networkidle", timeout=20000)
    except Exception:
        await page.goto(EDIT_URL, wait_until="domcontentloaded", timeout=15000)
    
    await asyncio.sleep(3)  # Let the editor JS initialize
    
    current_url = page.url
    logger.info("Current URL after navigation: %s", current_url)
    
    # ── Find the editor textarea ──
    # Dev.to uses a textarea with id="article_body_markdown"
    editor = await page.query_selector("#article_body_markdown")
    if not editor:
        # Fallback: try the general textarea in the editor area
        editor = await page.query_selector("textarea.crayons-textfield")
    if not editor:
        editor = await page.query_selector("textarea")
    
    if not editor:
        logger.error("COULD NOT FIND EDITOR TEXTAREA. Taking screenshot for debug.")
        await page.screenshot(path="debug_no_editor.png")
        await context.close()
        return
    
    logger.info("Found editor textarea. Clearing and replacing content...")
    
    # ── Clear + Replace (programmatic, not ghost_type) ──
    # Use JavaScript to directly set the textarea value — this is 100% reliable
    # and doesn't depend on the LLM figuring out keyboard shortcuts.
    await editor.evaluate("(el, content) => { el.value = content; el.dispatchEvent(new Event('input', {bubbles: true})); }", ARTICLE_BODY)
    
    logger.info("Article content injected (%d chars)", len(ARTICLE_BODY))
    await asyncio.sleep(2)  # Let Dev.to process the input event
    
    # ── Verify the content is in the editor ──
    current_value = await editor.evaluate("el => el.value")
    if "SandeepAi369/SearchWala" in current_value:
        logger.info("✅ Correct GitHub URL confirmed in editor!")
    else:
        logger.warning("⚠️ GitHub URL not found in editor content. Something may be wrong.")
    
    if len(current_value) < 100:
        logger.error("❌ Editor content too short (%d chars). Injection may have failed.", len(current_value))
        await page.screenshot(path="debug_short_content.png")
        await context.close()
        return
    
    logger.info("Editor content length: %d chars — looks good.", len(current_value))
    
    # ── Click Save ──
    save_btn = await page.query_selector("button:has-text('Save changes')")
    if not save_btn:
        save_btn = await page.query_selector("button:has-text('Publish')")
    if not save_btn:
        save_btn = await page.query_selector("button:has-text('Save')")
    
    if save_btn:
        logger.info("Clicking Save button...")
        await save_btn.click()
        await asyncio.sleep(5)  # Wait for save to complete
        
        final_url = page.url
        logger.info("Post-save URL: %s", final_url)
        
        # Verify the article is live
        if "/edit" not in final_url:
            logger.info("✅ ARTICLE RESTORED AND PUBLISHED SUCCESSFULLY!")
            logger.info("   Live URL: %s", final_url)
        else:
            logger.warning("Still on edit page after save. May need manual verification.")
            await page.screenshot(path="debug_after_save.png")
    else:
        logger.error("Could not find Save/Publish button. Taking screenshot.")
        await page.screenshot(path="debug_no_save_btn.png")
    
    # ── Navigate to the live post to verify ──
    live_url = "https://dev.to/chidari_sandeep_c8e0478a1/why-i-rewrote-my-90-engine-meta-search-in-rust-41l5"
    logger.info("Navigating to live post for verification: %s", live_url)
    try:
        await page.goto(live_url, wait_until="networkidle", timeout=15000)
        await asyncio.sleep(2)
        
        body_text = await page.evaluate("() => document.body.innerText")
        
        checks = {
            "Title present": "90+ Engine" in body_text or "Meta-Search" in body_text,
            "Rust mentioned": "Rust" in body_text,
            "RAM stats": "38 MB" in body_text,
            "GitHub link": "SandeepAi369/SearchWala" in body_text,
            "Not empty": len(body_text) > 500,
        }
        
        logger.info("=== VERIFICATION RESULTS ===")
        all_passed = True
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info("  %s %s", status, check_name)
            if not passed:
                all_passed = False
        
        if all_passed:
            logger.info("🎉 ALL CHECKS PASSED — Article is fully restored with correct link!")
        else:
            logger.warning("⚠️ Some checks failed. Manual review recommended.")
            await page.screenshot(path="debug_verification.png")
            
    except Exception as e:
        logger.warning("Verification navigation failed: %s", e)
    
    await context.close()
    logger.info("Done. Browser closed.")


if __name__ == "__main__":
    print("Starting Deterministic Article Restore...")
    asyncio.run(main())
