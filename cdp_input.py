"""CDP-Native Input Engine — Multi-strategy text injection for modern web frameworks.

Provides trusted input methods that bypass React SyntheticEvent filtering,
Lexical editor guards, and contenteditable validation layers.

Typing Strategy Waterfall:
  1. CDP Input.insertText (fastest, trusted, works on most sites)
  2. Playwright keyboard.type with human delays (per-character events)
  3. CDP Input.dispatchKeyEvent per character (lowest level)

Each strategy includes post-type verification to confirm text was accepted.
"""

from __future__ import annotations

import asyncio
import random
from typing import Optional

from playwright.async_api import Page

from app.logger import get_logger

logger = get_logger("cdp_input")


# ═══════════════════════════════════════════════════════════════════════════════
#  CDP Session Helper
# ═══════════════════════════════════════════════════════════════════════════════

async def _get_cdp_session(page: Page):
    """Create a CDP session from a Playwright page."""
    try:
        return await page.context.new_cdp_session(page)
    except Exception as e:
        logger.warning("CDP session creation failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Post-Type Verification
# ═══════════════════════════════════════════════════════════════════════════════

async def _verify_typed_text(page: Page, expected_text: str, timeout: float = 3.0) -> dict:
    """Verify that the active field contains the expected text.

    Returns dict with keys:
      - verified: bool
      - actual_length: int
      - actual_preview: str (first 100 chars)
      - match_ratio: float (0.0 to 1.0)
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        () => {
            // Strategy 1: Check focused element
            const active = document.activeElement;
            if (active) {
                // For input/textarea
                if (active.value !== undefined && active.value.length > 0) {
                    return { found: true, text: active.value };
                }
                // For contenteditable / Lexical / rich text editors
                if (active.isContentEditable || active.contentEditable === 'true') {
                    const text = active.innerText || active.textContent || '';
                    return { found: true, text: text };
                }
            }

            // Strategy 2: Check all focused-like elements
            const candidates = document.querySelectorAll(
                'input:focus, textarea:focus, [contenteditable="true"]:focus, '
                + '[contenteditable="true"][data-lexical-editor], '
                + '[role="textbox"]'
            );
            for (const el of candidates) {
                const text = el.value || el.innerText || el.textContent || '';
                if (text.length > 0) {
                    return { found: true, text: text };
                }
            }

            // Strategy 3: Broadest sweep — any visible input with content
            const allInputs = document.querySelectorAll('input[type="text"], textarea');
            for (const el of allInputs) {
                if (el.value && el.value.length > 0) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        return { found: true, text: el.value };
                    }
                }
            }

            return { found: false, text: '' };
        }
        """), timeout=timeout)

        if not result.get("found"):
            return {"verified": False, "actual_length": 0, "actual_preview": "", "match_ratio": 0.0}

        actual_text = result.get("text", "")
        actual_len = len(actual_text)
        expected_len = len(expected_text)

        # Calculate match ratio
        if expected_len == 0:
            match_ratio = 1.0 if actual_len == 0 else 0.0
        else:
            # Check if the expected text is contained in or matches the actual text
            if expected_text in actual_text or actual_text in expected_text:
                match_ratio = min(actual_len, expected_len) / max(actual_len, expected_len)
            else:
                # Character-level comparison
                matching_chars = sum(1 for a, b in zip(actual_text, expected_text) if a == b)
                match_ratio = matching_chars / expected_len

        return {
            "verified": match_ratio >= 0.85,  # 85% match threshold
            "actual_length": actual_len,
            "actual_preview": actual_text[:100],
            "match_ratio": match_ratio,
        }

    except Exception as e:
        logger.warning("Type verification failed: %s", e)
        return {"verified": False, "actual_length": 0, "actual_preview": "", "match_ratio": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 1: CDP Input.insertText (Fastest, Trusted)
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_cdp_insert_text(page: Page, text: str) -> bool:
    """Use CDP Input.insertText — generates trusted InputEvent accepted by most frameworks."""
    cdp = await _get_cdp_session(page)
    if not cdp:
        return False

    try:
        await cdp.send("Input.insertText", {"text": text})
        await asyncio.sleep(0.3)
        logger.debug("CDP insertText: injected %d chars", len(text))
        return True
    except Exception as e:
        logger.warning("CDP insertText failed: %s", e)
        return False
    finally:
        try:
            await cdp.detach()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 2: Playwright keyboard.type with human-like delays
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_playwright_type(page: Page, text: str) -> bool:
    """Use Playwright's keyboard.type() with variable per-character delays.

    This generates individual keydown/keyup events that React's
    onChange handlers can process one at a time.
    """
    try:
        # Adaptive delay based on text length
        if len(text) > 200:
            delay_ms = 8  # Fast for long text
        elif len(text) > 50:
            delay_ms = 20
        else:
            delay_ms = 40  # More human-like for short text

        await page.keyboard.type(text, delay=delay_ms)
        await asyncio.sleep(0.3)
        logger.debug("Playwright type: typed %d chars (delay=%dms)", len(text), delay_ms)
        return True
    except Exception as e:
        logger.warning("Playwright type failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3: CDP per-character key events (Lowest level, most compatible)
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_cdp_key_events(page: Page, text: str) -> bool:
    """Dispatch individual CDP key events per character with realistic timing.

    This is the most compatible method — generates keyDown, char, keyUp
    events that are indistinguishable from real keyboard input.
    """
    cdp = await _get_cdp_session(page)
    if not cdp:
        return False

    try:
        for i, char in enumerate(text):
            if char == '\n':
                # Enter key
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyDown", "key": "Enter", "code": "Enter",
                    "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13,
                })
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyUp", "key": "Enter", "code": "Enter",
                    "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13,
                })
            elif char == '\t':
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyDown", "key": "Tab", "code": "Tab",
                    "windowsVirtualKeyCode": 9, "nativeVirtualKeyCode": 9,
                })
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyUp", "key": "Tab", "code": "Tab",
                    "windowsVirtualKeyCode": 9, "nativeVirtualKeyCode": 9,
                })
            else:
                key_code = ord(char.upper()) if char.isalpha() else ord(char)
                code_name = f"Key{char.upper()}" if char.isalpha() else ""

                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyDown", "key": char, "code": code_name,
                    "text": char,
                    "windowsVirtualKeyCode": key_code,
                    "nativeVirtualKeyCode": key_code,
                    "autoRepeat": False,
                })
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "char", "text": char,
                    "key": char, "code": code_name,
                    "windowsVirtualKeyCode": key_code,
                    "nativeVirtualKeyCode": key_code,
                })
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyUp", "key": char, "code": code_name,
                    "windowsVirtualKeyCode": key_code,
                    "nativeVirtualKeyCode": key_code,
                })

            # Human-like variable delay between characters
            delay = random.gauss(0.055, 0.020)
            delay = max(0.015, min(0.120, delay))
            await asyncio.sleep(delay)

            # Periodic abort check
            if i > 0 and i % 50 == 0:
                try:
                    if page.is_closed():
                        logger.warning("CDP key events aborted: page closed at char %d/%d", i, len(text))
                        return False
                except Exception:
                    return False

        logger.debug("CDP key events: dispatched %d chars", len(text))
        return True
    except Exception as e:
        logger.warning("CDP key events failed: %s", e)
        return False
    finally:
        try:
            await cdp.detach()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Field Clearing
# ═══════════════════════════════════════════════════════════════════════════════

async def clear_field(page: Page) -> None:
    """Clear the currently focused input field using multiple strategies."""
    try:
        # Strategy 1: Select all + delete
        await page.keyboard.press("Control+A")
        await asyncio.sleep(0.1)
        await page.keyboard.press("Delete")
        await asyncio.sleep(0.1)

        # Strategy 2: Verify it's actually empty, if not try Backspace
        result = await _verify_typed_text(page, "", timeout=1.0)
        if result.get("actual_length", 0) > 0:
            await page.keyboard.press("Control+A")
            await asyncio.sleep(0.05)
            await page.keyboard.press("Backspace")
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.warning("Field clear failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point: Resilient Type with Waterfall
# ═══════════════════════════════════════════════════════════════════════════════

async def _delayed_recheck(
    page: Page,
    expected_text: str,
    delay_ms: int = 300,
) -> bool:
    """Wait delay_ms, then recheck if the active field still has the expected text.

    This catches React/Vue controlled component reversions where:
      1. CDP insertText sets the DOM value immediately
      2. Our verification reads it back → PASS
      3. React's setState reconciler overwrites it 100-300ms later → FAIL

    Returns True if text is still intact, False if it was reverted.
    """
    await asyncio.sleep(delay_ms / 1000.0)
    recheck = await _verify_typed_text(page, expected_text, timeout=2.0)
    if recheck["verified"]:
        return True
    else:
        logger.warning(
            "🔄 REACT REVERSION DETECTED: field was overwritten after %dms "
            "(now: %d chars, %.0f%% match: '%s')",
            delay_ms,
            recheck["actual_length"],
            recheck["match_ratio"] * 100,
            recheck["actual_preview"][:60],
        )
        return False


async def resilient_type(
    page: Page,
    text: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    clear_first: bool = True,
    max_retries: int = 3,
) -> dict:
    """Type text into the focused field using a multi-strategy waterfall.

    If x, y are provided, clicks to focus the field first.
    Tries increasingly aggressive strategies until text is verified.

    v2.0 UPGRADE: After initial verification, performs a delayed recheck
    (300ms) to catch React/Vue state reversions. If reverted, falls through
    to the next strategy (cdp_key_events fires individual keystrokes that
    React's onChange handler processes correctly).

    Returns dict:
      - success: bool
      - strategy: str (which strategy worked)
      - verified: bool
      - actual_length: int
      - attempts: int
      - react_stable: bool (True if text survived delayed recheck)
    """
    from ghost_input import ghost_click  # Avoid circular import

    strategies = [
        ("cdp_insertText", _strategy_cdp_insert_text),
        ("playwright_type", _strategy_playwright_type),
        ("cdp_key_events", _strategy_cdp_key_events),
    ]

    for attempt in range(max_retries):
        strategy_name, strategy_fn = strategies[min(attempt, len(strategies) - 1)]

        logger.info(
            "Typing attempt %d/%d via %s (%d chars)",
            attempt + 1, max_retries, strategy_name, len(text),
        )

        # Focus the field if coordinates provided
        if x is not None and y is not None:
            try:
                await ghost_click(page, x, y)
                await asyncio.sleep(0.3)
                # V15.0 F2: Collapse selection to end of field value (W3C setSelectionRange)
                # React/Angular onFocus handlers often call select() which highlights all text.
                # setSelectionRange(len, len) deselects and places cursor at end, preventing
                # subsequent insertText from overwriting existing content.
                await page.evaluate("""() => {
                    const el = document.activeElement;
                    if (el && typeof el.setSelectionRange === 'function') {
                        try {
                            const len = (el.value || '').length;
                            el.setSelectionRange(len, len);
                        } catch(_) {}
                    } else if (el && el.isContentEditable) {
                        try {
                            const sel = window.getSelection();
                            if (sel && el.lastChild) {
                                sel.collapse(el.lastChild, el.lastChild.length || 0);
                            }
                        } catch(_) {}
                    }
                }""")
            except Exception as e:
                logger.warning("Focus click failed: %s", e)

        # V15.1 Patch B: Smart clear detection — auto-detect append vs replace
        if clear_first:
            try:
                existing_val = await page.evaluate("""() => {
                    const el = document.activeElement;
                    if (!el) return '';
                    return (el.value || el.textContent || '').trim();
                }""")
                if existing_val:
                    if text.startswith(existing_val):
                        # Agent text is a superset — just append the delta
                        delta = text[len(existing_val):]
                        if delta:
                            logger.info(
                                "Smart clear: existing '%s...' is prefix of target, appending delta '%s...' (%d chars)",
                                existing_val[:30], delta[:30], len(delta),
                            )
                            text = delta
                        else:
                            logger.info("Smart clear: field already contains exact target text — skipping type")
                            return {
                                "success": True, "strategy": "smart_skip",
                                "verified": True, "actual_length": len(existing_val),
                                "attempts": attempt + 1, "react_stable": True,
                            }
                        clear_first = False
                    elif existing_val.endswith(text):
                        # Target text is already at the end — skip
                        logger.info("Smart clear: target text already at end of field — skipping type")
                        return {
                            "success": True, "strategy": "smart_skip",
                            "verified": True, "actual_length": len(existing_val),
                            "attempts": attempt + 1, "react_stable": True,
                        }
            except Exception as e:
                logger.debug("Smart clear detection failed: %s — falling back to normal clear", e)

        # Clear existing content
        if clear_first:
            await clear_field(page)

        # Execute the typing strategy
        ok = await strategy_fn(page, text)
        if not ok:
            logger.warning("%s execution failed — trying next strategy", strategy_name)
            continue

        # ── Phase 1: Immediate verification ──
        await asyncio.sleep(0.3)
        verification = await _verify_typed_text(page, text)

        if not verification["verified"]:
            # Text wasn't accepted at all — try next strategy
            logger.warning(
                "⚠️ %s: text not fully accepted (got %d chars, %.0f%% match: '%s')",
                strategy_name,
                verification["actual_length"],
                verification["match_ratio"] * 100,
                verification["actual_preview"][:60],
            )
            continue

        # ── Phase 2: Delayed React reversion check (v2.0) ──
        # Only for non-keystroke strategies (insertText, playwright_type)
        # CDP key events fire onChange per character, so React can't revert
        react_stable = True
        if strategy_name != "cdp_key_events":
            react_stable = await _delayed_recheck(page, text, delay_ms=300)
            if not react_stable:
                logger.warning(
                    "⚠️ %s passed immediate verify but React reverted text — "
                    "escalating to cdp_key_events",
                    strategy_name,
                )
                # Force next attempt to use cdp_key_events
                # by skipping to the last strategy index
                if attempt < max_retries - 1:
                    strategies[attempt + 1] = ("cdp_key_events", _strategy_cdp_key_events)
                continue

        logger.info(
            "✅ TYPE SUCCESS via %s: %d chars verified (%.0f%% match)%s",
            strategy_name,
            verification["actual_length"],
            verification["match_ratio"] * 100,
            " [React-stable]" if react_stable else "",
        )
        return {
            "success": True,
            "strategy": strategy_name,
            "verified": True,
            "actual_length": verification["actual_length"],
            "attempts": attempt + 1,
            "react_stable": react_stable,
        }

    # All strategies exhausted
    logger.error(
        "❌ TYPE FAILED: all %d strategies exhausted for %d-char text",
        max_retries, len(text),
    )
    return {
        "success": False,
        "strategy": "none",
        "verified": False,
        "actual_length": 0,
        "attempts": max_retries,
        "react_stable": False,
    }
