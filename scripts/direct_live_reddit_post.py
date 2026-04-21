#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page, async_playwright

PROFILE_DIR = Path("/home/sandeep/agent first IDE/persistence/browser_sessions/reddit/reddit-main-session")
TARGET_FEED_URL = "https://www.reddit.com/r/test/hot/"
LOGIN_URL = "https://www.reddit.com/login/"
RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
POST_TEXT = f"Gemini fallback live test from Agent First IDE ({RUN_TAG})."
MAX_LOGIN_ATTEMPTS = 5
EDITOR_WAIT_TIMEOUT_SECONDS = 180
DIAGNOSTICS_DIR = Path("/home/sandeep/agent first IDE/persistence/login_diagnostics")

LOGIN_SIGNAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("wrong_password", "incorrect username or password"),
    ("wrong_password_alt", "wrong password"),
    ("invalid_credentials", "invalid username or password"),
    ("rate_limited", "you are doing that too much"),
    ("too_many_requests", "too many requests"),
    ("challenge_required", "check your email"),
    ("captcha_required", "captcha"),
    ("two_factor", "2fa"),
    ("suspicious_activity", "suspicious activity"),
)


def _wait_for_enter(prompt: str) -> None:
    print(f"[ACTION REQUIRED] {prompt}")
    input("Press ENTER to continue... ")


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "snapshot"


async def _save_diagnostics_snapshot(page: Page, label: str) -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = _safe_name(f"{ts}_{label}")
    screenshot_path = DIAGNOSTICS_DIR / f"{base}.png"
    html_path = DIAGNOSTICS_DIR / f"{base}.html"

    try:
        await page.screenshot(path=str(screenshot_path), full_page=True)
    except Exception as exc:
        print(f"[WARN] Could not save screenshot: {exc}")

    try:
        html_path.write_text(await page.content(), encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Could not save page HTML: {exc}")


async def _auth_cookie_snapshot(context: BrowserContext) -> tuple[bool, list[str]]:
    cookies = await context.cookies("https://www.reddit.com")
    names = sorted({cookie.get("name", "") for cookie in cookies if cookie.get("name")})
    is_authenticated = "reddit_session" in names
    return is_authenticated, names


async def _fetch_me_from_page(page: Page) -> dict[str, Any]:
    script = """
async () => {
  try {
    const resp = await fetch('/api/v1/me', { credentials: 'include' });
    const body = await resp.text();
    let username = null;
    try {
      const parsed = JSON.parse(body);
      username = parsed?.name ?? null;
    } catch (_) {
      username = null;
    }
    return { ok: resp.ok, status: resp.status, username, snippet: body.slice(0, 220) };
  } catch (err) {
    return { ok: false, status: -1, username: null, snippet: String(err) };
  }
}
"""
    try:
        result = await asyncio.wait_for(page.evaluate(script), timeout=12)
    except Exception as exc:
        return {"ok": False, "status": -1, "username": None, "snippet": str(exc)}
    if isinstance(result, dict):
        return result
    return {"ok": False, "status": -1, "username": None, "snippet": "invalid response"}


def _detect_login_signals(page_html: str) -> list[str]:
    lowered = re.sub(r"\s+", " ", page_html).lower()
    found: list[str] = []
    for signal_key, needle in LOGIN_SIGNAL_PATTERNS:
        if needle in lowered:
            found.append(signal_key)
    return found


async def _collect_login_diagnostics(page: Page, context: BrowserContext) -> dict[str, Any]:
    is_authenticated_cookie, cookie_names = await _auth_cookie_snapshot(context)
    me = await _fetch_me_from_page(page)

    title = ""
    try:
        title = await page.title()
    except Exception:
        title = ""

    html = ""
    try:
        html = await page.content()
    except Exception:
        html = ""

    user_menu_count = 0
    try:
        user_menu_count = await page.locator(
            "button[aria-label*='User menu' i], #USER_DROPDOWN_ID, [id*='user-menu']"
        ).count()
    except Exception:
        user_menu_count = 0

    username_field_count = 0
    password_field_count = 0
    try:
        username_field_count = await page.locator("input[name='username']").count()
        password_field_count = await page.locator("input[type='password']").count()
    except Exception:
        username_field_count = 0
        password_field_count = 0

    return {
        "url": page.url,
        "title": title,
        "cookie_authenticated": is_authenticated_cookie,
        "cookie_names": cookie_names,
        "signals": _detect_login_signals(html),
        "me_ok": bool(me.get("ok")),
        "me_status": me.get("status"),
        "me_username": str(me.get("username") or "").strip(),
        "user_menu_count": user_menu_count,
        "username_field_count": username_field_count,
        "password_field_count": password_field_count,
    }


def _is_authenticated(diag: dict[str, Any]) -> bool:
    return (
        bool(diag.get("cookie_authenticated"))
        or bool(diag.get("me_username"))
        or int(diag.get("user_menu_count") or 0) > 0
    )


def _print_login_diagnostics(diag: dict[str, Any]) -> None:
    print("[DIAG] URL:", diag.get("url"))
    print("[DIAG] Title:", diag.get("title"))
    print(
        "[DIAG] Auth flags:",
        {
            "cookie_authenticated": diag.get("cookie_authenticated"),
            "me_ok": diag.get("me_ok"),
            "me_status": diag.get("me_status"),
            "me_username": diag.get("me_username"),
            "user_menu_count": diag.get("user_menu_count"),
        },
    )
    print(
        "[DIAG] Login form fields:",
        {
            "username_fields": diag.get("username_field_count"),
            "password_fields": diag.get("password_field_count"),
        },
    )
    print("[DIAG] Signals:", diag.get("signals"))
    print("[DIAG] Cookies sample:", list(diag.get("cookie_names", []))[:10])


async def _wait_for_authenticated_session(page: Page, context: BrowserContext) -> str | None:
    print("[INFO] Login gate is strict: posting is blocked until real authentication is confirmed.")
    print("[ACTION REQUIRED] Complete Reddit login manually in the opened browser window.")
    try:
        await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=45000)
    except Exception as exc:
        print(f"[WARN] Could not navigate to login page immediately: {exc}")

    for attempt in range(1, MAX_LOGIN_ATTEMPTS + 1):
        _wait_for_enter(
            f"Attempt {attempt}/{MAX_LOGIN_ATTEMPTS}: after pressing Reddit Login and completing any email/captcha/2FA, continue"
        )
        await page.wait_for_timeout(1200)

        diag = await _collect_login_diagnostics(page, context)
        await _save_diagnostics_snapshot(page, f"login_attempt_{attempt}")

        if _is_authenticated(diag):
            username = str(diag.get("me_username") or "authenticated-user")
            print(f"[INFO] Authentication confirmed. Session user: {username}")
            return username

        _print_login_diagnostics(diag)

        signals = set(diag.get("signals", []))
        if signals & {"wrong_password", "wrong_password_alt", "invalid_credentials"}:
            print(
                "[HINT] Reddit reports invalid credentials. Double-check username/password format and account login method."
            )
        elif signals & {"challenge_required", "two_factor", "captcha_required", "suspicious_activity"}:
            print(
                "[HINT] Reddit is requesting a security challenge. Complete email/captcha/2FA in the browser first."
            )
        elif signals & {"rate_limited", "too_many_requests"}:
            print("[HINT] Reddit is rate-limiting login attempts. Wait and try again later.")
        else:
            print(
                "[HINT] Login still not confirmed. Keep the same browser profile, finish manual login, then retry."
            )

    _, cookie_names = await _auth_cookie_snapshot(context)
    print(f"[ERROR] Authentication not confirmed after {MAX_LOGIN_ATTEMPTS} attempts. Final cookies={cookie_names}")
    print(
        "[BLOCKED] Posting cannot continue. Solve login in browser first, then rerun script. "
        f"Diagnostics saved in: {DIAGNOSTICS_DIR}"
    )
    return None


async def _open_first_post(page: Page) -> str | None:
    await page.wait_for_timeout(2500)

    href = await page.evaluate(
        """
() => {
  const links = Array.from(document.querySelectorAll('a[href*="/comments/"]'));
  for (const a of links) {
    const href = a.getAttribute('href') || '';
    if (!href) continue;
    if (href.includes('/r/test/comments/')) {
      return href.startsWith('http') ? href : `https://www.reddit.com${href}`;
    }
  }
  return null;
}
"""
    )

    if not href:
        return None

    await page.goto(href, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(2200)
    return href


async def _focus_comment_editor(page: Page) -> bool:
    editor_selectors = [
        "textarea[placeholder*='thoughts']",
        "textarea[aria-label*='comment' i]",
        "textarea",
        "div[contenteditable='true'][role='textbox']",
        "div[contenteditable='true']",
    ]

    max_attempts = max(1, EDITOR_WAIT_TIMEOUT_SECONDS // 2)
    for _ in range(max_attempts):
        for selector in editor_selectors:
            locator = page.locator(selector).first
            try:
                if await locator.is_visible(timeout=900):
                    await locator.click(timeout=1200)
                    return True
            except Exception:
                continue

        for button_selector in [
            "button:has-text('Comment')",
            "button:has-text('Add a comment')",
            "button:has-text('What are your thoughts?')",
        ]:
            try:
                button = page.locator(button_selector).first
                if await button.is_visible(timeout=900):
                    await button.click(timeout=1200)
                    await page.wait_for_timeout(600)
                    break
            except Exception:
                continue

        await page.wait_for_timeout(1200)

    return False


async def _type_comment(page: Page, text: str) -> bool:
    textarea = page.locator("textarea").first
    try:
        if await textarea.is_visible(timeout=800):
            await textarea.fill(text)
            return True
    except Exception:
        pass

    try:
        await page.keyboard.type(text, delay=35)
        return True
    except Exception:
        return False


async def _submit_comment(page: Page) -> bool:
    submit_selectors = [
        "button:has-text('Comment')",
        "button:has-text('Post')",
        "button[type='submit']",
    ]

    for selector in submit_selectors:
        button = page.locator(selector).first
        try:
            if await button.is_visible(timeout=1000) and await button.is_enabled(timeout=1000):
                await button.click(timeout=1800)
                await page.wait_for_timeout(3000)
                return True
        except Exception:
            continue

    try:
        await page.keyboard.press("Control+Enter")
        await page.wait_for_timeout(2500)
        return True
    except Exception:
        pass

    return False


async def _verify_comment_posted(page: Page, text: str) -> bool:
    try:
        for _ in range(7):
            text_match = page.get_by_text(re.compile(re.escape(text), re.IGNORECASE))
            if await text_match.count() > 0:
                return True
            await page.wait_for_timeout(1500)
    except Exception:
        pass
    return False


async def _verify_comment_server_side(context: BrowserContext, post_url: str, text: str) -> bool:
    json_url = post_url.rstrip("/") + "/.json?limit=500&raw_json=1"
    candidate_urls = [json_url, post_url]

    for _ in range(8):
        for url in candidate_urls:
            try:
                response = await context.request.get(url, timeout=15000)
                body = await response.text()
                if text in body:
                    return True
            except Exception:
                continue
        await asyncio.sleep(2)

    return False


async def _manual_focus_comment_editor(page: Page) -> bool:
    _wait_for_enter(
        "Click inside the Reddit comment editor in the browser window so the cursor is active."
    )
    try:
        await page.wait_for_timeout(400)
        return True
    except Exception:
        return False


async def _manual_submit_comment(page: Page) -> bool:
    _wait_for_enter(
        "Click the Reddit Comment/Post button in the browser window now."
    )
    try:
        await page.wait_for_timeout(1500)
        return True
    except Exception:
        return False


def _manual_post_retry(payload: str) -> None:
    _wait_for_enter(
        "If needed, manually submit the comment now. "
        f"Use this exact payload: {payload}"
    )


async def main() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="UTC",
            args=[
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )

        try:
            page = context.pages[0] if context.pages else await context.new_page()

            username = await _wait_for_authenticated_session(page, context)
            if not username:
                print("[ERROR] Authentication not confirmed. Aborting without posting.")
                return

            print(f"[INFO] Proceeding with authenticated session: {username}")
            print("[INFO] Opening Reddit test feed...")
            await page.goto(TARGET_FEED_URL, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(1800)

            post_url = await _open_first_post(page)
            if not post_url:
                print("[ERROR] Could not auto-open a post in r/test.")
                return

            print("[INFO] Target URL:", page.url)
            print("[INFO] Comment payload:", POST_TEXT)

            if not await _focus_comment_editor(page):
                print("[WARN] Could not focus comment editor automatically; falling back to manual focus.")
                if not await _manual_focus_comment_editor(page):
                    print("[ERROR] Manual editor focus failed.")
                    return

            if not await _type_comment(page, POST_TEXT):
                print("[ERROR] Failed to type comment text.")
                return

            submitted = await _submit_comment(page)
            if not submitted:
                print("[WARN] Could not submit automatically; falling back to manual submit.")
                if not await _manual_submit_comment(page):
                    print("[ERROR] Manual submit fallback failed.")
                    return

            await page.wait_for_timeout(2600)
            verified = await _verify_comment_posted(page, POST_TEXT)
            if not verified:
                verified = await _verify_comment_server_side(context, page.url, POST_TEXT)

            if not verified:
                print("[WARN] Initial verification failed; requesting manual submit confirmation step.")
                _manual_post_retry(POST_TEXT)
                try:
                    await page.reload(wait_until="domcontentloaded", timeout=30000)
                except Exception:
                    pass
                await page.wait_for_timeout(1800)
                verified = await _verify_comment_posted(page, POST_TEXT)
                if not verified:
                    verified = await _verify_comment_server_side(context, page.url, POST_TEXT)

            if verified:
                print("[SUCCESS] Comment appears on page. Live post flow completed.")
            else:
                print("[ERROR] Comment submission could not be verified. Treating as failed run.")
        finally:
            await context.close()


if __name__ == "__main__":
    asyncio.run(main())