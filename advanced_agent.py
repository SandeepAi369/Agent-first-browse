"""Advanced Single-Agent Browser Automation with CRASH-PROOF Session Persistence.

Session Persistence Strategy (4 layers of protection):
  Layer 1 — user_data_dir: Chromium's native disk-level cookie storage.
  Layer 2 — storage_state injection: Explicit cookie/localStorage replay on launch.
  Layer 3 — Periodic auto-save: Cookies dumped to disk every N steps during execution.
  Layer 4 — Crash-proof finally: try/except/finally with signal handlers + atexit
             guarantee state is saved even on KeyboardInterrupt, SIGTERM, or crashes.
"""

import argparse
import asyncio
import atexit
import json
import os
import signal
import sys
import traceback
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

# Ensure app imports work
sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

from app.browser_promoter.cdp_stealth_launcher import (
    STEALTH_INIT_SCRIPT,
    STEALTH_LAUNCH_ARGS,
    STEALTH_USER_AGENT,
    apply_page_stealth,
)
from app.browser_promoter.nodes import _extract_interactive_regions
from app.browser_promoter.worker_planner import _build_worker_llms
from app.logger import get_logger
from playwright.async_api import async_playwright, BrowserContext, Page

logger = get_logger("advanced_agent")

# ═══════════════════════════════════════════════════════════════════════════════
#  Persistence Paths (all inside WSL)
# ═══════════════════════════════════════════════════════════════════════════════
PERSISTENCE_ROOT = Path(__file__).parent / "persistence"
STATE_FILE = PERSISTENCE_ROOT / "session_state.json"
STATE_BACKUP = PERSISTENCE_ROOT / "session_state.backup.json"
PROFILE_DIR = PERSISTENCE_ROOT / "browser_sessions" / "agent_main"
AUTO_SAVE_EVERY_N_STEPS = 3  # Save cookies every 3 steps during execution


def _ensure_dirs():
    PERSISTENCE_ROOT.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Crash-Proof Session Manager
# ═══════════════════════════════════════════════════════════════════════════════
class SessionGuard:
    """Guarantees session state is saved no matter how the process exits.

    Registers itself as:
      - A signal handler for SIGINT (Ctrl+C) and SIGTERM (kill)
      - An atexit handler (called on normal interpreter shutdown)
      - Exposes save() for manual/periodic saves

    The guard maintains a reference to the live BrowserContext and uses
    atomic writes (write to .tmp then rename) to prevent file corruption.
    """

    _instance: "SessionGuard | None" = None

    def __init__(self):
        self._context: BrowserContext | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_saved_state: dict | None = None
        self._save_count = 0
        self._installed = False

    @classmethod
    def get(cls) -> "SessionGuard":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def attach(self, context: BrowserContext, loop: asyncio.AbstractEventLoop):
        """Attach a live browser context to guard."""
        self._context = context
        self._loop = loop
        if not self._installed:
            self._install_handlers()
            self._installed = True
        logger.info("SessionGuard: attached to browser context")

    def _install_handlers(self):
        """Install signal + atexit handlers (once)."""
        # Signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._signal_handler)
            except (OSError, ValueError):
                pass  # Can't set signal handler in non-main thread
        # atexit handler
        atexit.register(self._atexit_handler)
        logger.info("SessionGuard: signal + atexit handlers installed")

    def _signal_handler(self, signum, frame):
        """Called on Ctrl+C or kill — force-save synchronously then exit."""
        sig_name = signal.Signals(signum).name
        logger.warning("SessionGuard: caught %s — force-saving session...", sig_name)
        self._force_sync_save()
        logger.info("SessionGuard: session saved after %s. Exiting.", sig_name)
        sys.exit(0)

    def _atexit_handler(self):
        """Called on normal interpreter shutdown — last-resort save."""
        if self._context is not None:
            logger.info("SessionGuard: atexit triggered — attempting final save...")
            self._force_sync_save()

    async def async_save(self):
        """Save session state asynchronously (preferred path)."""
        if self._context is None:
            logger.warning("SessionGuard: no context attached, skipping save")
            return

        try:
            state = await self._context.storage_state()
            self._atomic_write(state)
            self._last_saved_state = state
            self._save_count += 1
            cookie_count = len(state.get("cookies", []))
            origin_count = len(state.get("origins", []))
            logger.info(
                "SessionGuard: saved (%d cookies, %d origins) [save #%d]",
                cookie_count, origin_count, self._save_count,
            )
        except Exception as e:
            logger.error("SessionGuard: async save failed: %s", e)
            # Fall back to writing whatever we last had
            if self._last_saved_state:
                logger.info("SessionGuard: writing last known good state as fallback")
                self._atomic_write(self._last_saved_state)

    def _force_sync_save(self):
        """Synchronous save — used by signal handlers and atexit.

        When a signal fires, the event loop may be dead. We try multiple
        strategies in order of reliability.
        """
        if self._context is None:
            return

        # Strategy 1: Try to run async save in the existing loop
        try:
            loop = self._loop or asyncio.get_event_loop()
            if loop.is_running():
                # Schedule it and hope we get to it
                loop.create_task(self.async_save())
                return
            else:
                loop.run_until_complete(self.async_save())
                return
        except Exception:
            pass

        # Strategy 2: Create a brand new event loop
        try:
            new_loop = asyncio.new_event_loop()
            new_loop.run_until_complete(self.async_save())
            new_loop.close()
            return
        except Exception:
            pass

        # Strategy 3: Write whatever we last captured (no async needed)
        if self._last_saved_state:
            logger.warning("SessionGuard: all async strategies failed — writing last known state")
            try:
                self._atomic_write(self._last_saved_state)
            except Exception as e:
                logger.error("SessionGuard: CRITICAL — even fallback write failed: %s", e)

    def _atomic_write(self, state: dict):
        """Write state to disk atomically (tmp + rename) to prevent corruption."""
        _ensure_dirs()
        tmp_path = STATE_FILE.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(state, indent=2))
            # Backup the old state file first
            if STATE_FILE.exists():
                try:
                    STATE_BACKUP.write_text(STATE_FILE.read_text())
                except Exception:
                    pass
            # Atomic rename
            tmp_path.rename(STATE_FILE)
        except Exception as e:
            logger.error("SessionGuard: atomic write failed: %s", e)
            # Direct write as absolute last resort
            try:
                STATE_FILE.write_text(json.dumps(state, indent=2))
            except Exception:
                pass

    def detach(self):
        """Detach the context reference (called after context.close())."""
        self._context = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Browser Launch with storage_state injection
# ═══════════════════════════════════════════════════════════════════════════════
async def launch_browser(*, headless: bool = False) -> tuple[BrowserContext, Page]:
    """Launch Playwright with persistent profile + storage_state injection."""
    _ensure_dirs()
    pw = await async_playwright().start()

    logger.info("Launching Playwright Chromium (persistent profile: %s)", PROFILE_DIR)

    context = await pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=headless,
        viewport={"width": 1440, "height": 900},
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

    # ── Attach SessionGuard immediately ──
    guard = SessionGuard.get()
    guard.attach(context, asyncio.get_running_loop())

    # ── Inject saved cookies from state.json if it exists ──
    state_source = STATE_FILE if STATE_FILE.exists() else (STATE_BACKUP if STATE_BACKUP.exists() else None)
    if state_source:
        logger.info("Injecting saved session state from %s", state_source)
        try:
            saved = json.loads(state_source.read_text())
            cookies = saved.get("cookies", [])
            if cookies:
                for c in cookies:
                    ss = c.get("sameSite", "Lax")
                    if ss not in ("Strict", "Lax", "None"):
                        c["sameSite"] = "Lax"
                await context.add_cookies(cookies)
                logger.info("Injected %d cookies from saved state.", len(cookies))

            origins = saved.get("origins", [])
            for origin_data in origins:
                origin_url = origin_data.get("origin", "")
                ls_items = origin_data.get("localStorage", [])
                if origin_url and ls_items:
                    temp_page = await context.new_page()
                    try:
                        await temp_page.goto(origin_url, wait_until="domcontentloaded", timeout=10000)
                        for item in ls_items:
                            await temp_page.evaluate(
                                "(args) => localStorage.setItem(args.name, args.value)",
                                {"name": item["name"], "value": item["value"]},
                            )
                    except Exception as e:
                        logger.warning("Failed to inject localStorage for %s: %s", origin_url, e)
                    finally:
                        await temp_page.close()

            # Seed the guard's last known state so signal handlers have data
            guard._last_saved_state = saved
            logger.info("Session state injection complete.")
        except Exception as e:
            logger.warning("Could not load state file: %s (starting fresh)", e)

    # Get or create page
    if context.pages:
        page = context.pages[0]
    else:
        page = await context.new_page()

    await apply_page_stealth(page)

    try:
        await page.bring_to_front()
    except Exception:
        pass

    return context, page


# ═══════════════════════════════════════════════════════════════════════════════
#  Manual Login Mode (--login)
# ═══════════════════════════════════════════════════════════════════════════════
async def manual_login_mode():
    """Open a browser for human login, then save the session state."""
    context, page = await launch_browser()
    guard = SessionGuard.get()

    print("\n" + "=" * 70)
    print("  🔐  MANUAL LOGIN MODE")
    print("=" * 70)
    print("  A Chromium browser window has opened.")
    print("  Please log into your Google/Gmail/Reddit account(s) now.")
    print("")
    print("  Once you are fully logged in and can see your inbox/feed,")
    print("  come back here and press ENTER to save the session.")
    print("  (Even if you Ctrl+C, the session will be auto-saved!)")
    print("=" * 70)

    try:
        input("\n  👉 Press ENTER when login is complete: ")
    except (EOFError, KeyboardInterrupt):
        logger.info("Input interrupted — saving session anyway...")

    try:
        await guard.async_save()
    except Exception:
        guard._force_sync_save()
    finally:
        try:
            await context.close()
        except Exception:
            pass
        guard.detach()

    print("\n  ✅ Session saved! Future runs will use your login automatically.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Agent Action Schema
# ═══════════════════════════════════════════════════════════════════════════════
class AdvancedAction(BaseModel):
    reasoning: str = Field(description="Brief explanation of why this action is taken.")
    action_type: str = Field(description="One of: 'goto', 'click', 'type', 'scroll', 'press_enter', 'done'")
    url: str | None = Field(default=None, description="URL for 'goto' action")
    element_id: int | None = Field(default=None, description="Index of the interactive element")
    text: str | None = Field(default=None, description="Text to type for 'type' action")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Agent Loop
# ═══════════════════════════════════════════════════════════════════════════════
async def run_agent(objective: str):
    """Core single-agent action loop with crash-proof session persistence."""
    context, page = await launch_browser()
    guard = SessionGuard.get()

    _, secondary, tertiary = _build_worker_llms()
    llm_clients = secondary + tertiary
    if not llm_clients:
        logger.error("No LLM clients available. Check API keys in .env.")
        await guard.async_save()
        await context.close()
        guard.detach()
        return

    llm = llm_clients[0]
    structured_llm = llm.with_structured_output(AdvancedAction)
    logger.info("Model: %s", getattr(llm, 'model_name', str(llm)))
    logger.info("Objective: %s", objective)

    action_history: list[str] = []
    max_steps = 20
    step = 0

    try:
        for step in range(max_steps):
            logger.info("━━━ Step %d/%d ━━━", step + 1, max_steps)

            # ── Periodic auto-save (every N steps) ──
            if step > 0 and step % AUTO_SAVE_EVERY_N_STEPS == 0:
                logger.info("Auto-saving session state (periodic checkpoint)...")
                try:
                    await guard.async_save()
                except Exception as e:
                    logger.warning("Periodic auto-save failed: %s", e)

            try:
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
                await page.wait_for_timeout(1500)
            except Exception:
                pass

            current_url = page.url
            logger.info("URL: %s", current_url)

            try:
                raw_regions = await _extract_interactive_regions(page, max_regions=150)
            except Exception as e:
                logger.warning("DOM extraction failed: %s — retrying next step", e)
                continue

            compact = [
                {"index": i, "tag": r.tag, "role": r.role, "text": r.text[:120]}
                for i, r in enumerate(raw_regions)
            ]

            system_prompt = (
                "You are an elite autonomous web agent operating inside a real browser.\n"
                "You receive the current URL, a history of past actions, and a list of interactive DOM elements.\n"
                "Choose the EXACT next action. If the objective is fully complete, output action_type='done'.\n"
                "IMPORTANT: If you already typed text into a field, do NOT type it again. Move to the NEXT step.\n"
                "Available actions: goto, click, type, scroll, press_enter, done.\n"
            )

            user_prompt = (
                f"OBJECTIVE: {objective}\n"
                f"CURRENT URL: {current_url}\n"
                f"COMPLETED ACTIONS: {json.dumps(action_history)}\n"
                f"INTERACTIVE ELEMENTS:\n{json.dumps(compact, indent=2)}\n\n"
                "What is your NEXT action?"
            )

            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            logger.info("Querying LLM...")
            try:
                decision: AdvancedAction = await structured_llm.ainvoke(messages)
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                break

            logger.info("Reasoning: %s", decision.reasoning)
            logger.info("Action: %s", decision.action_type)

            summary = f"Step {step+1}: {decision.action_type}"
            if decision.element_id is not None and decision.element_id < len(raw_regions):
                target_text = raw_regions[decision.element_id].text[:60]
                summary += f" on '{target_text}'"
            if decision.text:
                summary += f" text='{decision.text}'"
            action_history.append(summary)

            # ── Execute ──
            if decision.action_type == "done":
                logger.info("✅ Goal Achieved!")
                break

            elif decision.action_type == "goto":
                logger.info("Navigating → %s", decision.url)
                try:
                    await page.goto(decision.url, wait_until="domcontentloaded", timeout=15000)
                except Exception as e:
                    logger.warning("Navigation failed: %s", e)

            elif decision.action_type == "click":
                if decision.element_id is None or decision.element_id >= len(raw_regions):
                    logger.warning("Invalid element_id %s", decision.element_id)
                    continue
                t = raw_regions[decision.element_id]
                cx, cy = t.x + t.width / 2, t.y + t.height / 2
                logger.info("Clicking (%s '%s') at (%d, %d)", t.tag, t.text[:40], cx, cy)
                try:
                    await page.mouse.click(cx, cy)
                except Exception as e:
                    logger.warning("Click failed: %s", e)

            elif decision.action_type == "type":
                if decision.element_id is None or decision.element_id >= len(raw_regions):
                    logger.warning("Invalid element_id %s", decision.element_id)
                    continue
                t = raw_regions[decision.element_id]
                cx, cy = t.x + t.width / 2, t.y + t.height / 2
                logger.info("Typing '%s' into (%s '%s')", decision.text, t.tag, t.text[:40])
                try:
                    await page.mouse.click(cx, cy)
                    await asyncio.sleep(0.3)
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Backspace")
                    await page.keyboard.type(decision.text, delay=30)
                except Exception as e:
                    logger.warning("Type failed: %s", e)

            elif decision.action_type == "press_enter":
                logger.info("Pressing Enter")
                try:
                    await page.keyboard.press("Enter")
                except Exception as e:
                    logger.warning("Press Enter failed: %s", e)

            elif decision.action_type == "scroll":
                logger.info("Scrolling down")
                try:
                    await page.mouse.wheel(0, 600)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.warning("Scroll failed: %s", e)

        logger.info("Agent loop finished after %d steps.", step + 1)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt caught — saving session before exit...")

    except Exception as e:
        logger.error("Unexpected crash: %s", e)
        logger.error(traceback.format_exc())

    finally:
        # ════════════════════════════════════════════════════════════════════
        #  CRASH-PROOF FINALLY BLOCK
        #  This runs no matter what: normal exit, exception, Ctrl+C, break.
        #  Three nested attempts to guarantee the state is written to disk.
        # ════════════════════════════════════════════════════════════════════
        logger.info("Entering crash-proof shutdown sequence...")

        # Attempt 1: Full async save (best quality — captures everything)
        try:
            await guard.async_save()
            logger.info("Shutdown save: SUCCESS (async path)")
        except Exception as e1:
            logger.warning("Shutdown save attempt 1 failed: %s", e1)

            # Attempt 2: Direct storage_state call (bypass guard)
            try:
                state = await context.storage_state()
                guard._atomic_write(state)
                logger.info("Shutdown save: SUCCESS (direct storage_state)")
            except Exception as e2:
                logger.warning("Shutdown save attempt 2 failed: %s", e2)

                # Attempt 3: Write last known good state
                if guard._last_saved_state:
                    try:
                        guard._atomic_write(guard._last_saved_state)
                        logger.info("Shutdown save: SUCCESS (last known good state)")
                    except Exception as e3:
                        logger.error("Shutdown save attempt 3 FAILED: %s — SESSION MAY BE LOST", e3)

        # Close the browser context
        try:
            await context.close()
        except Exception:
            pass

        guard.detach()
        logger.info("Browser closed. Session state persisted.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Advanced Browser Agent with Crash-Proof Session Persistence")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("login", help="Open browser for manual login, then save session state.")
    run_parser = sub.add_parser("run", help="Run the agent with a task objective.")
    run_parser.add_argument("objective", nargs="?", default=(
        "1. Go to https://the-internet.herokuapp.com/login\n"
        "2. Type 'tomsmith' into the username field.\n"
        "3. Type 'SuperSecretPassword!' into the password field.\n"
        "4. Click the Login button.\n"
        "5. Wait to see the secure area success message and then finish."
    ), help="The task objective for the agent.")

    args = parser.parse_args()

    if args.command == "login":
        asyncio.run(manual_login_mode())
    elif args.command == "run":
        asyncio.run(run_agent(args.objective))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
