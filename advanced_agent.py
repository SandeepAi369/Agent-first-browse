"""Advanced Single-Agent Browser Automation with Native State Persistence (v8.0).

Session Persistence:
  Pure native user_data_dir — Chromium manages all cookies, localStorage,
  IndexedDB, and Cache natively on disk. No synthetic JSON injection.
  Login once via --login mode, then all subsequent runs inherit the session.

Enterprise Features (v5.0):
  - Smart DOM Compression with priority tiers (P0 forms > P1 links > P2 nav)
  - GhostCursor Bézier mouse paths + human typing with typo correction
  - Adversarial Content Critic (pre-publish red-team review)
  - Episodic Campaign Memory (SQLite dedup + cooldown enforcement)
  - Multi-key API failover: Groq → DeepSeek → NVIDIA GLM → Gemma 4

Self-Healing & Anti-Detection (v6.0+):
  - Anti-Loop Watchdog: detects repeated URL/action patterns, injects override
  - CAPTCHA/Cloudflare Detector: auto-clicks challenge checkboxes via GhostCursor
  - LLM Circuit Breaker: hard ceiling on requests/failures per run
  - Progressive Perception (v7.0): ReAct Heavy Mode with Vision + DOM
"""

import argparse
import asyncio
import atexit
import json
import signal
import sys
import traceback
from collections import Counter
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
    get_random_viewport,
)
from app.browser_promoter.browser_warmup import (
    run_warmup,
    extract_target_url_from_objective,
)
from app.browser_promoter.nodes import _extract_interactive_regions
from app.browser_promoter.worker_planner import _build_worker_llms
from app.logger import get_logger
from playwright.async_api import async_playwright, BrowserContext, Page

# Enterprise modules (v5.0)
from ghost_input import ghost_click, ghost_type, ghost_scroll
from campaign_memory import CampaignMemory
from content_critic import ContentCritic

logger = get_logger("advanced_agent")





# ═══════════════════════════════════════════════════════════════════════════════
#  LLM Circuit Breaker — Prevents runaway API drain
# ═══════════════════════════════════════════════════════════════════════════════
class CircuitBreaker:
    """Track cumulative LLM usage per run and enforce hard ceilings.

    Thresholds:
      max_requests=50  — enough for 20 steps × ~2.5 LLM calls/step
      max_failures=10  — if 10 calls fail across ALL models, something is broken
    """

    def __init__(self, max_requests: int = 50, max_failures: int = 10):
        self.request_count = 0
        self.failure_count = 0
        self.max_requests = max_requests
        self.max_failures = max_failures

    def record_success(self):
        self.request_count += 1

    def record_failure(self):
        self.failure_count += 1
        self.request_count += 1

    @property
    def tripped(self) -> bool:
        return (self.request_count >= self.max_requests
                or self.failure_count >= self.max_failures)

    @property
    def reason(self) -> str:
        if self.request_count >= self.max_requests:
            return f"Request ceiling hit ({self.request_count}/{self.max_requests})"
        if self.failure_count >= self.max_failures:
            return f"Failure ceiling hit ({self.failure_count}/{self.max_failures})"
        return ""

    def status_line(self) -> str:
        return f"[CB: {self.request_count}/{self.max_requests} reqs, {self.failure_count}/{self.max_failures} fails]"

# ═══════════════════════════════════════════════════════════════════════════════
#  Persistence Paths (all inside WSL)
# ═══════════════════════════════════════════════════════════════════════════════
PERSISTENCE_ROOT = Path(__file__).parent / "persistence"
PROFILE_DIR = PERSISTENCE_ROOT / "browser_sessions" / "agent_main"


def _ensure_dirs():
    PERSISTENCE_ROOT.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Browser Lifecycle Guard
# ═══════════════════════════════════════════════════════════════════════════════
class SessionGuard:
    """Lightweight lifecycle guard for the browser context.

    With native user_data_dir persistence, Chromium handles all cookie/storage
    persistence internally. This guard only ensures the browser context is
    closed cleanly on exit (which flushes pending state to disk).
    """

    _instance: "SessionGuard | None" = None

    def __init__(self):
        self._context: BrowserContext | None = None
        self._installed = False

    @classmethod
    def get(cls) -> "SessionGuard":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def attach(self, context: BrowserContext):
        """Attach a live browser context to guard."""
        self._context = context
        if not self._installed:
            self._install_handlers()
            self._installed = True
        logger.info("SessionGuard: attached to browser context")

    def _install_handlers(self):
        """Install signal + atexit handlers (once)."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._signal_handler)
            except (OSError, ValueError):
                pass
        atexit.register(self._atexit_handler)
        logger.info("SessionGuard: signal + atexit handlers installed")

    def _signal_handler(self, signum, frame):
        """Called on Ctrl+C or kill — close context then exit."""
        sig_name = signal.Signals(signum).name
        logger.warning("SessionGuard: caught %s — closing browser...", sig_name)
        # Context.close() flushes all native state to user_data_dir
        self.detach()
        sys.exit(0)

    def _atexit_handler(self):
        """Called on normal interpreter shutdown."""
        if self._context is not None:
            logger.info("SessionGuard: atexit — detaching context")
            self.detach()

    def detach(self):
        """Detach the context reference (called after context.close())."""
        self._context = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Browser Launch — Pure Native Persistence (v8.0)
# ═══════════════════════════════════════════════════════════════════════════════
async def launch_browser(*, headless: bool = False) -> tuple[BrowserContext, Page]:
    """Launch Playwright with a persistent user_data_dir profile.

    All cookies, localStorage, IndexedDB, and Cache are managed natively
    by the Chromium engine on disk. No synthetic injection needed.
    """
    _ensure_dirs()
    pw = await async_playwright().start()

    logger.info("Launching Playwright Chromium (native profile: %s)", PROFILE_DIR)

    # Randomize viewport per session to prevent fingerprint linkage
    session_viewport = get_random_viewport()
    logger.info("Session viewport: %dx%d", session_viewport["width"], session_viewport["height"])

    context = await pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=headless,
        viewport=session_viewport,
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

    # ── Attach lifecycle guard ──
    guard = SessionGuard.get()
    guard.attach(context)

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
    """Open a browser for human login. Chromium natively persists the session."""
    context, page = await launch_browser()

    print("\n" + "=" * 70)
    print("  🔐  MANUAL LOGIN MODE (Native Persistence)")
    print("=" * 70)
    print("  A Chromium browser window has opened.")
    print("  Please log into your Google/Gmail/Reddit account(s) now.")
    print("")
    print("  Once you are fully logged in and can see your inbox/feed,")
    print("  come back here and press ENTER to close the browser.")
    print("  Your session is saved AUTOMATICALLY in the browser profile.")
    print("=" * 70)

    try:
        input("\n  👉 Press ENTER when login is complete: ")
    except (EOFError, KeyboardInterrupt):
        logger.info("Input interrupted — closing browser...")

    try:
        await context.close()
    except Exception:
        pass
    SessionGuard.get().detach()

    print("\n  ✅ Session persisted! Future runs will use your login automatically.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Agent Action Schema
# ═══════════════════════════════════════════════════════════════════════════════
class AdvancedAction(BaseModel):
    reasoning: str = Field(description="Brief explanation of why this action is taken.")
    action_type: str = Field(description="One of: 'goto', 'click', 'type', 'scroll', 'press_enter', 'done'")
    url: str | None = Field(default=None, description="URL for 'goto' action")
    element_id: int | None = Field(default=None, description="Index of the interactive element")
    text: str | None = Field(default=None, description="Text to type for 'type' action")


class HeavyAction(BaseModel):
    """ReAct schema for Heavy Mode (Vision + DOM)."""
    current_state_analysis: str = Field(description="What exactly do you see on the screen right now? Are there any popups, overlays, or login prompts?")
    obstacle_detection: str = Field(description="Is the path to the goal blocked? If so, state the micro-plan to resolve it (e.g., 'Must click Continue with Google first').")
    reasoning: str = Field(description="Based on the analysis, why is this specific next action chosen?")
    action_type: str = Field(description="One of: 'goto', 'click', 'type', 'scroll', 'press_enter', 'done'")
    url: str | None = Field(default=None, description="URL for 'goto' action")
    element_id: int | None = Field(default=None, description="Index of the interactive element")
    text: str | None = Field(default=None, description="Text to type for 'type' action")


# ═══════════════════════════════════════════════════════════════════════════════
#  Smart DOM Compression (Priority-Tiered)
# ═══════════════════════════════════════════════════════════════════════════════
def _compress_dom_for_llm(raw_regions: list, max_tokens_budget: int = 2000) -> list[dict]:
    """Compress DOM elements into a token-efficient representation with priority tiers.

    Priority tiers:
      P0 (critical): input, textarea, select, button, [role=button]
      P1 (important): a (links with unique text)
      P2 (low): generic divs, navs, repeated elements

    Deduplicates elements with identical text and drops P2 elements
    when the budget is tight.
    """
    P0_TAGS = {"input", "textarea", "select", "button"}
    P0_ROLES = {"button", "textbox", "combobox", "searchbox", "checkbox", "radio"}

    seen_texts: set[str] = set()
    p0_items: list[dict] = []
    p1_items: list[dict] = []
    p2_items: list[dict] = []

    for i, r in enumerate(raw_regions):
        text_key = (r.tag + ":" + r.text[:60]).lower().strip()

        # Dedup: skip elements with identical tag+text
        if text_key in seen_texts and r.tag not in P0_TAGS:
            continue
        seen_texts.add(text_key)

        item = {
            "index": i,
            "tag": r.tag,
            "role": r.role,
            "text": r.text[:100],
            "x": r.x,
            "y": r.y,
        }

        if r.tag in P0_TAGS or r.role in P0_ROLES:
            p0_items.append(item)
        elif r.tag == "a" and r.text.strip():
            p1_items.append(item)
        else:
            p2_items.append(item)

    # Build final list within budget (rough estimate: ~30 tokens per element)
    budget_elements = max_tokens_budget // 30
    result = p0_items[:budget_elements]
    remaining = budget_elements - len(result)

    if remaining > 0:
        result.extend(p1_items[:remaining])
        remaining -= min(len(p1_items), remaining)

    if remaining > 0:
        result.extend(p2_items[:remaining])

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Agent Loop
# ═══════════════════════════════════════════════════════════════════════════════
async def _invoke_with_failover(
    failover_chain: list,
    messages: list,
    schema,
    circuit_breaker: CircuitBreaker | None = None,
    base64_image: str | None = None,
) -> tuple:
    """Try each model in the failover chain until one succeeds.

    Returns: (decision, model_name_used)
    Raises: Exception if ALL models fail or circuit breaker is tripped.
    """
    if circuit_breaker and circuit_breaker.tripped:
        raise RuntimeError(f"🔌 CIRCUIT BREAKER TRIPPED: {circuit_breaker.reason}")

    last_error = None
    for idx, llm in enumerate(failover_chain):
        model_name = getattr(llm, 'model_name', getattr(llm, 'model', str(llm)))
        
        # ── Dynamic Vision Payload Support ──
        supports_vision = any(x in model_name.lower() for x in [
            "gpt-4o", "claude-3-5", "claude-3-opus", "gemini-1.5", "glm-4v", "vision", "pixtral", "gemma"
        ])
        
        current_messages = messages
        if supports_vision and base64_image and len(messages) > 0:
            last_msg = messages[-1]
            if isinstance(last_msg, HumanMessage):
                multimodal_content = [
                    {"type": "text", "text": str(last_msg.content)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
                current_messages = messages[:-1] + [HumanMessage(content=multimodal_content)]
                
        try:
            structured = llm.with_structured_output(schema)
            decision = await structured.ainvoke(current_messages)
            if circuit_breaker:
                circuit_breaker.record_success()
            return decision, model_name
        except Exception as e:
            if circuit_breaker:
                circuit_breaker.record_failure()
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
            is_server_error = "500" in error_str or "502" in error_str or "503" in error_str
            is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower()

            if is_rate_limit:
                logger.warning("⚠️  FAILOVER [%d/%d]: %s hit RATE LIMIT (429) — switching to next model...",
                               idx + 1, len(failover_chain), model_name)
            elif is_server_error:
                logger.warning("⚠️  FAILOVER [%d/%d]: %s returned SERVER ERROR — switching to next model...",
                               idx + 1, len(failover_chain), model_name)
            elif is_timeout:
                logger.warning("⚠️  FAILOVER [%d/%d]: %s TIMED OUT — switching to next model...",
                               idx + 1, len(failover_chain), model_name)
            else:
                logger.warning("⚠️  FAILOVER [%d/%d]: %s failed (%s) — switching to next model...",
                               idx + 1, len(failover_chain), model_name, error_str[:150])
            last_error = e
            continue

    # All models exhausted
    raise RuntimeError(f"ALL {len(failover_chain)} models failed. Last error: {last_error}")


async def run_agent(objective: str):
    """Core single-agent action loop with native state persistence,
    zero-downtime LLM failover, GhostCursor humanization, adversarial critic,
    episodic campaign memory, and self-healing anti-detection (v8.0)."""

    context, page = await launch_browser()
    guard = SessionGuard.get()

    # ── Human Warm-Up Routine (WAF Bypass) ──
    # Establishes credible browsing history before deep navigation.
    target_hint = extract_target_url_from_objective(objective)
    try:
        await run_warmup(page, target_url=target_hint)
    except Exception as warmup_err:
        logger.warning("Warm-up routine failed (non-fatal): %s", warmup_err)

    # ── Enterprise modules ──
    memory = CampaignMemory()
    critic = ContentCritic()
    breaker = CircuitBreaker(max_requests=50, max_failures=10)

    # ── Build the full failover chain ──
    # Diagnostic-proven order: Groq (confirmed alive) → NVIDIA (fallback) → DeepSeek → Gemma
    primary, secondary, deepseek, tertiary = _build_worker_llms()
    # secondary=Groq (FAST, working), primary=NVIDIA (slow/timeout), tertiary=Gemma (vision)
    failover_chain = secondary + primary + deepseek + tertiary
    if not failover_chain:
        logger.error("No LLM clients available. Check API keys in .env.")
        await context.close()
        guard.detach()
        return

    chain_names = [getattr(m, 'model_name', getattr(m, 'model', '?')) for m in failover_chain]
    logger.info("LLM Failover Chain (%d models): %s", len(failover_chain), " → ".join(chain_names))
    logger.info("Objective: %s", objective)

    # ── Campaign memory context ──
    campaign_log = memory.get_campaign_log(limit=5)
    if campaign_log:
        logger.info("Campaign history: %s", memory.get_summary())

    action_history: list[str] = []
    url_history: list[str] = []  # Anti-Loop Watchdog tracker
    captcha_retries = 0
    heavy_mode_active = False
    last_action_signature = ""
    consecutive_identical_actions = 0
    max_steps = 20
    step = 0

    try:
        for step in range(max_steps):
            logger.info("━━━ Step %d/%d %s ━━━", step + 1, max_steps, breaker.status_line())

            # ── Circuit Breaker: abort if tripped ──
            if breaker.tripped:
                logger.critical("🔌 CIRCUIT BREAKER: %s — initiating graceful shutdown.", breaker.reason)
                break



            try:
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
                await page.wait_for_timeout(1500)
            except Exception:
                pass

            current_url = page.url
            logger.info("URL: %s", current_url)

            # ── Progressive Perception: Check for Action Stagnation ──
            if consecutive_identical_actions >= 2:
                heavy_mode_active = True
                logger.warning("🔄 Repeated action detected. Activating HEAVY MODE (Vision + DOM) for next step.")

            # ── Anti-Loop Watchdog: detect navigation loops ──
            url_history.append(current_url)
            loop_override = ""
            if len(url_history) >= 6:
                recent_urls = url_history[-6:]
                url_counts = Counter(recent_urls)
                most_common_url, visit_count = url_counts.most_common(1)[0]
                if visit_count >= 3:
                    heavy_mode_active = True
                    logger.warning(
                        "🔄 LOOP DETECTED: '%s' visited %d times in last 6 steps! Activating HEAVY MODE.",
                        most_common_url, visit_count,
                    )
                    loop_override = (
                        "\n\n⚠️ SYSTEM OVERRIDE: You are TRAPPED in a navigation loop. "
                        f"You have visited '{most_common_url}' {visit_count} times in the last 6 steps. "
                        "STOP revisiting the same pages. "
                        "If you are on a login page, you MUST click 'Continue with Google' "
                        "or 'Sign in with Google' IMMEDIATELY. "
                        "Do NOT navigate away from the login page. Authenticate FIRST. "
                        "If there is no Google login button visible, SCROLL DOWN to find it."
                    )

            # ── CAPTCHA / Cloudflare Detector ──
            try:
                captcha_signals = await page.evaluate("""
                () => {
                    const s = [];
                    if (document.querySelector('iframe[src*="challenges.cloudflare.com"]'))
                        s.push('cloudflare_turnstile');
                    if (document.querySelector('#cf-turnstile-container, .cf-turnstile'))
                        s.push('cloudflare_turnstile');
                    if (document.querySelector('iframe[src*="google.com/recaptcha"]'))
                        s.push('recaptcha_v2');
                    if (document.querySelector('.g-recaptcha'))
                        s.push('recaptcha_v2');
                    if (document.querySelector('iframe[src*="hcaptcha.com"]'))
                        s.push('hcaptcha');
                    const bt = (document.body?.innerText || '').toLowerCase();
                    if (bt.includes('verify you are human') || bt.includes('checking your browser'))
                        s.push('generic_challenge');
                    return s;
                }
                """)
                if captcha_signals:
                    captcha_type = captcha_signals[0]
                    captcha_retries += 1
                    if captcha_retries > 2:
                        logger.warning("🛡️ CAPTCHA ignored (hit retry limit of 2). Letting LLM proceed.")
                    else:
                        logger.warning("🛡️ CAPTCHA DETECTED: %s — attempting GhostCursor bypass (attempt %d)...", captcha_type, captcha_retries)
                        if captcha_type in ("cloudflare_turnstile", "recaptcha_v2", "hcaptcha"):
                            iframe_el = await page.query_selector(
                                'iframe[src*="challenges.cloudflare.com"], '
                                'iframe[src*="google.com/recaptcha"], '
                                'iframe[src*="hcaptcha.com"]'
                            )
                            if iframe_el:
                                box = await iframe_el.bounding_box()
                                if box:
                                    click_x = box["x"] + 30
                                    click_y = box["y"] + 30
                                    await ghost_click(page, click_x, click_y)
                                    logger.info("🛡️ Clicked CAPTCHA checkbox at (%d, %d)", click_x, click_y)
                                    await asyncio.sleep(4)
                        elif captcha_type == "generic_challenge":
                            logger.info("🛡️ Waiting 8s for Cloudflare browser check to complete...")
                            await asyncio.sleep(8)
                        continue  # Re-evaluate the page after handling the challenge
                else:
                    captcha_retries = 0  # Reset if no challenge found
            except Exception as e:
                logger.warning("CAPTCHA detection failed: %s", e)

            try:
                raw_regions = await _extract_interactive_regions(page, max_regions=150)
            except Exception as e:
                logger.warning("DOM extraction failed: %s — retrying next step", e)
                continue

            # ── Smart DOM Compression (Priority-Tiered) ──
            compact = _compress_dom_for_llm(raw_regions, max_tokens_budget=2000)

            # ── Progressive Perception (Heavy Mode) ──
            base64_image = None
            cleaned_dom = ""
            if heavy_mode_active:
                logger.info("👁️ Capturing heavy perception (Screenshot + DOM)...")
                try:
                    screenshot_bytes = await page.screenshot(type="jpeg", quality=50)
                    import base64
                    base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
                    cleaned_dom = await page.evaluate('''() => {
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll("script, style, svg, img, video").forEach(el => el.remove());
                        return clone.innerText.substring(0, 3000);
                    }''')
                except Exception as e:
                    logger.warning("Heavy perception capture failed: %s", e)
                
                schema_to_use = HeavyAction
                system_prompt = (
                    "You are an elite autonomous web agent operating inside a real browser.\n"
                    "HEAVY MODE ACTIVATED: You are stuck. Look at the provided screenshot and DOM text to figure out what is blocking you (e.g. a popup, login overlay, or layout shift).\n"
                    "You must output 'current_state_analysis' and 'obstacle_detection' before deciding the next action.\n"
                    "IMPORTANT: If you already typed text into a field, do NOT type it again. Move to the NEXT step.\n"
                    "Available actions: goto, click, type, scroll, press_enter, done.\n"
                )
                user_prompt = (
                    f"OBJECTIVE: {objective}\n"
                    f"CURRENT URL: {current_url}\n"
                    f"COMPLETED ACTIONS: {json.dumps(action_history)}\n"
                    f"INTERACTIVE ELEMENTS:\n{json.dumps(compact, indent=2)}\n\n"
                    f"PAGE TEXT CONTEXT (DOM):\n{cleaned_dom}\n\n"
                    "What is your NEXT action?"
                    f"{loop_override}"
                )
            else:
                schema_to_use = AdvancedAction
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
                    f"{loop_override}"
                )

            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            current_chain = failover_chain
            if heavy_mode_active:
                # Forcefully prioritize Gemma models for vision
                gemma_models = [m for m in failover_chain if "gemma" in getattr(m, 'model_name', getattr(m, 'model', str(m))).lower()]
                other_models = [m for m in failover_chain if m not in gemma_models]
                if gemma_models:
                    current_chain = gemma_models + other_models
                    logger.info("👁️ Vision needed: forcefully routing to %s", getattr(gemma_models[0], 'model_name', 'Gemma'))

            logger.info("Querying LLM (failover chain: %d models)...", len(current_chain))
            try:
                decision, used_model = await _invoke_with_failover(
                    current_chain, messages, schema_to_use, circuit_breaker=breaker, base64_image=base64_image
                )
                logger.info("Answered by: %s", used_model)
            except RuntimeError as e:
                logger.error("LLM FAILURE: %s — pausing 30s before retry...", e)
                await asyncio.sleep(30)
                continue

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
                    await ghost_click(page, cx, cy)
                except Exception as e:
                    logger.warning("Click failed: %s", e)

            elif decision.action_type == "type":
                if decision.element_id is None or decision.element_id >= len(raw_regions):
                    logger.warning("Invalid element_id %s", decision.element_id)
                    continue
                t = raw_regions[decision.element_id]
                cx, cy = t.x + t.width / 2, t.y + t.height / 2

                text_to_type = decision.text or ""

                # ── Adversarial Critic: review body content before typing ──
                if len(text_to_type) > 100:
                    logger.info("Content Critic: reviewing %d-char body before typing...", len(text_to_type))
                    try:
                        verdict = await critic.review(
                            content=text_to_type,
                            platform=current_url,
                            failover_chain=failover_chain,
                            invoke_fn=_invoke_with_failover,
                        )
                        if not verdict.approved and verdict.rewritten_content:
                            logger.info(
                                "Critic rewrote content (tone=%.2f, issues=%d): %s",
                                verdict.tone_score, len(verdict.issues),
                                "; ".join(verdict.issues[:3]),
                            )
                            text_to_type = verdict.rewritten_content
                        elif verdict.approved:
                            logger.info("Critic approved content (tone=%.2f)", verdict.tone_score)
                    except Exception as e:
                        logger.warning("Critic review failed: %s — using original content", e)

                logger.info("Typing '%s' into (%s '%s')", text_to_type[:80], t.tag, t.text[:40])
                try:
                    await ghost_click(page, cx, cy)
                    await asyncio.sleep(0.3)
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Backspace")
                    await ghost_type(page, text_to_type)
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
                    await ghost_scroll(page, 600)
                except Exception as e:
                    logger.warning("Scroll failed: %s", e)

            # ── Progressive Perception: Track action state ──
            current_action_signature = f"{decision.action_type}:{decision.element_id}:{decision.text}"
            if current_action_signature == last_action_signature:
                consecutive_identical_actions += 1
            else:
                consecutive_identical_actions = 0
                if heavy_mode_active:
                    logger.info("✅ Progress detected. Deactivating HEAVY MODE.")
                    heavy_mode_active = False

            last_action_signature = current_action_signature

        logger.info("Agent loop finished after %d steps.", step + 1)

        # ── Record in campaign memory ──
        try:
            platform = "unknown"
            final_url = page.url
            for p in ["dev.to", "medium.com", "x.com", "twitter.com", "hashnode.com", "github.com"]:
                if p in final_url:
                    platform = p
                    break
            memory.record_post(
                platform=platform,
                title=objective[:120],
                content=objective,
                url=final_url,
                agent_model=chain_names[0] if chain_names else "unknown",
                steps_taken=step + 1,
            )
        except Exception as e:
            logger.warning("Campaign memory record failed: %s", e)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt caught — saving session before exit...")

    except Exception as e:
        logger.error("Unexpected crash: %s", e)
        logger.error(traceback.format_exc())

    finally:
        # ════════════════════════════════════════════════════════════════════
        #  CLEAN SHUTDOWN — context.close() flushes all native state to disk
        # ════════════════════════════════════════════════════════════════════
        logger.info("Shutting down...")
        try:
            await context.close()
            logger.info("Browser closed. Native profile state persisted.")
        except Exception as e:
            logger.warning("Browser close failed: %s", e)
        guard.detach()


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
