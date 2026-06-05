"""Advanced Single-Agent Browser Automation with Deep Intelligence (v9.0).

v9.0 Deep Merge Architecture:
  - ModelRegistry: ALL API keys, strict TEXT/VISION separation, singleton
  - Mission Planner: pre-loop task decomposition into checkpoints
  - Progressive State Machine: 7-phase tracker with failure memory
  - Action Outcome Verifier: post-action page-change detection
  - Self-Correction Engine: auto-recover from silently failing actions
  - Recovery Advisor: separate LLM call when stuck
  - Continuous Campaign Logger: records at every phase transition
  - Enhanced Failover: 3s recovery, not 30s dead time

Session Persistence:
  Pure native user_data_dir — Chromium manages all cookies, localStorage,
  IndexedDB, and Cache natively on disk.
"""

import argparse
import asyncio
import atexit
import base64
import json
import signal
import sys
import time as _time
import traceback
from collections import Counter, deque
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
    VISUAL_CURSOR_INIT_SCRIPT,
)
from app.browser_promoter.browser_warmup import (
    run_warmup,
    extract_target_url_from_objective,
)
from app.browser_promoter.worker_planner import ReasoningAgent, VisionAgent
from app.logger import get_logger
from playwright.async_api import async_playwright, BrowserContext, Page

# Enterprise modules
from ghost_input import ghost_click, ghost_type, ghost_scroll
from campaign_memory import CampaignMemory
from content_critic import ContentCritic
from model_registry import ModelRegistry
import dom_parser
from content_store import ContentStore
from form_engine import FormState, classify_field
from platform_detect import detect_platform, get_char_limit
from verification_engine import take_snapshot, verify_outcome, fingerprint_ax_tree

# Cognitive Architecture V2.1
from cognitive_core import PlanState, WorkingMemory, dom_data_to_a11y_format, AgentMetrics
from orchestrator.critic_v12 import CriticV12, Verdict
from execution_safety import is_domain_allowed, cove_pre_done_check

logger = get_logger("advanced_agent")





# ═══════════════════════════════════════════════════════════════════════════════
#  v10.0 Progressive State Machine — Multi-field tracker with failure memory
# ═══════════════════════════════════════════════════════════════════════════════

class AgentPhaseTracker:
    """Track completed phases per field, failed actions, and correction context."""

    def __init__(self):
        self.phases = {
            "navigated_to_target": False,
            "login_verified": False,
            "compose_opened": False,
            "title_filled": False,
            "body_filled": False,
            "all_fields_ready": False,
            "submit_clicked": False,
            "submit_verified": False,
        }
        self.typed_text = ""
        self.current_checkpoint = ""
        self.failed_actions: list[str] = []
        self.correction_context = ""
        self.recovery_advice = ""
        # v10.0: track what's in each field
        self.title_content = ""
        self.body_content = ""

    def mark(self, phase: str):
        if phase in self.phases:
            self.phases[phase] = True
            logger.info("✅ Phase '%s' completed", phase)

    def is_done(self, phase: str) -> bool:
        return self.phases.get(phase, False)

    def progress_pct(self) -> int:
        done = sum(1 for v in self.phases.values() if v)
        return int(100 * done / max(len(self.phases), 1))

    # Backwards compatibility aliases
    @property
    def text_typed_done(self) -> bool:
        return self.is_done("title_filled") or self.is_done("body_filled")

    def get_state_override(self, has_article: bool = False) -> str:
        """Build contextual state info for the LLM based on field fill status."""
        parts = []

        title_done = self.is_done("title_filled")
        body_done = self.is_done("body_filled")
        all_ready = self.is_done("all_fields_ready")

        if all_ready:
            parts.append(
                "\n\n🚨 ALL FIELDS ARE FILLED. Click the Publish/Post/Submit button NOW. "
                "Do NOT type anything else."
            )
        elif title_done and not body_done:
            if has_article:
                parts.append(
                    f"\n\n📝 FIELD STATUS: Title filled ✓ ({len(self.title_content)} chars). "
                    "Body/Content field is still EMPTY — click on the body/content textarea "
                    "and the system will auto-inject the article content."
                )
            else:
                parts.append(
                    f"\n\n📝 FIELD STATUS: Title/First field filled ✓. "
                    "Now type into the body/content field, or click Post/Submit if finished."
                )
        elif title_done and body_done:
            parts.append(
                f"\n\n📝 FIELD STATUS: Title ✓ ({len(self.title_content)} chars) | "
                f"Body ✓ ({len(self.body_content)} chars). "
                "All fields filled — click Publish/Post/Submit button."
            )
        elif not title_done and self.typed_text:
            parts.append(
                f"\n\n📝 FIELD STATUS: You typed: '{self.typed_text[:60]}...' — "
                "now fill the remaining fields or click Submit."
            )


        if self.correction_context:
            parts.append(self.correction_context)
        if self.recovery_advice:
            parts.append(f"\n\n🛠️ RECOVERY ADVICE: {self.recovery_advice}")
        return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  v9.0 Mission Planner — Decomposes objective into checkpoints
# ═══════════════════════════════════════════════════════════════════════════════

async def _decompose_objective(
    objective: str, failover_chain: list, breaker, health_tracker,
) -> list[str]:
    """Break objective into 3-5 sequential checkpoints using the TEXT chain."""
    prompt = (
        "Break this browser automation objective into 3-5 short sequential checkpoints. "
        "Each checkpoint is ONE specific browser action (navigate, click, type, verify). "
        "Return ONLY a JSON array of strings. No markdown, no explanation.\n\n"
        f"Objective: {objective}"
    )
    try:
        response, _ = await _invoke_with_failover(
            failover_chain, [HumanMessage(content=prompt)],
            None, breaker, health_tracker=health_tracker,
        )
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        checkpoints = json.loads(text)
        if isinstance(checkpoints, list) and all(isinstance(c, str) for c in checkpoints):
            return checkpoints[:7]
    except Exception as e:
        logger.warning("Mission planner failed: %s — using objective as single checkpoint", e)
    return [objective]


# ═══════════════════════════════════════════════════════════════════════════════
#  v2.1 Three-Layer Action Grounding (RQ-B06 UPGRADE)
#  Layer 1: element_id → selector_map lookup (Browser-Use pattern, PREFERRED)
#  Layer 2: elementFromPoint hit-test (from report B, FALLBACK)
#  Layer 3: nearest-element snap by distance (v2.0, TERTIARY)
# ═══════════════════════════════════════════════════════════════════════════════

async def _ground_or_reject(
    decision, selector_map: dict, elements_list: list[dict],
    page, threshold: float = 60.0
) -> tuple[dict | None, str | float]:
    """Three-layer grounding validation (v2.1 RQ-B06).

    Layer 1 (PREFERRED): element_id → selector_map lookup
    Layer 2 (FALLBACK):  elementFromPoint hit-test on coordinates
    Layer 3 (TERTIARY):  nearest-element snap by distance

    Returns:
        (element_dict, distance_float) if grounded successfully
        (None, reason_string) if coordinates are hallucinated / rejected
    """
    # ── Layer 1: Element ID resolution (Browser-Use pattern) ──
    if decision.element_id is not None:
        el = selector_map.get(decision.element_id)
        if el is None:
            return None, f"element_id '{decision.element_id}' not in current snapshot"
        # Resolve coordinates from the element
        decision.x = float(el.get("x", decision.x or 0))
        decision.y = float(el.get("y", decision.y or 0))
        logger.info(
            "🎯 Grounded via element_id [%s] '%s' → (%d,%d)",
            decision.element_id,
            el.get("text", el.get("name", "?"))[:30],
            int(decision.x), int(decision.y),
        )
        return el, 0.0  # perfect ground

    # Below here: LLM gave coordinates but no element_id
    if decision.x is None or decision.y is None:
        return None, "no element_id or coordinates"

    # ── Layer 2: elementFromPoint hit-test (from report B) ──
    try:
        hit = await page.evaluate(
            "([x,y])=>{const e=document.elementFromPoint(x,y);"
            "if(!e)return null;"
            "const tag=e.tagName.toUpperCase();"
            "const interactive=['A','BUTTON','INPUT','SELECT','TEXTAREA']"
            ".includes(tag)||e.getAttribute('role')==='button'"
            "||e.getAttribute('contenteditable')==='true';"
            "return{tag,interactive,text:(e.textContent||'').trim().slice(0,40)};}",
            [decision.x, decision.y],
        )
        if hit and hit.get("interactive"):
            logger.info(
                "🎯 Grounded via hit-test: %s '%s' at (%d,%d)",
                hit.get("tag", "?"), hit.get("text", "?")[:30],
                int(decision.x), int(decision.y),
            )
            return {"ref": "hit-test", "name": hit.get("text", "")[:30],
                    "x": decision.x, "y": decision.y}, 0.0
    except Exception:
        pass

    # ── Layer 3: Nearest-element snap (v2.0 fallback) ──
    if not elements_list:
        return None, "no elements in snapshot"
    best_el: dict | None = None
    best_dist = float("inf")
    for el in elements_list:
        ex = el.get("x", 0)
        ey = el.get("y", 0)
        if ex == 0 and ey == 0:
            continue  # Skip elements with no position data
        dist = ((decision.x - ex) ** 2 + (decision.y - ey) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_el = el

    if best_el is None:
        return None, "no positioned elements in snapshot"
    if best_dist > threshold:
        return None, (
            f"nearest element [{best_el.get('id', best_el.get('ref', '?'))}] "
            f"'{best_el.get('text', best_el.get('name', '?'))[:25]}' "
            f"is {best_dist:.0f}px away (>{threshold:.0f}px threshold)"
        )
    # Snap to real element center
    decision.x = float(best_el.get("x", decision.x))
    decision.y = float(best_el.get("y", decision.y))
    if best_dist > 5.0:
        logger.info(
            "🎯 Grounded via snap to [%s] '%s' (%.0fpx)",
            best_el.get('id', best_el.get('ref', '?')),
            best_el.get('text', best_el.get('name', '?'))[:30],
            best_dist,
        )
    return best_el, best_dist


# ═══════════════════════════════════════════════════════════════════════════════
#  v9.0 Recovery Advisor — Asks a separate LLM call to unblock
# ═══════════════════════════════════════════════════════════════════════════════

async def _ask_recovery_advisor(
    objective: str, current_url: str, action_history: list,
    failover_chain: list, breaker, health_tracker,
) -> str:
    """When stuck, ask a text LLM for expert recovery advice."""
    prompt = (
        "You are a browser automation RECOVERY CONSULTANT. The agent is STUCK. "
        "Analyze the situation and give ONE specific actionable instruction.\n\n"
        f"OBJECTIVE: {objective}\nURL: {current_url}\n"
        f"RECENT ACTIONS:\n{json.dumps(action_history[-6:], indent=1)}\n\n"
        "Respond in 1-2 sentences ONLY."
    )
    try:
        response, model = await _invoke_with_failover(
            failover_chain, [HumanMessage(content=prompt)],
            None, breaker, health_tracker=health_tracker,
        )
        advice = response.content if hasattr(response, "content") else str(response)
        logger.info("🛠️ Recovery Advisor (%s): %s", model, advice[:200])
        return advice.strip()
    except Exception as e:
        logger.warning("Recovery advisor failed: %s", e)
        return "Try scrolling down or dismissing any visible popups."


# ═══════════════════════════════════════════════════════════════════════════════
#  v9.0 Continuous Campaign Logger
# ═══════════════════════════════════════════════════════════════════════════════

def _record_phase_transition(memory, platform: str, objective: str, url: str, step: int, phase: str):
    """Record progress to campaign memory at each phase transition."""
    try:
        memory.record_post(
            platform=platform,
            title=f"[Phase:{phase}] {objective[:80]}",
            content=objective,
            url=url,
            agent_model="v9.0-deep-merge",
            steps_taken=step + 1,
        )
    except Exception as e:
        logger.warning("Campaign log failed: %s", e)

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
        args=STEALTH_LAUNCH_ARGS + dom_parser.TLS_STEALTH_ARGS,
        ignore_default_args=["--enable-automation"],
    )
    await context.add_init_script(STEALTH_INIT_SCRIPT)
    await context.add_init_script(VISUAL_CURSOR_INIT_SCRIPT)
    await dom_parser.install_shadow_piercer(context)

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
#  Autonomous Action Schema — Observe-Think-Act (OTA) Loop
# ═══════════════════════════════════════════════════════════════════════════════
class AutonomousAction(BaseModel):
    """ReAct-style autonomous action with mandatory chain-of-thought.

    The LLM MUST populate the 4 thinking fields before choosing an action.
    This prevents blind step-following and forces genuine situational awareness.
    """

    # ── OBSERVE: What does the screen show? ──
    screen_state: str = Field(
        description=(
            "Describe what you SEE on the current screen in 1-3 sentences. "
            "Include: page type (login page, feed, compose box, error page), "
            "key visible elements (buttons, inputs, text), any popups/overlays/modals, "
            "and whether the user appears logged in or logged out."
        )
    )

    # ── THINK: Reason through the situation ──
    previous_action_result: str = Field(
        description=(
            "Evaluate the outcome of your LAST action. Did the page change? "
            "Did a new element appear? Did an error show up? "
            "If this is the very first step, say 'First action — no prior state.'"
        )
    )
    goal_progress: str = Field(
        description=(
            "Assess progress toward the ULTIMATE goal. What percentage is done? "
            "What specific sub-tasks remain? "
            "Example: '40% done — logged in successfully, now need to find compose box and type the tweet.'"
        )
    )
    reasoning: str = Field(
        description=(
            "Based on screen_state and goal_progress, explain WHY this specific "
            "next action is the correct choice. Consider alternatives you rejected. "
            "Example: 'I see the compose box is visible at element e5. I should click it to focus, "
            "then type the tweet content. I considered scrolling first but the box is already visible.'"
        )
    )

    # ── ACT: The chosen action ──
    action_type: str = Field(
        description="One of: 'goto', 'click', 'type', 'scroll', 'press_enter', 'wait', 'done'"
    )
    element_id: str | None = Field(
        default=None,
        description=(
            "PREFERRED: The element ID from the page structure (e.g., 'e5'). "
            "Use this instead of x,y coordinates when possible. "
            "The system will resolve the element's real position automatically."
        ),
    )
    url: str | None = Field(default=None, description="URL for 'goto' action")
    x: float | None = Field(default=None, description="X coordinate (fallback if element_id unavailable)")
    y: float | None = Field(default=None, description="Y coordinate (fallback if element_id unavailable)")
    text: str | None = Field(default=None, description="Text to type for 'type' action")
    wait_ms: int | None = Field(default=None, description="Milliseconds to wait for 'wait' action (default: 1000)")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Agent Loop — Enhanced Failover (v9.0)
# ═══════════════════════════════════════════════════════════════════════════════
async def _invoke_with_failover(
    failover_chain: list,
    messages: list,
    schema,
    circuit_breaker=None,
    base64_image: str | None = None,
    health_tracker=None,
) -> tuple:
    """Try each model in the failover chain until one succeeds.

    v9.0 improvements:
      - Configurable timeout (30s default, not hardcoded 20s)
      - 3s quarantine wait, not 30s dead time
      - Proper exponential backoff via ProviderHealthTracker
    """
    if circuit_breaker and circuit_breaker.tripped:
        raise RuntimeError(f"🔌 CIRCUIT BREAKER TRIPPED: {circuit_breaker.reason}")

    last_error = None
    model_names = [
        getattr(m, 'model_name', getattr(m, 'model', str(m)))
        for m in failover_chain
    ]

    # If all providers quarantined, wait briefly for recovery
    if health_tracker and health_tracker.all_quarantined(model_names):
        wait_time = min(health_tracker.shortest_quarantine_wait() + 0.5, 8.0)
        logger.warning("⏳ All providers quarantined. Waiting %.1fs...", wait_time)
        await asyncio.sleep(wait_time)

    for idx, llm in enumerate(failover_chain):
        model_name = model_names[idx]

        if health_tracker and not health_tracker.is_available(model_name):
            if idx < len(failover_chain) - 1:
                continue

        # Dynamic Vision Payload Support
        supports_vision = any(x in model_name.lower() for x in [
            "gpt-4o", "claude-3-5", "gemini", "glm-4v", "vision", "pixtral", "gemma"
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
            if schema is not None:
                structured = llm.with_structured_output(schema)
            else:
                structured = llm
            decision = await asyncio.wait_for(
                structured.ainvoke(current_messages),
                timeout=30.0,
            )
            if circuit_breaker:
                circuit_breaker.record_success()
            if health_tracker:
                health_tracker.record_success(model_name)
            return decision, model_name
        except asyncio.TimeoutError:
            if circuit_breaker:
                circuit_breaker.record_failure()
            if health_tracker:
                health_tracker.record_failure(model_name)
            logger.warning("⚠️  FAILOVER [%d/%d]: %s TIMED OUT (>30s)", idx + 1, len(failover_chain), model_name)
            last_error = TimeoutError(f"{model_name} timed out")
        except Exception as e:
            if circuit_breaker:
                circuit_breaker.record_failure()
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
            if health_tracker:
                health_tracker.record_failure(model_name, is_rate_limit=is_rate_limit)
            logger.warning("⚠️  FAILOVER [%d/%d]: %s — %s", idx + 1, len(failover_chain), model_name, error_str[:150])
            if is_rate_limit:
                await asyncio.sleep(1.5)
            last_error = e

    raise RuntimeError(f"ALL {len(failover_chain)} models failed. Last: {last_error}")


async def run_agent(objective: str):
    """Core autonomous agent loop — v9.0 Deep Intelligence Architecture.

    Features: ModelRegistry, Mission Planner, Progressive State Machine,
    Action Verifier, Self-Correction Engine, Recovery Advisor,
    Continuous Campaign Logger, Enhanced Failover.
    """
    context, page = await launch_browser()
    guard = SessionGuard.get()

    # ── Human Warm-Up Routine ──
    target_hint = extract_target_url_from_objective(objective)
    try:
        await run_warmup(page, target_url=target_hint)
    except Exception as warmup_err:
        logger.warning("Warm-up routine failed (non-fatal): %s", warmup_err)

    # ── v9.0: ModelRegistry as single source of truth ──
    registry = ModelRegistry.get_instance()
    memory = CampaignMemory()
    critic = ContentCritic()
    breaker = registry.breaker  # Shared circuit breaker
    health_tracker = registry.health  # Shared health tracker

    # Agents pull from ModelRegistry automatically
    vision_agent = VisionAgent()
    reasoning_agent = ReasoningAgent()

    failover_chain = reasoning_agent.get_failover_chain()
    chain_names = reasoning_agent.get_chain_names()

    if not failover_chain:
        logger.error("No LLM clients available. Check API keys in .env.")
        await context.close()
        guard.detach()
        return

    logger.info("LLM Failover Chain (%d models): %s", len(failover_chain), " → ".join(chain_names))
    logger.info("Objective: %s", objective[:200])

    # ── v10.0: Content Store — separate long articles from objective ──
    store = ContentStore(base_dir=Path(__file__).parent / "persistence")
    llm_objective, has_article = store.load_from_objective(objective)
    if has_article:
        logger.info(
            "📄 Article loaded: title='%s' | body=%d chars | platform=%s",
            store.get_title()[:60], len(store.get_body()), store.article.target_platform,
        )
        logger.info("📄 Shortened objective for LLM (%d chars)", len(llm_objective))
    else:
        llm_objective = objective

    # ── Campaign memory context ──
    campaign_log = memory.get_campaign_log(limit=5)
    if campaign_log:
        logger.info("Campaign history: %s", memory.get_summary())

    # ── v2.0: Mission Planner → PlanState (fixes B-01 amnesia, B-10 dead checkpoints) ──
    mission_checkpoints = await _decompose_objective(llm_objective, failover_chain, breaker, health_tracker)
    plan = PlanState.from_checkpoints(llm_objective, mission_checkpoints)
    working_mem = WorkingMemory()
    metrics = AgentMetrics()  # v2.1: Quality metrics tracking (S-5)
    logger.info("📋 Mission Plan (%d steps):", len(plan.steps))
    for s in plan.steps:
        logger.info("  [%d] %s", s["id"] + 1, s["desc"])

    action_history: deque[str] = deque(maxlen=50)    # Reduced — WorkingMemory handles the rest
    url_history: deque[str] = deque(maxlen=50)         # V-01: bounded ring buffer
    last_action_signature = ""
    consecutive_identical_actions = 0
    correction_failures = 0  # v9.0: Self-correction streak counter
    # v2.1: Dynamic budget (RQ-B05) — base + field bonus, extensible on progress
    base_budget = 20
    field_bonus = 5 if store.has_article else 0
    max_steps = base_budget + field_bonus
    step = 0

    # ── v13.0: Form State Machine (replaces phase_tracker aliases) ──
    form_state = FormState(
        required_fields={"title", "body"} if store.has_article else {"body"}
    )
    # ── v9.0: Progressive State Machine (kept for navigation/submit phases) ──
    phase_tracker = AgentPhaseTracker()
    # Backwards compat aliases (kept for code paths that still read them)
    completed_phases = phase_tracker.phases
    typed_text_content = ""

    # ── v13.0: Triple-state loop detector (RQ-12) ──
    _loop_window: deque[tuple] = deque(maxlen=6)

    # ── v2.0: CriticV12 (RQ-B03) — replaces dead hash verifier ──
    progress_critic: CriticV12 | None = None  # initialized after page is ready

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

            # ── v2.0: Initialize CriticV12 on first step (needs page reference) ──
            if progress_critic is None:
                progress_critic = CriticV12(page)
                logger.info("🧠 CriticV12 initialized (6-signal progress detection)")

            # ── Anti-Loop Watchdog v13: (url, dom_fp, action_sig) triple ──
            url_history.append(current_url)
            loop_context = ""
            if len(url_history) >= 6:
                recent_urls = list(url_history)[-6:]
                url_counts = Counter(recent_urls)
                most_common_url, visit_count = url_counts.most_common(1)[0]
                if visit_count >= 3:
                    logger.warning(
                        "🔄 LOOP DETECTED: '%s' visited %d times in last 6 steps!",
                        most_common_url, visit_count,
                    )
                    loop_context = (
                        f"\n\n⚠️ CRITICAL LOOP WARNING: You have visited '{most_common_url}' "
                        f"{visit_count} times in the last 6 steps. You are stuck. "
                        "STOP and THINK deeply: Why are you repeating the same navigation? "
                        "Your previous approach is NOT working. You MUST try a DIFFERENT strategy: "
                        "scroll to reveal hidden elements, wait for dynamic content to load, "
                        "click a different element, or re-evaluate whether the goal is already achieved. "
                        "Do NOT repeat the same action you just tried."
                    )

            # ── Determine Context-Aware Target Hint for Sniper Mode ──
            target_hint = None
            if store.has_article:
                if not phase_tracker.is_done("title_filled"):
                    target_hint = "title"
                elif not phase_tracker.is_done("body_filled"):
                    target_hint = "body"
                elif not phase_tracker.is_done("submit_clicked"):
                    target_hint = "submit"

            # ═══ TIER 1: God-Mode DOM Parser V11 (instant, ~50ms) ═══
            dom_data = await dom_parser.extract(page, target_hint=target_hint, timeout=5.0)
            elements_list = dom_data.get("elements", [])
            dom_markdown = dom_data.get("markdown", "")

            # v2.1: Build selector_map for element-ID grounding (Browser-Use pattern)
            selector_map: dict[str, dict] = {}
            for el in elements_list:
                eid = el.get("id", el.get("ref", ""))
                if eid:
                    selector_map[eid] = el
            vision_map_json = json.dumps({"elements": elements_list, "image_size": dom_data.get("image_size", {})}, ensure_ascii=True)

            # ═══ TIER 2: Vision API — ABSOLUTE LAST RESORT ═══
            # Only fires if the DOM parser found literally zero elements
            # (e.g., the page is a full-canvas app or a PDF viewer)
            if not elements_list:
                logger.warning("God-Mode DOM found 0 elements — falling back to Vision API")
                try:
                    screenshot_bytes = await page.screenshot(type="jpeg", quality=55)
                    base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
                except Exception as e:
                    logger.warning("Screenshot capture failed: %s", e)
                    continue

                vision_map_json = await vision_agent.detect_elements(
                    screenshot_base64=base64_image,
                    screenshot_encoding="base64:jpeg",
                )
                vision_data = json.loads(vision_map_json) if vision_map_json.strip() else {}
                elements_list = vision_data.get("elements", [])
                if elements_list:
                    logger.info("Vision fallback: extracted %d elements", len(elements_list))
                else:
                    logger.warning("Vision fallback also empty — page may be non-interactive")

            # ── Login State Detector: check if user is already authenticated ──
            login_hint = ""
            try:
                login_state = await asyncio.wait_for(page.evaluate("""
                () => {
                    const url = window.location.href;
                    const body = document.body ? document.body.innerText.slice(0, 500) : '';
                    const hasProfile = !!document.querySelector('[data-testid="AppTabBar_Profile_Link"], [aria-label="Profile"], img[alt*="profile" i]');
                    const hasCompose = !!document.querySelector('[data-testid="tweetTextarea_0"], [data-testid="tweetButtonInline"], [aria-label="Post"]');
                    const hasFeed = !!document.querySelector('[data-testid="primaryColumn"], [data-testid="tweet"]');
                    const hasLoginForm = !!document.querySelector('input[name="text"], input[name="password"], [data-testid="LoginForm_Login_Button"]');
                    return {
                        url: url,
                        hasProfile: hasProfile,
                        hasCompose: hasCompose,
                        hasFeed: hasFeed,
                        hasLoginForm: hasLoginForm
                    };
                }
                """), timeout=5.0)

                if login_state.get("hasProfile") or login_state.get("hasFeed") or login_state.get("hasCompose"):
                    login_hint = (
                        "\n\n🔑 LOGIN STATUS: You are ALREADY LOGGED IN. "
                        "Profile icon and/or feed content detected. "
                        "DO NOT attempt to log in again. Skip any login steps and proceed directly to your goal."
                    )
                    if login_state.get("hasCompose"):
                        login_hint += " A tweet compose area is available on this page."
                    logger.info("🔑 Login detected: profile=%s feed=%s compose=%s",
                                login_state.get("hasProfile"), login_state.get("hasFeed"), login_state.get("hasCompose"))
                elif login_state.get("hasLoginForm"):
                    login_hint = "\n\n🔑 LOGIN STATUS: You are NOT logged in. A login form is visible."
                    logger.info("🔑 Login form detected — user is logged out")
            except Exception as login_err:
                logger.warning("Login state detection failed: %s", login_err)

            # ════════════════════════════════════════════════════════════════
            #  OTA SYSTEM PROMPT — Autonomous Observe-Think-Act Persona
            # ════════════════════════════════════════════════════════════════
            # ── v2.0: Plan + Memory injected into system prompt (RQ-B01) ──
            plan_context = plan.render()
            facts_context = working_mem.render_facts()

            system_prompt = (
                "You are an elite autonomous web agent with FULL CREATIVE FREEDOM. "
                "You operate inside a real Chromium browser and control it through actions.\n\n"

                "═══ YOUR CURRENT PLAN ═══\n"
                f"{plan_context}\n\n"

                + (f"═══ WHAT YOU KNOW ═══\n{facts_context}\n\n" if facts_context else "")

                + "═══ CORE IDENTITY ═══\n"
                "You are an INDEPENDENT THINKER, not a step-follower. "
                "The user gives you a GOAL (their intent). Any numbered steps they provide are "
                "SUGGESTIONS, not commands. YOU decide the actual actions based on what you SEE.\n\n"

                "═══ OBSERVE-THINK-ACT LOOP ═══\n"
                "Every turn you MUST:\n"
                "1. OBSERVE: Describe what you actually see on screen (screen_state field).\n"
                "   - Is this a login page? A feed? A compose dialog? An error?\n"
                "   - Are there popups, overlays, cookie banners, or CAPTCHAs blocking you?\n"
                "   - Is the user logged in or logged out? Look for profile icons, usernames, feeds.\n"
                "2. THINK: Reason through the situation (previous_action_result + goal_progress + reasoning).\n"
                "   - Did your last action work? How do you know?\n"
                "   - What fraction of the goal is done? What remains?\n"
                "   - Consider at least 2 possible next actions and explain why you chose one.\n"
                "3. ACT: Execute exactly ONE action based on your reasoning.\n\n"

                "═══ AUTONOMOUS INTELLIGENCE RULES ═══\n"
                "• NEVER blindly follow user steps if the screen shows a different state. "
                "If the user says 'login' but you see a feed with a profile icon, you are ALREADY logged in — skip login.\n"
                "• If an action FAILED (you can see from action history), do NOT repeat it. Try an alternative approach.\n"
                "• If you see a popup, overlay, modal, or cookie banner, HANDLE IT FIRST before continuing your main task.\n"
                "• ARTICLES: For long articles, the content is pre-loaded. Type ONLY the title into the title field. "
                "For the body/content field, just click on it with a type action — the system will auto-inject the full article.\n"
                "• MULTI-FIELD FORMS: On platforms like dev.to, there are SEPARATE title and body fields. "
                "Fill the TITLE first, then click the BODY field. Do NOT re-type the title.\n"
                "• If the FIELD STATUS says 'All fields filled', your ONLY remaining action is to click Publish/Submit.\n"
                "• SCROLL if you cannot find the target element in the vision map — it may be below the fold.\n"
                "• Use 'wait' (1000-2000ms) after clicks that trigger page loads or dynamic content.\n"
                "• When the goal is FULLY achieved, output action_type='done' immediately. Do not take unnecessary extra steps.\n"
                "• PLATFORM LIMITS: Twitter/X = 280 chars max. Reddit titles = 300. Keep text within limits.\n"
                "• EXACT TEXT: When the user provides specific text to post, type it EXACTLY as given. Do NOT rewrite, shorten, or modify it.\n"
                "• After typing text, your NEXT action should be to CLICK the submit/post button. Do not re-type.\n\n"

                "═══ PAGE STRUCTURE (Semantic Markdown) ═══\n"
                "You receive a compressed Markdown map of all interactive elements, grouped by their semantic container (form, nav, header, etc). "
                "Each element has: [eN] id, kind, label, and (x,y) coordinates. "
                "Use the element_id field (e.g., 'e5') to reference elements — coordinates will be resolved automatically. "
                "Use the container grouping to understand which elements belong together.\n\n"

                "═══ AVAILABLE ACTIONS ═══\n"
                "goto — Navigate to a URL (set url field)\n"
                "click — Click element (PREFERRED: set element_id like 'e5'; fallback: set x, y)\n"
                "type — Click then type (PREFERRED: set element_id + text; fallback: x, y + text)\n"
                "scroll — Scroll down to reveal more content\n"
                "press_enter — Press Enter key (e.g., to submit a form)\n"
                "wait — Wait for content to load (set wait_ms, default 1000)\n"
                "done — Goal achieved, stop execution\n"
            )

            # ════════════════════════════════════════════════════════════════
            #  USER PROMPT — Intent + Context + Vision Map
            # ════════════════════════════════════════════════════════════════
            # ── v9.0: Consolidated State Override ──
            state_override = phase_tracker.get_state_override(store.has_article)

            # ── Hard Action Dedup: break repetition loops ──
            dedup_override = ""
            if consecutive_identical_actions >= 2:
                dedup_override = (
                    f"\n\n🛑 REPETITION BLOCK: EXACT SAME action {consecutive_identical_actions + 1} times. "
                    "You MUST choose a COMPLETELY DIFFERENT action NOW."
                )

            user_prompt = (
                f"═══ YOUR MISSION ═══\n"
                f"{llm_objective}\n\n"
                f"NOTE: The above describes the user's INTENT. Any numbered steps are guidance, not strict commands. "
                f"YOU decide the actual actions based on what you see on screen.\n\n"
                f"═══ CURRENT CONTEXT ═══\n"
                f"URL: {current_url}\n"
                f"Step: {step+1}/{max_steps} ({max_steps - step - 1} remaining)"
                f"{login_hint}"
                f"{state_override}"
                f"{dedup_override}\n\n"
                f"═══ ACTION HISTORY (compressed) ═══\n"
                f"{working_mem.compress_history() if working_mem.episodic else '(first step)'}\n\n"
                f"═══ PAGE STRUCTURE ═══\n"
                f"{dom_markdown}\n\n"
                f"Now OBSERVE the page structure above, THINK about your situation, and choose your NEXT action."
                f"{loop_context}"
            )

            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            logger.info("Querying reasoning LLM (failover chain: %d models)...", len(failover_chain))
            try:
                decision, used_model = await _invoke_with_failover(
                    failover_chain,
                    messages,
                    AutonomousAction,
                    circuit_breaker=breaker,
                    health_tracker=health_tracker,
                )
                logger.info("Answered by: %s", used_model)
            except RuntimeError as e:
                wait_secs = min(health_tracker.shortest_quarantine_wait() + 1.0, 15.0) if health_tracker else 5.0
                logger.error("LLM FAILURE: %s — waiting %.1fs for recovery...", e, wait_secs)
                await asyncio.sleep(wait_secs)
                step -= 1  # Don't burn a step when ALL models are down
                continue

            # ── Log the full OTA chain-of-thought ──
            logger.info("👁️ OBSERVE: %s", decision.screen_state[:200])
            logger.info("🔄 PREVIOUS: %s", decision.previous_action_result[:150])
            logger.info("📊 PROGRESS: %s", decision.goal_progress[:150])
            logger.info("🧠 REASONING: %s", decision.reasoning[:200])
            logger.info("⚡ ACTION: %s", decision.action_type)

            # ── Build enriched action history entry ──
            summary = f"Step {step+1}: {decision.action_type}"
            if decision.x is not None and decision.y is not None:
                summary += f" at ({int(decision.x)},{int(decision.y)})"
            if decision.text:
                text_preview = decision.text[:100] + ("..." if len(decision.text) > 100 else "")
                summary += f" text='{text_preview}' ({len(decision.text)} chars)"
            if decision.url:
                summary += f" url='{decision.url[:60]}'"

            # ── v2.0: PRE-ACT CriticV12 snapshot (RQ-B03) ──
            try:
                pre_a11y = dom_data_to_a11y_format(dom_data)
                await progress_critic.snapshot_before(pre_a11y)
            except Exception as snap_err:
                logger.warning("Critic snapshot failed: %s", snap_err)
                try:
                    await progress_critic.snapshot_before(None)
                except Exception:
                    pass

            # ── Execute with outcome tracking ──
            action_outcome = "→ OK"

            # ── Execute ──
            intercepted_by_smart_engine = False
            metrics.total_actions += 1  # v2.1: Track total actions

            # v2.1: MemGPT eviction check at start of each action
            working_mem.evict_if_needed()

            if decision.action_type == "done":
                # v2.1: CoVe pre-done check (RQ-B10) — block premature done
                cove_ok, cove_reason = await cove_pre_done_check(page, plan, working_mem)
                if not cove_ok:
                    logger.warning("🛡️ CoVe BLOCKED premature done: %s", cove_reason)
                    action_history.append(f"Step {step+1}: done BLOCKED — {cove_reason}")
                    working_mem.record_step(step + 1, "done", f"BLOCKED: {cove_reason[:60]}")
                    metrics.done_blocked += 1
                    continue
                logger.info("✅ Goal Achieved! (CoVe verified: %s)", cove_reason[:60])
                break

            elif decision.action_type == "goto":
                # v2.1: Domain safety check (RQ-B09, arXiv 2511.19477)
                if decision.url and not is_domain_allowed(decision.url):
                    logger.warning("🛡️ Domain blocked: %s", decision.url[:80])
                    action_history.append(f"Step {step+1}: goto BLOCKED — domain not in allowlist")
                    working_mem.record_step(step + 1, "goto", f"BLOCKED: domain not allowed")
                    continue
                logger.info("Navigating → %s", decision.url)
                try:
                    await page.goto(decision.url, wait_until="domcontentloaded", timeout=15000)
                except Exception as e:
                    logger.warning("Navigation failed: %s", e)
                    action_outcome = f"→ FAILED: {e}"

            elif decision.action_type in ("type", "click"):
                # v2.1: Check if element_id was provided (preferred) or coordinates (fallback)
                has_element_id = decision.element_id is not None
                has_coords = decision.x is not None and decision.y is not None
                if not has_element_id and not has_coords:
                    # V-04: Record rejected action instead of silent step burn
                    reject_msg = f"Step {step+1}: {decision.action_type} REJECTED — no element_id or coordinates"
                    logger.warning("⚠️ %s", reject_msg)
                    action_history.append(reject_msg)
                    working_mem.record_step(step + 1, decision.action_type, "REJECTED: no target")
                    metrics.grounding_rejects += 1
                    continue

                # ── v2.1: Three-Layer Grounding Validation (RQ-B06) ──
                grounded_el, ground_info = await _ground_or_reject(
                    decision, selector_map, elements_list, page
                )
                if grounded_el is None and isinstance(ground_info, str):
                    reject_msg = f"Step {step+1}: {decision.action_type} GROUNDING REJECT — {ground_info}"
                    logger.warning("🎯 %s", reject_msg)
                    action_history.append(reject_msg)
                    working_mem.record_failure(
                        f"{decision.action_type}({decision.element_id or ''}@{int(decision.x or 0)},{int(decision.y or 0)})",
                        f"grounding: {ground_info}"
                    )
                    working_mem.record_step(step + 1, decision.action_type, f"GROUNDING REJECT: {ground_info[:60]}")
                    metrics.grounding_rejects += 1
                    continue

                # ══════════════════════════════════════════════════════════
                #  v10.0: Smart Content Injection Engine
                # ══════════════════════════════════════════════════════════
                is_article_mode = store.has_article
                intercepted_by_smart_engine = False

                if is_article_mode and not phase_tracker.is_done("all_fields_ready"):
                    # ── Detect which field we're targeting ──
                    form_data = await dom_parser.detect_form_fields(page)
                    title_field = dom_parser.find_field(form_data, "title")
                    body_field = dom_parser.find_field(form_data, "body")

                    target_is_title = False
                    target_is_body = False

                    if title_field and not phase_tracker.is_done("title_filled"):
                        tdist = abs(decision.y - title_field["y"]) + abs(decision.x - title_field["x"])
                        if tdist < 200:
                            target_is_title = True
                    if body_field and not phase_tracker.is_done("body_filled"):
                        bdist = abs(decision.y - body_field["y"]) + abs(decision.x - body_field["x"])
                        if bdist < 200:
                            target_is_body = True

                    # (Removed heuristic fallback that wrongly triggered on Publish button)


                    if target_is_title:
                        intercepted_by_smart_engine = True
                        # ── TITLE: Use ghost_type ──
                        title_text = store.get_title()
                        logger.info("📝 TITLE INJECTION: '%s' (%d chars) at (%d, %d)",
                                    title_text[:60], len(title_text), decision.x, decision.y)
                        try:
                            await ghost_click(page, decision.x, decision.y)
                            await asyncio.sleep(0.3)
                            await page.keyboard.press("Control+A")
                            await asyncio.sleep(0.1)
                            await page.keyboard.press("Delete")
                            await asyncio.sleep(0.2)
                            await page.keyboard.insert_text(title_text)
                            await asyncio.sleep(0.3)

                            audit = await dom_parser.audit_form_fields(page)
                            title_audit = audit.get("title", {})
                            if title_audit.get("found") and title_audit.get("length", 0) > 0:
                                logger.info("✅ TITLE VERIFIED: %d chars — '%s'",
                                            title_audit["length"], title_audit.get("value", "")[:60])
                                phase_tracker.mark("title_filled")
                                phase_tracker.title_content = title_text
                                phase_tracker.typed_text = title_text
                                completed_phases["title_filled"] = True
                                action_outcome = f"→ TITLE OK ({title_audit['length']} chars)"
                            else:
                                logger.warning("⚠️ TITLE UNVERIFIED — field not found after typing")
                                phase_tracker.mark("title_filled")
                                phase_tracker.title_content = title_text
                                completed_phases["title_filled"] = True
                                action_outcome = "→ TITLE typed (unverified)"

                            _record_phase_transition(memory, "unknown", objective, page.url, step, "title_filled")
                        except Exception as e:
                            logger.warning("Title typing failed: %s", e)
                            action_outcome = f"→ FAILED: {e}"

                    elif target_is_body:
                        intercepted_by_smart_engine = True
                        # ── BODY: Direct injection ──
                        body_text = store.get_body()
                        logger.info("📄 BODY INJECTION: %d chars via insert_text at (%d, %d)",
                                    len(body_text), decision.x, decision.y)
                        try:
                            await ghost_click(page, decision.x, decision.y)
                            await asyncio.sleep(0.5)
                            await page.keyboard.press("Control+A")
                            await asyncio.sleep(0.1)
                            await page.keyboard.press("Delete")
                            await asyncio.sleep(0.2)
                            # Direct injection — Instant and rate-limit safe via JS evaluation + input events
                            try:
                                await page.evaluate('''
                                    (text) => {
                                        const el = document.activeElement;
                                        if (!el) return;
                                        if (typeof el.value !== "undefined") {
                                            const nativeSetter = Object.getOwnPropertyDescriptor(
                                                window.HTMLTextAreaElement.prototype, "value"
                                            )?.set || Object.getOwnPropertyDescriptor(
                                                window.HTMLInputElement.prototype, "value"
                                            )?.set;
                                            if (nativeSetter) {
                                                nativeSetter.call(el, text);
                                            } else {
                                                el.value = text;
                                            }
                                            el.dispatchEvent(new Event('input', { bubbles: true }));
                                            el.dispatchEvent(new Event('change', { bubbles: true }));
                                        } else if (el.isContentEditable) {
                                            el.textContent = text;
                                            el.dispatchEvent(new Event('input', { bubbles: true }));
                                        }
                                    }
                                ''', body_text)
                                await asyncio.sleep(0.5)
                            except Exception as js_err:
                                logger.warning("JS direct injection failed (%s), falling back to insert_text", js_err)
                                await page.keyboard.insert_text(body_text)
                            
                            await asyncio.sleep(1.0)

                            audit = await dom_parser.audit_form_fields(page)
                            body_audit = audit.get("body", {})
                            if body_audit.get("found") and body_audit.get("length", 0) > 100:
                                logger.info("✅ BODY VERIFIED: %d chars injected", body_audit["length"])
                                phase_tracker.mark("body_filled")
                                phase_tracker.body_content = body_text
                                completed_phases["body_filled"] = True
                                action_outcome = f"→ BODY OK ({body_audit['length']} chars)"
                            else:
                                logger.warning("⚠️ BODY INJECTION PARTIAL: audit=%d chars", body_audit.get("length", 0))
                                phase_tracker.mark("body_filled")
                                phase_tracker.body_content = body_text
                                completed_phases["body_filled"] = True
                                action_outcome = f"→ BODY injected (audit: {body_audit.get('length', 0)} chars)"

                            _record_phase_transition(memory, "unknown", objective, page.url, step, "body_filled")
                        except Exception as e:
                            logger.warning("Body injection failed: %s", e)
                            action_outcome = f"→ FAILED: {e}"

                    if phase_tracker.is_done("title_filled") and phase_tracker.is_done("body_filled"):
                        if not phase_tracker.is_done("all_fields_ready"):
                            phase_tracker.mark("all_fields_ready")
                            completed_phases["all_fields_ready"] = True
                            logger.info("🎯 ALL ARTICLE FIELDS READY — awaiting submit click")

                if intercepted_by_smart_engine:
                    pass # We handled it above
                elif decision.action_type == "click":
                    logger.info("Clicking at (%d, %d)", decision.x, decision.y)
                    try:
                        # V-14: Timeout-wrapped ghost_click
                        await asyncio.wait_for(
                            ghost_click(page, decision.x, decision.y), timeout=15.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("ghost_click timed out at (%d,%d) — page may be frozen", decision.x, decision.y)
                        action_history.append(f"Step {step+1}: click TIMEOUT at ({decision.x},{decision.y})")
                        continue
                    except Exception as e:
                        logger.warning("Click failed: %s", e)
                        action_outcome = f"→ FAILED: {e}"
                elif decision.action_type == "type":
                    # ══════════════════════════════════════════════════════════
                    #  Standard typing (tweets, short posts, etc.)
                    # ══════════════════════════════════════════════════════════
                    text_to_type = decision.text or ""
                    # V-06: Proper URL-based platform char limits
                    max_chars = get_char_limit(current_url)

                    if len(text_to_type) > max_chars:
                        logger.warning(
                            "⚠️ Text too long (%d chars) for platform limit (%d). Truncating.",
                            len(text_to_type), max_chars,
                        )
                        text_to_type = text_to_type[:max_chars]

                    # ── Content Critic: ONLY for agent-generated content ──
                    objective_lower = objective.lower()
                    words_in_objective = sum(1 for w in text_to_type.split() if w.lower() in objective_lower)
                    text_from_user = words_in_objective >= len(text_to_type.split()) * 0.5

                    if len(text_to_type) > 100 and not text_from_user:
                        logger.info("Content Critic: reviewing %d-char AGENT-GENERATED body...", len(text_to_type))
                        try:
                            verdict = await critic.review(
                                content=text_to_type,
                                platform=current_url,
                                failover_chain=failover_chain,
                                invoke_fn=_invoke_with_failover,
                            )
                            if not verdict.approved and verdict.rewritten_content:
                                rewritten = verdict.rewritten_content
                                if len(rewritten) <= max_chars:
                                    logger.info(
                                        "Critic rewrote content (tone=%.2f, issues=%d): %s",
                                        verdict.tone_score, len(verdict.issues),
                                        "; ".join(verdict.issues[:3]),
                                    )
                                    text_to_type = rewritten
                                else:
                                    logger.warning(
                                        "Critic rewrite (%d chars) exceeds %d limit — using original",
                                        len(rewritten), max_chars,
                                    )
                            elif verdict.approved:
                                logger.info("Critic approved content (tone=%.2f)", verdict.tone_score)
                        except Exception as e:
                            logger.warning("Critic review failed: %s — using original content", e)
                    elif text_from_user:
                        logger.info("📝 User-provided text detected — skipping critic review")

                    logger.info("Typing '%s' (%d chars) at (%d, %d)", text_to_type[:80], len(text_to_type), decision.x, decision.y)
                    try:
                        # Click to focus the field
                        await ghost_click(page, decision.x, decision.y)
                        await asyncio.sleep(0.3)

                        # Clear existing content
                        await page.mouse.click(decision.x, decision.y, click_count=3)
                        await asyncio.sleep(0.15)
                        await page.keyboard.press("Backspace")
                        await asyncio.sleep(0.15)
                        await page.keyboard.press("Control+A")
                        await asyncio.sleep(0.1)
                        await page.keyboard.press("Delete")
                        await asyncio.sleep(0.2)

                        # INSTANT FILL — replace slow ghost_type with insert_text
                        await page.keyboard.insert_text(text_to_type)
                        await asyncio.sleep(0.3)

                        # ── Post-Type Verification ──
                        try:
                            audit = await dom_parser.audit_form_fields(page)
                            # Check title first, then body, then generic
                            t = audit.get("title", {})
                            b = audit.get("body", {})
                            verified_len = max(t.get("length", 0), b.get("length", 0))
                            if verified_len > 0:
                                logger.info("✅ TYPE VERIFIED: %d chars in field", verified_len)
                                phase_tracker.mark("title_filled")
                                phase_tracker.typed_text = text_to_type
                                completed_phases["title_filled"] = True
                                typed_text_content = text_to_type
                                action_outcome = f"→ OK (verified: {verified_len} chars)"
                                _record_phase_transition(memory, "unknown", objective, page.url, step, "title_filled")
                            else:
                                # Fallback: try the old selector method
                                verify_result = await asyncio.wait_for(page.evaluate("""
                                () => {
                                    const el = document.querySelector('[data-testid="tweetTextarea_0"], [contenteditable="true"], textarea:focus, input:focus');
                                    if (el) {
                                        const text = el.textContent || el.value || '';
                                        return { found: true, length: text.length, preview: text.slice(0, 100) };
                                    }
                                    return { found: false, length: 0, preview: '' };
                                }
                                """), timeout=3.0)
                                if verify_result.get("found") and verify_result.get("length", 0) > 0:
                                    logger.info("✅ TYPE VERIFIED (fallback): %d chars — '%s'",
                                                verify_result["length"], verify_result["preview"][:60])
                                    phase_tracker.mark("title_filled")
                                    phase_tracker.typed_text = text_to_type
                                    completed_phases["title_filled"] = True
                                    typed_text_content = text_to_type
                                    action_outcome = f"→ OK (verified: {verify_result['length']} chars)"
                                else:
                                    logger.warning("⚠️ TYPE UNVERIFIED: field not found or empty after typing")
                                    action_outcome = "→ PARTIAL (field not verified)"
                        except Exception as verify_err:
                            logger.warning("Type verification failed: %s", verify_err)
                            phase_tracker.mark("title_filled")
                            phase_tracker.typed_text = text_to_type
                            completed_phases["title_filled"] = True
                            typed_text_content = text_to_type
                    except Exception as e:
                        logger.warning("Type failed: %s", e)
                        action_outcome = f"→ FAILED: {e}"


            elif decision.action_type == "press_enter":
                logger.info("Pressing Enter")
                try:
                    await page.keyboard.press("Enter")
                except Exception as e:
                    logger.warning("Press Enter failed: %s", e)
                    action_outcome = f"→ FAILED: {e}"

            elif decision.action_type == "wait":
                wait_ms = decision.wait_ms or 800
                logger.info("Waiting %dms", wait_ms)
                try:
                    await page.wait_for_timeout(wait_ms)
                except Exception as e:
                    logger.warning("Wait failed: %s", e)
                    action_outcome = f"→ FAILED: {e}"

            elif decision.action_type == "scroll":
                logger.info("Scrolling down")
                try:
                    # V-14: Timeout-wrapped ghost_scroll
                    await asyncio.wait_for(
                        ghost_scroll(page, 600), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("ghost_scroll timed out — page may be frozen")
                    action_history.append(f"Step {step+1}: scroll TIMEOUT")
                    continue
                except Exception as e:
                    logger.warning("Scroll failed: %s", e)
                    action_outcome = f"→ FAILED: {e}"

            # Record action with outcome + screen context
            history_entry = f"{summary} {action_outcome}"
            if decision.screen_state:
                history_entry += f" | Screen: {decision.screen_state[:80]}"
            action_history.append(history_entry)

            # ── v2.0: WorkingMemory episodic recording ──
            working_mem.record_step(
                step + 1,
                f"{decision.action_type}" + (f"({int(decision.x or 0)},{int(decision.y or 0)})" if decision.x else ""),
                action_outcome,
                screen_hint=decision.screen_state[:80] if decision.screen_state else "",
            )
            # Update spatial memory with current page elements
            if elements_list:
                working_mem.update_page_map(current_url, elements_list)

            # ── v2.1: CriticV12 POST-ACT Verification (RQ-B03) + Reflexion (RQ-B02) + Key-Node (RQ-B05) ──
            if decision.action_type in ("click", "type", "goto", "scroll"):
                try:
                    post_dom = await dom_parser.extract(page, timeout=3.0)
                    post_a11y = dom_data_to_a11y_format(post_dom)
                except Exception:
                    post_a11y = None
                # Build target_ref from element_id if available, else from coordinates
                target_ref = decision.element_id or f"({int(decision.x or 0)},{int(decision.y or 0)})"
                verdict = await progress_critic.evaluate(
                    action=decision.action_type,
                    target_ref=target_ref,
                    target_name=decision.text[:30] if decision.text else "",
                    a11y_data_after=post_a11y,
                )
                if verdict.success:
                    correction_failures = 0
                    phase_tracker.correction_context = ""
                    metrics.critic_progress += 1
                    logger.info("✓ Critic: %s [%.0f%%]", verdict.reason[:100], verdict.confidence * 100)
                    # v2.1: Key-node advancement (RQ-B05)
                    if not verdict.circuit_breaker_triggered:
                        working_mem.note(f"step{step+1}_progress", verdict.reason[:60])
                        cur_step = plan._current()
                        if cur_step and any(kw in verdict.reason.lower() for kw in
                                            ["url changed", "structure changed", "state changed",
                                             "new elements", "elements disappeared"]):
                            plan.advance()
                            working_mem.note(
                                f"keynode_{step+1}",
                                f"plan step '{cur_step['desc'][:30]}' completed"
                            )
                            logger.info(
                                "🎯 Key-node: plan step '%s' completed (progress=%d%%)",
                                cur_step['desc'][:40], plan.progress_pct,
                            )
                    # v2.1: Dynamic budget extension (RQ-B05)
                    if step >= max_steps - 3 and plan.progress_pct >= 60:
                        max_steps += 5
                        logger.info(
                            "📊 Budget extended to %d (progress=%d%%)",
                            max_steps, plan.progress_pct,
                        )
                else:
                    correction_failures += 1
                    metrics.critic_no_progress += 1
                    phase_tracker.failed_actions.append(
                        f"Step {step+1}: {decision.action_type} → {verdict.reason[:60]}"
                    )
                    working_mem.record_failure(
                        f"{decision.action_type}({target_ref})",
                        verdict.reason[:80],
                    )
                    if verdict.circuit_breaker_triggered:
                        phase_tracker.correction_context = (
                            f"\n\n🛑 CRITIC CIRCUIT BREAKER: {verdict.reason} "
                            "You MUST try a COMPLETELY DIFFERENT approach: scroll, dismiss overlays, wait, or navigate elsewhere."
                        )
                        logger.warning("🛑 Critic circuit breaker: %s", verdict.reason[:100])
                        progress_critic.reset_circuit_breaker()
                    elif correction_failures >= 2:
                        phase_tracker.correction_context = (
                            f"\n\n⚠️ SELF-CORRECTION: Last {correction_failures} actions had NO visible effect. "
                            f"Failed: {'; '.join(phase_tracker.failed_actions[-3:])}. "
                            "Try a COMPLETELY DIFFERENT approach: scroll, dismiss overlays, or wait."
                        )
                        logger.warning("🧠 Critic: no progress (streak=%d)", correction_failures)

                        # v2.1: Gated Reflexion (RQ-B02) — fires ONLY on stagnation streak ≥ 2
                        try:
                            metrics.reflexion_triggers += 1
                            reflect_prompt = (
                                f"I attempted {correction_failures} actions with no progress. "
                                f"Recent failures: {'; '.join(phase_tracker.failed_actions[-3:])}. "
                                f"Current plan step: {plan._current()['desc'] if plan._current() else 'unknown'}. "
                                f"What went wrong? What should I try differently?"
                            )
                            reflection_msgs = [
                                SystemMessage(content=(
                                    "You are a self-reflection agent analyzing a web automation failure. "
                                    "Be specific about what went wrong and suggest a concrete alternative approach."
                                )),
                                HumanMessage(content=reflect_prompt),
                            ]
                            reflection_result, _ = await _invoke_with_failover(
                                failover_chain, reflection_msgs, None,
                                breaker, health_tracker=health_tracker,
                            )
                            if hasattr(reflection_result, 'content') and reflection_result.content:
                                reflection_text = reflection_result.content[:200]
                                plan.add_reflection(reflection_text)
                                working_mem.note("last_reflection", reflection_text)
                                logger.info("🪞 Reflexion: %s", reflection_text[:150])
                        except Exception as ref_err:
                            logger.warning("Reflexion call failed: %s", ref_err)

            # ── v9.0: Phase Transitions + Continuous Campaign Logging ──
            # ── v13.0: Platform detection via URL parsing (V-06) ──
            platform = detect_platform(page.url) or "unknown"

            if decision.action_type == "goto" and not phase_tracker.is_done("navigated_to_target"):
                phase_tracker.mark("navigated_to_target")
                _record_phase_transition(memory, platform, objective, page.url, step, "navigated")

            # ── v13.0: Submit detection gated on FormState (V-05/V-09) ──
            is_submit_click = False
            if decision.action_type == "click" and not intercepted_by_smart_engine:
                is_submit_click = form_state.ready_to_submit

            if is_submit_click:
                phase_tracker.mark("submit_clicked")
                completed_phases["submit_clicked"] = True
                logger.info("📤 Post button click detected")
                _record_phase_transition(memory, platform, objective, page.url, step, "submit_clicked")

                # ── Mission Completion: verify and exit gracefully ──
                await asyncio.sleep(3.0)  # Let the platform process the post
                try:
                    post_url = page.url

                    # CRITICAL FIX: Handle /content_disclosure popup on X
                    # This popup is NOT success — it's a blocker that needs dismissal
                    if "content_disclosure" in post_url:
                        logger.warning("⚠️ Content disclosure popup detected — attempting to dismiss")
                        try:
                            # Look for dismiss/continue/post buttons in the popup
                            dismiss_btn = await page.query_selector(
                                'button:has-text("Post"), button:has-text("Continue"), '
                                'button:has-text("Got it"), button:has-text("Dismiss"), '
                                '[data-testid="confirmationSheetConfirm"]'
                            )
                            if dismiss_btn:
                                await dismiss_btn.click()
                                logger.info("✅ Disclosure popup dismissed")
                                await asyncio.sleep(3.0)
                                post_url = page.url  # Re-check URL after dismissal
                            else:
                                logger.warning("No dismiss button found in disclosure popup")
                        except Exception as disc_err:
                            logger.warning("Disclosure dismissal failed: %s", disc_err)

                    # Check if compose box cleared (post went through) or URL changed
                    compose_empty = await asyncio.wait_for(page.evaluate("""
                    () => {
                        const href = window.location.href;
                        // On X, the URL stays /compose/post but the box disappears
                        if (href.includes('twitter.com') || href.includes('x.com')) {
                            const box = document.querySelector('[data-testid="tweetTextarea_0"]');
                            return !box || (box.textContent || '').trim().length < 5;
                        }
                        // On Reddit, successful post ALWAYS navigates away from /submit
                        if (href.includes('reddit.com/submit') || href.includes('reddit.com/user/me/submit')) {
                            return false;
                        }
                        // Generic fallback: check if we navigated away from a submit-like URL
                        if (href.includes('/submit') || href.includes('/compose')) {
                            return false;
                        }
                        return true;
                    }
                    """), timeout=3.0)

                    # Only consider it a real success if we're NOT still on a blocker URL
                    still_blocked = "content_disclosure" in page.url or "compose" in page.url
                    real_success = compose_empty and not still_blocked

                except Exception:
                    real_success = False  # Don't assume success on errors
                    still_blocked = False

                if real_success:
                    phase_tracker.mark("submit_verified")
                    _record_phase_transition(memory, platform, objective, page.url, step, "submit_verified")
                    completed_phases["mission_success"] = True

                    # v10.0: Clean up content store after successful post
                    if store.has_article:
                        store.mark_posted()

                    # ── SUCCESS BANNER ──
                    total_phases = len(phase_tracker.phases)
                    done_phases = sum(1 for v in phase_tracker.phases.values() if v)
                    logger.info("")
                    logger.info("━" * 60)
                    logger.info("🎯  MISSION ACCOMPLISHED")
                    logger.info("━" * 60)
                    logger.info("  Platform:  %s", platform)
                    logger.info("  URL:       %s", post_url)
                    logger.info("  Steps:     %d", step + 1)
                    logger.info("  Phases:    %d/%d (%d%%)", done_phases, total_phases, phase_tracker.progress_pct())
                    logger.info("  Objective: %s", objective[:80])
                    logger.info("━" * 60)
                    logger.info("")
                    break  # Exit the OTA loop — mission complete

            # ── Progressive Perception: Track action dedup ──
            current_action_signature = f"{decision.action_type}:{decision.text[:30] if decision.text else 'none'}"
            if current_action_signature == last_action_signature:
                consecutive_identical_actions += 1
                if consecutive_identical_actions >= 3:
                    logger.warning(
                        "🛑 STUCK DETECTED: '%s' repeated %d times — calling Recovery Advisor",
                        decision.action_type, consecutive_identical_actions + 1,
                    )
                    # v9.0: Ask Recovery Advisor instead of just logging a warning
                    advice = await _ask_recovery_advisor(
                        objective, page.url, action_history,
                        failover_chain, breaker, health_tracker,
                    )
                    phase_tracker.recovery_advice = advice
                    consecutive_identical_actions = 0  # Reset after getting advice
            else:
                consecutive_identical_actions = 0
                phase_tracker.recovery_advice = ""  # Clear old advice on new action

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
        # ── Structured Crash Context ──
        try:
            crash_context = {
                "step": step + 1,
                "url": page.url if page and not page.is_closed() else "unknown",
                "last_action": action_history[-1] if action_history else "none",
                "breaker_status": breaker.status_line(),
            }
            logger.error("💥 CRASH CONTEXT: %s", json.dumps(crash_context))
        except Exception:
            pass
        logger.error("Unexpected crash: %s", e)
        logger.error(traceback.format_exc())

    finally:
        # ════════════════════════════════════════════════════════════════════
        #  CLEAN SHUTDOWN — context.close() flushes all native state to disk
        # ════════════════════════════════════════════════════════════════════
        mission_success = completed_phases.get("mission_success", False)
        if mission_success:
            logger.info("✅ Agent shutting down after successful mission.")
        else:
            logger.info("⚠️  Agent shutting down (mission incomplete — ran out of steps or was interrupted).")
        try:
            await context.close()
            logger.info("Browser closed. Native profile state persisted.")
        except Exception as e:
            logger.warning("Browser close failed: %s", e)
        guard.detach()

        if mission_success:
            print("\n" + "═" * 60)
            print("  ✅  MISSION COMPLETE — Post published successfully!")
            print("═" * 60 + "\n")


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
