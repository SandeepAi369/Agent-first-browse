"""Advanced Single-Agent Browser Automation with Deep Intelligence (v14.0).

v14.0 General-Purpose Architecture:
  - ModelRegistry: ALL API keys, strict TEXT/VISION separation, singleton
  - Mission Planner: pre-loop task decomposition into checkpoints
  - CriticV12: 6-signal progress detection
  - Action Outcome Verifier: post-action page-change detection
  - Self-Correction Engine: auto-recover from silently failing actions
  - Recovery Advisor: separate LLM call when stuck
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
from ghost_input import ghost_click, ghost_type, ghost_scroll, ghost_move_to
from campaign_memory import CampaignMemory
from model_registry import ModelRegistry
from cdp_input import resilient_type
from overlay_detector import smart_click_with_penetration, check_click_target
from cdp_click import resilient_click, ClickResult
from action_verifier import ActionVerifier, VerificationResult
from action_classifier import classify_action, ActionRisk, requires_simulation
from web_dreamer import WebDreamer, should_invoke_dreamer, DreamerResult, CandidateAction
from prm_critic import PRMCritic, ChecklistItem, StepScore
from skill_memory import SkillMemory
import dom_parser

# Cognitive Architecture V2.1
from cognitive_core import PlanState, WorkingMemory, dom_data_to_a11y_format, AgentMetrics
from orchestrator.critic_v12 import CriticV12, Verdict
from execution_safety import is_domain_allowed, cove_pre_done_check, auto_allow_from_objective

logger = get_logger("advanced_agent")





# ═══════════════════════════════════════════════════════════════════════════════
#  v14.0 Task Progress Tracker — Neutral, general-purpose
# ═══════════════════════════════════════════════════════════════════════════════

class TaskProgressTracker:
    """Track task progress, failed actions, and correction context.

    This is a general-purpose tracker — no assumptions about task type.
    """

    def __init__(self):
        self.failed_actions: list[str] = []
        self.correction_context = ""
        self.recovery_advice = ""
        self.mission_success = False

    def get_state_override(self) -> str:
        """Build contextual state info for the LLM based on progress."""
        parts = []
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
        f"RECENT ACTIONS:\n{json.dumps(list(action_history)[-6:], indent=1)}\n\n"
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


# (v14.0: Campaign logger removed — task history recorded at end of run only)

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
    """Core autonomous agent loop — v14.0 General-Purpose Architecture.

    Features: ModelRegistry, Mission Planner, CriticV12,
    Action Verifier, Self-Correction Engine, Recovery Advisor,
    Enhanced Failover.
    """
    context, page = await launch_browser()
    guard = SessionGuard.get()

    # ── Human Warm-Up Routine ──
    target_hint = extract_target_url_from_objective(objective)
    try:
        await run_warmup(page, target_url=target_hint)
    except Exception as warmup_err:
        logger.warning("Warm-up routine failed (non-fatal): %s", warmup_err)

    # ── v14.0: ModelRegistry as single source of truth ──
    registry = ModelRegistry.get_instance()
    memory = CampaignMemory()  # Neutral task history recorder
    breaker = registry.breaker  # Shared circuit breaker
    health_tracker = registry.health  # Shared health tracker

    # v14.0: Auto-allow domains mentioned in the objective
    auto_allow_from_objective(objective)

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

    # v14.0: Pass objective through to the LLM verbatim — no ContentStore rewriting
    llm_objective = objective

    # ── v2.0: Mission Planner → PlanState (fixes B-01 amnesia, B-10 dead checkpoints) ──
    mission_checkpoints = await _decompose_objective(llm_objective, failover_chain, breaker, health_tracker)
    plan = PlanState.from_checkpoints(llm_objective, mission_checkpoints)
    working_mem = WorkingMemory()
    metrics = AgentMetrics()  # v2.1: Quality metrics tracking (S-5)
    logger.info("📋 Mission Plan (%d steps):", len(plan.steps))
    for s in plan.steps:
        logger.info("  [%d] %s", s["id"] + 1, s["desc"])

    # ── Phase 2: PRM Checklist Generation ──
    prm_critic: PRMCritic | None = None
    prm_checklist: list[ChecklistItem] = []
    try:
        prm_critic = PRMCritic(_invoke_with_failover, failover_chain, breaker, health_tracker)
        prm_checklist = await prm_critic.generate_checklist(llm_objective)
        logger.info("📋 PRM Checklist (%d items) generated", len(prm_checklist))
    except Exception as e:
        logger.warning("PRM checklist generation failed (non-fatal): %s", e)

    # ── Phase 1: WebDreamer Initialization ──
    dreamer: WebDreamer | None = None
    try:
        dreamer = WebDreamer(
            invoke_fn=_invoke_with_failover,
            failover_chain=failover_chain,
            breaker=breaker,
            health_tracker=health_tracker,
            num_candidates=3,
            num_simulations=1,
        )
        logger.info("🌙 WebDreamer planning engine initialized (k=3, H=1)")
    except Exception as e:
        logger.warning("WebDreamer init failed (non-fatal): %s", e)

    # ── Phase 3: Skill Memory — Retrieve relevant workflows ──
    skill_mem: SkillMemory | None = None
    skill_context = ""
    try:
        skill_mem = SkillMemory()
        auto_domain = ""
        for url_part in [llm_objective]:
            import re as _re
            url_match = _re.search(r'https?://(?:www\.)?([\w.-]+)', url_part)
            if url_match:
                auto_domain = url_match.group(1)
                break
        relevant_workflows = skill_mem.retrieve_relevant(llm_objective, domain=auto_domain)
        skill_context = skill_mem.inject_into_prompt(relevant_workflows)
        if skill_context:
            logger.info("🧠 SkillMemory: %d relevant workflows injected into prompt", len(relevant_workflows))
        stats = skill_mem.get_stats()
        logger.info("🧠 SkillMemory: %d total workflows, %d reliable, %d domains",
                    stats['total_workflows'], stats['reliable_workflows'], stats['domains_covered'])
    except Exception as e:
        logger.warning("SkillMemory init failed (non-fatal): %s", e)

    action_history: deque[str] = deque(maxlen=50)    # Reduced — WorkingMemory handles the rest
    url_history: deque[str] = deque(maxlen=50)         # V-01: bounded ring buffer
    last_action_signature = ""
    consecutive_identical_actions = 0
    correction_failures = 0  # Self-correction streak counter
    same_url_streak = 0  # Phase 1: How many steps on same URL
    last_url_for_streak = ""  # Phase 1: Track URL changes
    # v14.0: Flat step budget — no publishing bonus
    max_steps = 25
    step = 0

    # ── v14.0: General-purpose progress tracker (no publishing phases) ──
    progress_tracker = TaskProgressTracker()

    # ── v13.0: Triple-state loop detector (RQ-12) ──
    _loop_window: deque[tuple] = deque(maxlen=6)

    # ── v2.0: CriticV12 (RQ-B03) — replaces dead hash verifier ──
    progress_critic: CriticV12 | None = None  # initialized after page is ready

    # ── Phase 0: Action Verifier for post-action DOM diff + React recheck ──
    action_verifier: ActionVerifier | None = None

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

            # ── Phase 0: Initialize ActionVerifier ──
            if action_verifier is None:
                action_verifier = ActionVerifier(page)
                logger.info("🔍 ActionVerifier initialized (DOM diff + React recheck)")

            # ── Anti-Loop Watchdog v13: (url, dom_fp, action_sig) triple ──
            url_history.append(current_url)
            loop_context = ""
            if len(url_history) >= 6:
                recent_urls = list(url_history)[-6:]
                url_counts = Counter(recent_urls)
                most_common_url, visit_count = url_counts.most_common(1)[0]
                if visit_count >= 3:
                    # V15.0 F7: Smart loop detection — check action DIVERSITY before flagging
                    # On form pages, agent legitimately stays on same URL for 4+ steps
                    # (type title → type body → click submit). That's NOT a loop.
                    FORM_URL_PATTERNS = ('/submit', '/compose', '/new', '/create',
                                         '/checkout', '/form', '/editor', '/settings',
                                         '/cart', '/review')
                    is_form_page = any(p in most_common_url for p in FORM_URL_PATTERNS)

                    # Check action diversity: same action on same target = loop
                    # Different actions on different targets = form fill (OK)
                    recent_actions_strs = list(action_history)[-6:] if len(action_history) >= 6 else list(action_history)
                    unique_action_targets = set()
                    for entry in recent_actions_strs:
                        # Extract "type at (x,y)" or "click at (x,y)" pattern
                        import re as _re
                        _match = _re.search(r'(type|click|scroll)\s+at\s+\((\d+),(\d+)\)', entry)
                        if _match:
                            unique_action_targets.add(f"{_match.group(1)}_{_match.group(2)}_{_match.group(3)}")

                    has_action_diversity = len(unique_action_targets) >= 2

                    if is_form_page and has_action_diversity:
                        # Agent is filling a form — different actions on different fields. Not a loop.
                        logger.debug(
                            "🔄 Same URL '%s' %d times but form page with diverse actions — not a loop",
                            most_common_url[:60], visit_count,
                        )
                    else:
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

            # v14.0: No target_hint bias — DOM parser extracts ALL interactive elements neutrally
            target_hint = None

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
                    // Generic profile/auth indicators
                    const hasProfile = !!document.querySelector(
                        '[aria-label*="profile" i], [aria-label*="account" i], '
                        + 'img[alt*="avatar" i], img[alt*="profile" i], '
                        + '[data-testid*="profile" i], [data-testid*="user" i], '
                        + '.user-menu, .profile-menu, #user-nav'
                    );
                    // Generic login form indicators (valid CSS only — no :has-text)
                    let hasLoginForm = !!document.querySelector(
                        'input[type="password"], '
                        + 'form[action*="login" i], form[action*="signin" i]'
                    );
                    // Check for sign-in/log-in buttons via textContent (CSS can't do this)
                    if (!hasLoginForm) {
                        const buttons = document.querySelectorAll('button, a[role="button"], input[type="submit"]');
                        for (const btn of buttons) {
                            const txt = (btn.textContent || '').trim().toLowerCase();
                            if (txt === 'sign in' || txt === 'log in' || txt === 'login' || txt === 'signin') {
                                hasLoginForm = true;
                                break;
                            }
                        }
                    }
                    return {
                        url: url,
                        hasProfile: hasProfile,
                        hasLoginForm: hasLoginForm
                    };
                }
                """), timeout=5.0)

                if login_state.get("hasProfile"):
                    login_hint = (
                        "\n\n🔑 LOGIN STATUS: You appear to be LOGGED IN. "
                        "Profile/account indicators detected. "
                        "DO NOT attempt to log in again. Proceed directly to your goal."
                    )
                    logger.info("🔑 Login detected: profile=%s", login_state.get("hasProfile"))
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
                "You are an autonomous browser automation agent. "
                "You operate inside a real Chromium browser and control it through actions.\n\n"

                "═══ YOUR CURRENT PLAN ═══\n"
                f"{plan_context}\n\n"

                + (f"═══ WHAT YOU KNOW ═══\n{facts_context}\n\n" if facts_context else "")

                + "═══ CORE RULES ═══\n"
                "1. READ the user's objective carefully. Do EXACTLY what they ask — "
                "nothing more, nothing less. NEVER invent tasks the user didn't ask for.\n"
                "2. OBSERVE the page before acting. Describe what you see.\n"
                "3. THINK about which action brings you closer to the goal. "
                "Consider at least 2 possible actions and explain why you chose one.\n"
                "4. ACT with one precise action per turn.\n"
                "5. If a popup, overlay, modal, or cookie banner blocks you, DISMISS IT FIRST.\n"
                "6. If an action FAILED (visible in action history), do NOT repeat it. Try an alternative approach.\n"
                "7. SCROLL if you can't find the target element — it may be below the fold.\n"
                "8. Use 'wait' (1000-2000ms) after clicks that trigger page loads.\n"
                "9. When the goal is FULLY achieved, output action_type='done' immediately.\n"
                "10. When the user provides specific text to type, type it EXACTLY as given.\n\n"

                "═══ OBSERVE-THINK-ACT FORMAT ═══\n"
                "Every turn you MUST:\n"
                "1. OBSERVE: Describe what you see on screen (screen_state).\n"
                "   - What kind of page is this? What are the key visible elements?\n"
                "   - Are there any blocking elements (popups, overlays)?\n"
                "2. THINK: Evaluate your last action, assess progress, reason about next step.\n"
                "   - Did your last action work? How do you know?\n"
                "   - What fraction of the goal is done? What remains?\n"
                "3. ACT: Execute exactly ONE action.\n\n"

                "═══ PAGE STRUCTURE (Semantic Markdown) ═══\n"
                "You receive a semantic map of all interactive elements. "
                "Each element has: [eN] id, kind, label, and (x,y) coordinates. "
                "Use the element_id field (e.g., 'e5') to reference elements — coordinates are resolved automatically.\n\n"

                "═══ AVAILABLE ACTIONS ═══\n"
                "goto — Navigate to a URL (set url field)\n"
                "click — Click element (PREFERRED: set element_id like 'e5'; fallback: set x, y)\n"
                "type — Click then type (PREFERRED: set element_id + text; fallback: x, y + text)\n"
                "scroll — Scroll down to reveal more content\n"
                "press_enter — Press Enter key (e.g., to submit a search or form)\n"
                "wait — Wait for content to load (set wait_ms, default 1000)\n"
                "done — Goal achieved, stop execution\n"
            )

            # ── Phase 3: Inject learned workflows into system prompt ──
            if skill_context:
                system_prompt += f"\n\n{skill_context}"

            # ════════════════════════════════════════════════════════════════
            #  USER PROMPT — Intent + Context + Vision Map
            # ════════════════════════════════════════════════════════════════
            # ── v9.0: Consolidated State Override ──
            # ── v14.0: Neutral progress state ──
            state_override = progress_tracker.get_state_override()

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

            # ── Phase 1: Track same-URL streak for WebDreamer gate ──
            if current_url == last_url_for_streak:
                same_url_streak += 1
            else:
                same_url_streak = 0
                last_url_for_streak = current_url
                if dreamer:
                    dreamer.clear_cache()  # Clear sim cache on navigation

            # ── Phase 0: Action Risk Classification ──
            target_name = ""
            if decision.element_id and decision.element_id in selector_map:
                el_data = selector_map[decision.element_id]
                target_name = el_data.get("name", el_data.get("text", ""))[:60]
            action_risk = classify_action(
                action_type=decision.action_type,
                target_name=target_name,
                target_text=decision.text[:60] if decision.text else "",
                url=current_url,
                element_kind=selector_map.get(decision.element_id, {}).get("kind", "") if decision.element_id else "",
            )
            if action_risk != ActionRisk.REVERSIBLE:
                logger.info("⚠️ ACTION RISK: %s", action_risk.name)

            # ── Phase 1: WebDreamer Simulate-Before-Acting Gate ──
            if dreamer and decision.action_type not in ("done", "wait"):
                invoke_dreamer = should_invoke_dreamer(
                    element_count=dom_data.get("element_count", 0),
                    action_risk_level=action_risk.name,
                    consecutive_no_progress=correction_failures,
                    step_number=step,
                    same_url_streak=same_url_streak,
                )
                if invoke_dreamer:
                    try:
                        logger.info("🌙 WebDreamer: Simulating %d candidates before acting...", 3)
                        proposed_candidate = CandidateAction(
                            action_type=decision.action_type,
                            element_id=decision.element_id,
                            text=decision.text,
                            url=decision.url,
                            x=decision.x,
                            y=decision.y,
                            reasoning=decision.reasoning,
                        )
                        dreamer_result = await asyncio.wait_for(
                            dreamer.plan_and_select(
                                dom_markdown=dom_markdown,
                                objective=llm_objective,
                                plan_context=plan.render(),
                                action_history=working_mem.compress_history(),
                                current_url=current_url,
                                proposed_action=proposed_candidate,
                            ),
                            timeout=60.0,
                        )

                        # Override decision if WebDreamer found a better action
                        best = dreamer_result.best_action
                        if dreamer_result.best_score >= 0.4:
                            # Map WebDreamer candidate back to AutonomousAction fields
                            if best.action_type != decision.action_type or best.element_id != decision.element_id:
                                logger.info(
                                    "🌙 WebDreamer OVERRIDE: %s (score=%.2f) replaces %s",
                                    best.describe(), dreamer_result.best_score,
                                    decision.action_type,
                                )
                                decision.action_type = best.action_type
                                if best.element_id:
                                    decision.element_id = best.element_id
                                    # Re-resolve coordinates from selector_map
                                    if best.element_id in selector_map:
                                        coords = selector_map[best.element_id]
                                        decision.x = coords.get("x", decision.x)
                                        decision.y = coords.get("y", decision.y)
                                if best.text:
                                    decision.text = best.text
                                if best.url:
                                    decision.url = best.url
                            else:
                                logger.info(
                                    "🌙 WebDreamer CONFIRMS: %s (score=%.2f)",
                                    best.describe(), dreamer_result.best_score,
                                )
                        else:
                            logger.warning(
                                "🌙 WebDreamer: All candidates scored low (best=%.2f). Proceeding with original.",
                                dreamer_result.best_score,
                            )
                    except asyncio.TimeoutError:
                        logger.warning("🌙 WebDreamer timed out — proceeding with original decision")
                    except Exception as dreamer_err:
                        logger.warning("🌙 WebDreamer error (non-fatal): %s", dreamer_err)

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
            metrics.total_actions += 1  # v2.1: Track total actions

            # v2.1: MemGPT eviction check at start of each action
            working_mem.evict_if_needed()

            if decision.action_type == "done":
                # v2.1: CoVe pre-done check (RQ-B10) — block premature done
                cove_ok, cove_reason = await cove_pre_done_check(page, plan, working_mem, objective)
                if not cove_ok:
                    logger.warning("🛡️ CoVe BLOCKED premature done: %s", cove_reason)
                    action_history.append(f"Step {step+1}: done BLOCKED — {cove_reason}")
                    working_mem.record_step(step + 1, "done", f"BLOCKED: {cove_reason[:60]}")
                    metrics.done_blocked += 1
                    continue
                logger.info("✅ Goal Achieved! (CoVe verified: %s)", cove_reason[:60])
                progress_tracker.mission_success = True
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

                # V15.2: Scroll target element into viewport center before action
                if decision.element_id:
                    try:
                        await page.evaluate("""(eid) => {
                            const el = document.querySelector(`[data-eid="${eid}"]`);
                            if (el) el.scrollIntoView({behavior: 'smooth', block: 'center'});
                        }""", decision.element_id)
                        await asyncio.sleep(0.5)  # Let scroll settle
                    except Exception as e:
                        logger.debug("Failed to scroll element %s into view: %s", decision.element_id, e)

                # v14.0: No content injection engine — LLM decides what to type/click

                if decision.action_type == "click":
                    logger.info("Clicking at (%d, %d)", decision.x, decision.y)
                    try:
                        # UCRF: Pre-flight overlay detection before click
                        penetration = await smart_click_with_penetration(
                            page, decision.x, decision.y
                        )
                        if penetration.get("overlay_bypassed"):
                            logger.info(
                                "🎯 Overlay penetrated at (%d, %d) via %s",
                                decision.x, decision.y, penetration.get("method", "?")
                            )
                            # Small delay to let pointer-events disable take effect
                            await asyncio.sleep(0.1)

                        # Phase 0: Humanized mouse movement + CDP native click
                        # Step 1: Move mouse along Bézier curve to target
                        await ghost_move_to(page, decision.x, decision.y)

                        # Step 2: Dispatch click via CDP resilient waterfall
                        click_result = await asyncio.wait_for(
                            resilient_click(
                                page, decision.x, decision.y,
                                max_retries=4, settle_ms=800,
                            ),
                            timeout=30.0,
                        )
                        if click_result.success:
                            action_outcome = (
                                f"→ OK (click via {click_result.strategy}"
                                f"{', navigated' if click_result.navigation else ''}"
                                f"{', DOM changed' if click_result.dom_changed else ''})"
                            )
                        else:
                            action_outcome = f"→ CLICK INEFFECTIVE: {click_result.error[:80]}"
                            logger.warning("Click had no effect after %d strategies", click_result.attempts)

                    except asyncio.TimeoutError:
                        logger.warning("resilient_click timed out at (%d,%d) — page may be frozen", decision.x, decision.y)
                        action_history.append(f"Step {step+1}: click TIMEOUT at ({decision.x},{decision.y})")
                        continue
                    except Exception as e:
                        logger.warning("Click failed: %s", e)
                        action_outcome = f"→ FAILED: {e}"
                elif decision.action_type == "type":
                    # ══════════════════════════════════════════════════════════
                    #  UCRF: Multi-strategy CDP typing with waterfall & verify
                    # ══════════════════════════════════════════════════════════
                    text_to_type = decision.text or ""

                    logger.info("Typing '%s' (%d chars) at (%d, %d)", text_to_type[:80], len(text_to_type), decision.x, decision.y)
                    try:
                        type_result = await asyncio.wait_for(
                            resilient_type(
                                page,
                                text_to_type,
                                x=decision.x,
                                y=decision.y,
                                clear_first=True,
                                max_retries=3,
                            ),
                            timeout=60.0,  # Allow up to 60s for long text with retries
                        )
                        if type_result["success"]:
                            logger.info(
                                "✅ TYPE VERIFIED via %s: %d chars (attempt %d)",
                                type_result["strategy"],
                                type_result["actual_length"],
                                type_result["attempts"],
                            )
                            action_outcome = f"→ OK (verified: {type_result['actual_length']} chars via {type_result['strategy']})"
                        else:
                            logger.warning("⚠️ TYPE UNVERIFIED: all strategies exhausted")
                            action_outcome = "→ PARTIAL (type unverified after all strategies)"
                    except asyncio.TimeoutError:
                        logger.warning("Type timed out after 60s")
                        action_outcome = "→ TIMEOUT"
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
                    progress_tracker.correction_context = ""
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
                    progress_tracker.failed_actions.append(
                        f"Step {step+1}: {decision.action_type} → {verdict.reason[:60]}"
                    )
                    working_mem.record_failure(
                        f"{decision.action_type}({target_ref})",
                        verdict.reason[:80],
                    )
                    if verdict.circuit_breaker_triggered:
                        progress_tracker.correction_context = (
                            f"\n\n🛑 CRITIC CIRCUIT BREAKER: {verdict.reason} "
                            "You MUST try a COMPLETELY DIFFERENT approach: scroll, dismiss overlays, wait, or navigate elsewhere."
                        )
                        logger.warning("🛑 Critic circuit breaker: %s", verdict.reason[:100])
                        progress_critic.reset_circuit_breaker()
                    elif correction_failures >= 2:
                        progress_tracker.correction_context = (
                            f"\n\n⚠️ SELF-CORRECTION: Last {correction_failures} actions had NO visible effect. "
                            f"Failed: {'; '.join(progress_tracker.failed_actions[-3:])}. "
                            "Try a COMPLETELY DIFFERENT approach: scroll, dismiss overlays, or wait."
                        )
                        logger.warning("🧠 Critic: no progress (streak=%d)", correction_failures)

                        # v2.1: Gated Reflexion (RQ-B02) — fires ONLY on stagnation streak ≥ 2
                        try:
                            metrics.reflexion_triggers += 1
                            reflect_prompt = (
                                f"I attempted {correction_failures} actions with no progress. "
                                f"Recent failures: {'; '.join(progress_tracker.failed_actions[-3:])}. "
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

            # v14.0: No publishing-specific phase transitions.
            # The LLM decides when the task is 'done' via action_type='done'.

            # ── Phase 2: PRM Checklist Scoring (dense goal-aware progress) ──
            if prm_critic and prm_checklist and decision.action_type in ("click", "type", "goto"):
                try:
                    # Re-extract DOM for PRM (shares the post_dom from CriticV12 above)
                    prm_dom_md = dom_markdown  # Reuse current step's DOM markdown
                    prm_score = await prm_critic.score_step(
                        checklist=prm_checklist,
                        dom_markdown=prm_dom_md,
                        current_url=current_url,
                        step_number=step + 1,
                    )
                    if prm_score.newly_completed:
                        logger.info(
                            "✅ PRM: Completed: %s",
                            ", ".join(c[:40] for c in prm_score.newly_completed),
                        )
                    # Use PRM to extend budget if making good progress
                    if prm_score.total_score >= 0.7 and step >= max_steps - 3:
                        max_steps += 5
                        logger.info(
                            "📊 PRM Budget extended to %d (checklist %.0f%%)",
                            max_steps, prm_score.total_score * 100,
                        )
                except Exception as prm_err:
                    logger.debug("PRM scoring error (non-fatal): %s", prm_err)

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
                    progress_tracker.recovery_advice = advice
                    consecutive_identical_actions = 0  # Reset after getting advice
            else:
                consecutive_identical_actions = 0
                progress_tracker.recovery_advice = ""  # Clear old advice on new action

            last_action_signature = current_action_signature

        logger.info("Agent loop finished after %d steps.", step + 1)

        # ── Record task execution in history ──
        try:
            final_url = page.url
            memory.record_post(
                platform="web",
                title=objective[:120],
                content=objective,
                url=final_url,
                agent_model=chain_names[0] if chain_names else "unknown",
                steps_taken=step + 1,
            )
        except Exception as e:
            logger.warning("Task history record failed: %s", e)

        # ── Phase 3: Record workflow in Skill Memory ──
        try:
            if skill_mem and working_mem.episodic:
                episodic_steps = [dict(s) for s in working_mem.episodic]
                skill_mem.record_workflow(
                    objective=objective,
                    steps=episodic_steps,
                    success=progress_tracker.mission_success,
                    total_steps=step + 1,
                )
                logger.info("📝 Workflow recorded in SkillMemory (success=%s)", progress_tracker.mission_success)
        except Exception as e:
            logger.warning("SkillMemory record failed: %s", e)

        # ── Log Phase metrics ──
        if dreamer:
            logger.info("🌙 WebDreamer total LLM calls: %d", dreamer.total_llm_calls)
        if prm_critic:
            logger.info("📊 PRM Critic total LLM calls: %d", prm_critic.total_llm_calls)

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
        mission_success = progress_tracker.mission_success
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
            print("  ✅  MISSION COMPLETE — Task finished successfully!")
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
