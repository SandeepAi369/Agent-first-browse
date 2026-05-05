from __future__ import annotations

import base64
import zlib
from datetime import UTC, datetime
from typing import Any, Literal

from playwright.async_api import Page

from app.browser_promoter.browser_runtime import BrowserRuntime
from app.browser_promoter.dashboard import (
    pause_for_manual_intervention,
    print_terminal_dashboard,
    resolve_platform_name,
)
from app.observability import build_llm_config
from app.browser_promoter.state import AgentState, BrowserAction, InteractiveRegion, ScreenshotFrame, WorkerFeedback
from app.browser_promoter.supervisor_subgraph import run_supervisor_subgraph
from app.browser_promoter.worker_planner import StrategicPlanner, TacticalSelector, VisionTool
from app.browser_promoter.zero_token_executor import (
    ActionExecutionError,
    ActionExecutionResult,
    ZeroTokenActionExecutor,
)
from app.logger import get_logger

logger = get_logger(__name__)


_STRATEGIC_PLANNER: StrategicPlanner | None = None
_TACTICAL_SELECTOR: TacticalSelector | None = None
_VISION_TOOL: VisionTool | None = None
_ZERO_TOKEN_EXECUTOR: ZeroTokenActionExecutor | None = None


async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Run hierarchical Supervisor subgraph and publish only final HighLevelCommand."""
    logger.info("Running hierarchical supervisor subgraph")
    command = await run_supervisor_subgraph(state)
    return {"high_level_command": command}


def _get_strategic_planner() -> StrategicPlanner:
    global _STRATEGIC_PLANNER
    if _STRATEGIC_PLANNER is None:
        _STRATEGIC_PLANNER = StrategicPlanner()
    return _STRATEGIC_PLANNER

def _get_tactical_selector() -> TacticalSelector:
    global _TACTICAL_SELECTOR
    if _TACTICAL_SELECTOR is None:
        _TACTICAL_SELECTOR = TacticalSelector()
    return _TACTICAL_SELECTOR

def _get_vision_tool() -> VisionTool:
    global _VISION_TOOL
    if _VISION_TOOL is None:
        _VISION_TOOL = VisionTool()
    return _VISION_TOOL


async def strategic_planner_node(state: AgentState) -> dict[str, Any]:
    """Tier 1: NVIDIA GLM 5.1 analyzes context and drafts a strategy or requests vision."""
    logger.info("Strategic Planner (NVIDIA GLM) drafting strategy")
    planner = _get_strategic_planner()
    high_level_command = _resolve_high_level_command(state)
    
    llm_config = build_llm_config(
        run_name="swarm.strategic_planner",
        tags=["strategic_planner", "nvidia"],
        metadata={"thread_id": state.thread_id},
    )

    decision = await planner.draft_strategy(
        high_level_command=high_level_command,
        current_url=state.current_url,
        interactive_regions=state.current_interactive_regions,
        accessibility_tree=state.current_accessibility_tree,
        action_history=list(state.action_history),
        shared_reasoning_log=list(state.shared_reasoning_log),
        llm_config=llm_config,
    )

    log_entry = f"[NVIDIA GLM 5.1] Strategy Draft: {decision.strategy}"
    updated_log = list(state.shared_reasoning_log)
    updated_log.append(log_entry)

    updated_routing = state.routing.model_copy(
        update={
            "vision_requested": decision.vision_requested,
            "next_hop": "tactical_selector",
        }
    )

    return {
        "shared_reasoning_log": updated_log,
        "routing": updated_routing,
        "ephemeral": {"current_strategy": decision.strategy},
    }

async def tactical_selector_node(state: AgentState) -> dict[str, Any]:
    """Tier 2: Groq GPT-OSS-120B converts the strategy into exact JSON BrowserAction."""
    if state.routing.vision_requested:
        # If vision is requested, skip tactics and route to vision tool
        return {}

    logger.info("Tactical Selector (Groq 120B) parsing action from strategy")
    selector = _get_tactical_selector()
    current_strategy = state.ephemeral.get("current_strategy", "Proceed with general search.")
    
    llm_config = build_llm_config(
        run_name="swarm.tactical_selector",
        tags=["tactical_selector", "groq"],
        metadata={"thread_id": state.thread_id},
    )

    decision = await selector.select_tactics(
        strategy=current_strategy,
        interactive_regions=state.current_interactive_regions,
        llm_config=llm_config,
    )
    
    summary = decision.scene_summary.strip() or f"Tactical selection made."
    worker_feedback = list(state.worker_feedback)
    worker_action_queue = list(state.worker_action_queue)

    if decision.confused or decision.action is None:
        feedback = WorkerFeedback(
            command_id=_resolve_command_id(state),
            status="failed",
            message="Tactical Selector could not map strategy to DOM element.",
            details={"confusion_reason": decision.confusion_reason},
        )
        worker_feedback = [*worker_feedback, feedback][-state.feedback_window_size :]

        updated_routing = state.routing.model_copy(
            update={
                "next_hop": "stuck_evaluator",
                "requires_browser_action": False,
                "requires_supervisor_review": True,
            }
        )

        return {
            "worker_feedback": worker_feedback,
            "worker_last_confidence": decision.confidence,
            "worker_last_confused": True,
            "worker_last_confusion_reason": decision.confusion_reason or "Tactical model was uncertain.",
            "current_scene_summary": summary,
            "routing": updated_routing,
            **_garbage_collect_visual_state(state, summary),
        }

    feedback = WorkerFeedback(
        command_id=decision.action.action_id,
        status="completed",
        message="Tactical action mapped successfully.",
        details={"action": decision.action.model_dump(mode="json")},
    )
    worker_feedback = [*worker_feedback, feedback][-state.feedback_window_size :]
    worker_action_queue.append(decision.action)

    updated_routing = state.routing.model_copy(
        update={
            "next_hop": "browser_controller",
            "requires_browser_action": True,
            "requires_supervisor_review": False,
        }
    )

    return {
        "worker_action_queue": worker_action_queue,
        "worker_feedback": worker_feedback,
        "worker_last_confidence": decision.confidence,
        "worker_last_confused": False,
        "worker_last_confusion_reason": "",
        "current_scene_summary": summary,
        "routing": updated_routing,
        **_garbage_collect_visual_state(state, summary),
    }

async def vision_tool_node(state: AgentState) -> dict[str, Any]:
    """Tier 3: Google Gemma 4 performs visual analysis when requested by text models."""
    logger.info("Vision Tool (Gemma 4) analyzing screenshot")
    vision_tool = _get_vision_tool()
    
    llm_config = build_llm_config(
        run_name="swarm.vision_tool",
        tags=["vision_tool", "gemma"],
        metadata={"thread_id": state.thread_id},
    )

    analysis = await vision_tool.analyze(
        screenshot_base64=state.current_screenshot_base64,
        screenshot_encoding=state.current_screenshot_encoding,
        llm_config=llm_config,
    )

    log_entry = f"[Gemma 4 Vision] Screen Analysis: {analysis}"
    updated_log = list(state.shared_reasoning_log)
    updated_log.append(log_entry)

    # Turn off vision_requested so the loop goes back to strategic planner naturally
    updated_routing = state.routing.model_copy(
        update={
            "vision_requested": False,
        }
    )

    return {
        "shared_reasoning_log": updated_log,
        "routing": updated_routing,
    }

def stuck_evaluator_node(state: AgentState) -> dict[str, Any]:
    """Evaluates the proposed action against recent history to break infinite loops."""
    logger.info("Evaluating for stuck/infinite loops")
    
    proposed_actions = state.worker_action_queue
    if not proposed_actions:
        return {}

    proposed = proposed_actions[0]
    action_dict = {"action": proposed.action, "selector": proposed.selector, "text": proposed.text, "x": proposed.x, "y": proposed.y, "url": proposed.url}
    
    history = list(state.action_history)
    consecutive_failures = state.consecutive_failures
    
    is_loop = False
    if len(history) >= 2:
        last_action = history[-1]
        prev_action = history[-2]
        
        # Check exact repetition
        if action_dict == last_action and action_dict == prev_action:
            is_loop = True
            
        # Check ping-pong (A -> B -> A)
        elif action_dict == prev_action and len(history) >= 3 and history[-3] == last_action:
            is_loop = True

        # Check zero-progress (Scroll over and over without finding anything)
        if proposed.action == "scroll" and last_action.get("action") == "scroll" and prev_action.get("action") == "scroll":
             is_loop = True

    if is_loop:
        consecutive_failures += 1
        logger.warning(f"Loop detected! consecutive_failures={consecutive_failures}")
    else:
        consecutive_failures = 0
        
    ephemeral = dict(state.ephemeral)
    
    if consecutive_failures >= 2:
        logger.error("Agent is STUCK. Injecting KICK PROMPT and routing to Supervisor.")
        ephemeral["kick_prompt"] = (
            "HIGH SALIENCE SYSTEM OVERRIDE: You are caught in an infinite loop repeating the same actions "
            "(e.g., scrolling endlessly or clicking the same element). ABANDON your current strategy completely. "
            "If you are on Reddit and not finding results, PIVOT IMMEDIATELY to a different platform "
            "(e.g., HackerNews, GitHub Discussions, Facebook Groups, Dev.to) using the web search tools. "
            "Do NOT propose the same action again."
        )
        return {
            "consecutive_failures": consecutive_failures,
            "ephemeral": ephemeral,
            "worker_action_queue": [],
            "worker_last_confused": True,
            "worker_last_confusion_reason": "Infinite loop detected by stuck evaluator.",
            "action_history": [*history, action_dict][-6:]
        }
        
    return {
        "consecutive_failures": consecutive_failures,
        "action_history": [*history, action_dict][-6:]
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Auth Check Node — Human-in-the-Loop Login Pause
# ═══════════════════════════════════════════════════════════════════════════════

# Known login URL patterns that should trigger HITL pause
_LOGIN_URL_SIGNALS = (
    "accounts.google.com",
    "login.reddit.com",
    "/login",
    "/signin",
    "/sign_in",
    "/auth",
    "facebook.com/login",
    "github.com/login",
    "github.com/session",
    "news.ycombinator.com/login",
    "dev.to/enter",
)

# DOM text signals indicating a login/auth form
_LOGIN_DOM_SIGNALS = (
    "password",
    "sign in",
    "log in",
    "create account",
    "captcha",
    "verification code",
    "two-factor",
    "2fa",
    "mfa",
)


def auth_check_node(state: AgentState) -> dict[str, Any]:
    """Detect login walls and pause execution for human authentication.

    Checks two signals:
      1. URL pattern: If the current URL matches known auth endpoints.
      2. DOM analysis: If interactive regions contain password/login fields.

    When a login wall is detected, prints a prominent terminal message and
    blocks on `input()` so the human can log in manually in the open
    Playwright browser window. Once the human presses Enter, the agent
    resumes with the authenticated session.
    """
    logger.info("Checking for authentication walls")

    url = state.current_url.lower()
    needs_login = any(sig in url for sig in _LOGIN_URL_SIGNALS)

    # Secondary check: scan DOM interactive regions for auth-related fields
    if not needs_login:
        for region in state.current_interactive_regions:
            combined = f"{region.text} {region.role} {region.tag}".lower()
            if any(kw in combined for kw in _LOGIN_DOM_SIGNALS):
                needs_login = True
                break

    if not needs_login:
        return {}

    # ── HITL Login Pause ──
    logger.warning("Login wall detected at: %s", state.current_url)
    print("\n" + "=" * 70)
    print("  🚨 LOGIN REQUIRED")
    print("=" * 70)
    print(f"  URL: {state.current_url}")
    print("")
    print("  The agent detected a login or authentication wall.")
    print("  Please switch to the Playwright Chromium browser window")
    print("  and log in manually (Reddit, Google, GitHub, etc.).")
    print("")
    print("  Once you are logged in, come back here and press ENTER.")
    print("=" * 70)
    try:
        input("\n  👉 Press ENTER when you have finished logging in: ")
    except EOFError:
        logger.info("No interactive stdin — continuing without blocking.")
    print("  ✅ Resuming agent execution...\n")

    return {}


def router_node(state: AgentState) -> dict[str, Any]:
    """Minimal router node; decisions are made in router_function."""
    logger.info("Evaluating routing flags")
    next_cycle = state.cycle_count + 1
    hard_max = state.max_cycles
    still_allowed = state.should_continue and next_cycle < hard_max
    print_terminal_dashboard(state=state, cycle_value=next_cycle, hard_max=hard_max)

    ephemeral = dict(state.ephemeral)
    if _has_pending_manual_intervention_action(state):
        if state.autonomous_continuation:
            remaining_actions = list(state.worker_action_queue)
            if remaining_actions and remaining_actions[0].action == "manual_intervention_required":
                remaining_actions.pop(0)

            feedback_history = list(state.worker_feedback)
            feedback_history = [
                *feedback_history,
                WorkerFeedback(
                    command_id="manual_intervention_autonomous_skip",
                    status="completed",
                    message="Autonomous continuation skipped manual pause and requested supervisor re-plan.",
                    details={
                        "cycle": next_cycle,
                        "platform": resolve_platform_name(state),
                    },
                ),
            ][-state.feedback_window_size :]

            ephemeral["manual_intervention_auto_skipped"] = True
            updated_routing = state.routing.model_copy(
                update={
                    "next_hop": "supervisor",
                    "requires_browser_action": False,
                    "requires_supervisor_review": True,
                }
            )
            return {
                "cycle_count": next_cycle,
                "should_continue": still_allowed,
                "worker_action_queue": remaining_actions,
                "worker_feedback": feedback_history,
                "worker_last_confused": True,
                "worker_last_confusion_reason": (
                    "Autonomous continuation bypassed manual intervention; supervisor should re-plan."
                ),
                "routing": updated_routing,
                "ephemeral": ephemeral,
            }

        pause_for_manual_intervention(state=state, cycle_value=next_cycle, hard_max=hard_max)
        remaining_actions = list(state.worker_action_queue)
        if remaining_actions and remaining_actions[0].action == "manual_intervention_required":
            remaining_actions.pop(0)

        ephemeral["manual_intervention_just_completed"] = True
        ephemeral.pop("manual_intervention_auto_skipped", None)
        return {
            "cycle_count": next_cycle,
            "should_continue": still_allowed,
            "worker_action_queue": remaining_actions,
            "ephemeral": ephemeral,
        }

    ephemeral.pop("manual_intervention_just_completed", None)
    ephemeral.pop("manual_intervention_auto_skipped", None)
    return {
        "cycle_count": next_cycle,
        "should_continue": still_allowed,
        "ephemeral": ephemeral,
    }


def router_function(state: AgentState) -> Literal["browser_controller", "supervisor", "end"]:
    """Deterministic route selector for Step-5 traffic control."""
    hard_max = state.max_cycles
    if state.routing.stop_requested:
        return "end"

    if state.ephemeral.get("manual_intervention_just_completed"):
        return "browser_controller"

    if state.ephemeral.get("manual_intervention_auto_skipped"):
        return "supervisor"

    if not state.should_continue:
        return "end"

    if state.cycle_count >= hard_max:
        return "end"

    if state.worker_last_confused:
        return "supervisor"

    if state.worker_last_confidence < state.worker_confidence_threshold:
        return "supervisor"

    if state.routing.requires_supervisor_review:
        return "supervisor"

    if _worker_has_valid_browser_action(state):
        return "browser_controller"

    return "supervisor"


def _worker_has_valid_browser_action(state: AgentState) -> bool:
    """Validate that Worker produced an actionable browser command."""
    if not state.worker_action_queue:
        return False

    action = state.worker_action_queue[0]
    try:
        BrowserAction.model_validate(action.model_dump(mode="json"))
    except Exception:
        return False
    return True


def _has_pending_manual_intervention_action(state: AgentState) -> bool:
    if not state.worker_action_queue:
        return False
    return state.worker_action_queue[0].action == "manual_intervention_required"


def _get_worker_planner() -> WorkerVisionPlanner:
    global _WORKER_PLANNER
    if _WORKER_PLANNER is None:
        _WORKER_PLANNER = WorkerVisionPlanner()
    return _WORKER_PLANNER


def _get_zero_token_executor() -> ZeroTokenActionExecutor:
    global _ZERO_TOKEN_EXECUTOR
    if _ZERO_TOKEN_EXECUTOR is None:
        _ZERO_TOKEN_EXECUTOR = ZeroTokenActionExecutor()
    return _ZERO_TOKEN_EXECUTOR


def _resolve_high_level_command(state: AgentState) -> str:
    """Resolve current high-level command from explicit state or queued supervisor commands."""
    if state.high_level_command is not None:
        pieces = [
            f"action_type={state.high_level_command.action_type}",
            f"target={state.high_level_command.target_description}",
            f"behavior={state.high_level_command.behavior_plan}",
        ]
        if state.high_level_command.draft_text:
            pieces.append(f"draft_text={state.high_level_command.draft_text}")
        if state.high_level_command.stealth_adjustments:
            pieces.append(
                "stealth_adjustments=" + "; ".join(state.high_level_command.stealth_adjustments[:5])
            )
        return " | ".join(pieces)

    if state.supervisor_commands:
        # Prefer highest-priority most-recent command.
        ranked = sorted(
            state.supervisor_commands,
            key=lambda c: (c.priority, c.command_id),
            reverse=True,
        )
        return ranked[0].instruction.strip()

    return state.campaign.objective.strip()


def _resolve_command_id(state: AgentState) -> str:
    if state.supervisor_commands:
        return state.supervisor_commands[-1].command_id
    return "worker_confusion"


def _garbage_collect_visual_state(state: AgentState, scene_summary: str) -> dict[str, Any]:
    """
    Purge heavy visual payloads after Worker consumes them.

    This keeps runtime memory bounded by removing screenshot bytes and dense
    region/accessibility payloads while preserving minimal textual summaries.
    """
    compact_history: list[ScreenshotFrame] = []
    for frame in state.screenshot_history:
        compact_history.append(
            frame.model_copy(
                update={
                    "screenshot_base64": "",
                    "screenshot_encoding": "purged",
                    "compressed_bytes": 0,
                    "original_bytes": 0,
                    "accessibility_tree": {},
                    "interactive_regions": [],
                    "scene_summary": frame.scene_summary or scene_summary,
                }
            )
        )

    return {
        "screenshot_history": compact_history,
        "current_screenshot_base64": "",
        "current_screenshot_encoding": "purged",
        "current_accessibility_tree": {},
        "current_interactive_regions": [],
    }


# _resolve_platform_name is now imported from app.browser_promoter.dashboard
# as resolve_platform_name and used across the codebase.


def _trim_accessibility_tree(tree: dict[str, Any] | None, max_nodes: int = 250) -> dict[str, Any]:
    """Trim accessibility payload to keep memory bounded in long-running loops."""
    if not tree:
        return {}

    seen = 0

    def walk(node: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal seen
        if seen >= max_nodes:
            return None

        seen += 1
        compact: dict[str, Any] = {}
        for key in (
            "role",
            "name",
            "value",
            "description",
            "checked",
            "selected",
            "expanded",
            "disabled",
            "focused",
        ):
            value = node.get(key)
            if value not in (None, "", False):
                compact[key] = value

        children = node.get("children") or []
        compact_children: list[dict[str, Any]] = []
        for child in children:
            if seen >= max_nodes:
                break
            child_result = walk(child)
            if child_result is not None:
                compact_children.append(child_result)
        if compact_children:
            compact["children"] = compact_children

        return compact

    return walk(tree) or {}


async def _extract_interactive_regions(page: Page, max_regions: int = 120) -> list[InteractiveRegion]:
    """Collect bounding boxes for clickable/focusable elements."""
    raw_regions = await page.evaluate(
        """
(limit) => {
  const isVisible = (el) => {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
  };

  const getSimpleSelector = (el) => {
    if (el.id) return `#${el.id}`;
    const testId = el.getAttribute('data-testid');
    if (testId) return `[data-testid="${testId}"]`;
    const name = el.getAttribute('name');
    if (name) return `${el.tagName.toLowerCase()}[name="${name}"]`;

    let index = 1;
    let sibling = el;
    while ((sibling = sibling.previousElementSibling) != null) {
      if (sibling.tagName === el.tagName) index += 1;
    }
    return `${el.tagName.toLowerCase()}:nth-of-type(${index})`;
  };

  const candidates = document.querySelectorAll(
    'a, button, input, textarea, select, [role="button"], [role="link"], [tabindex]'
  );

  const output = [];
  for (const el of candidates) {
    if (!isVisible(el)) continue;

    const rect = el.getBoundingClientRect();
    output.push({
      tag: el.tagName.toLowerCase(),
      role: (el.getAttribute('role') || '').slice(0, 40),
      text: ((el.innerText || el.value || el.getAttribute('aria-label') || '').trim()).slice(0, 140),
      selector: getSimpleSelector(el).slice(0, 180),
      x: Math.round(rect.left + window.scrollX),
      y: Math.round(rect.top + window.scrollY),
      width: Math.round(rect.width),
      height: Math.round(rect.height)
    });

    if (output.length >= limit) break;
  }

  return output;
}
""",
        max_regions,
    )

    regions: list[InteractiveRegion] = []
    for item in raw_regions:
        try:
            regions.append(InteractiveRegion.model_validate(item))
        except Exception:
            continue
    return regions


async def _capture_snapshot(page: Page) -> ScreenshotFrame:
    """
    Capture compressed screenshot and rich UI metadata.

    The screenshot is JPEG-compressed first, then zlib-compressed before base64
    encoding so state transport remains compact.
    Resilient to navigation race conditions — returns minimal snapshot on failure.
    """
    # Wait for the page to be in a stable state before capturing
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=10000)
    except Exception:
        pass  # Best-effort; page may still be loading

    # Small delay for JS rendering to settle
    await page.wait_for_timeout(500)

    screenshot_bytes = await page.screenshot(type="jpeg", quality=55, full_page=False)
    compressed_bytes = zlib.compress(screenshot_bytes, level=6)
    encoded_image = base64.b64encode(compressed_bytes).decode("ascii")

    # These can fail if a navigation just happened — catch and return empty
    trimmed_tree: dict[str, Any] = {}
    interactive_regions: list[InteractiveRegion] = []
    try:
        try:
            raw_tree = await getattr(page, "accessibility").snapshot(interesting_only=False)
        except AttributeError:
            raw_tree = {}
        trimmed_tree = _trim_accessibility_tree(raw_tree, max_nodes=250)
        interactive_regions = await _extract_interactive_regions(page, max_regions=120)
    except Exception:
        # Navigation destroyed the execution context — return what we have
        pass

    return ScreenshotFrame(
        captured_at=datetime.now(UTC),
        url=page.url,
        screenshot_base64=encoded_image,
        screenshot_encoding="zlib+base64:jpeg",
        original_bytes=len(screenshot_bytes),
        compressed_bytes=len(compressed_bytes),
        accessibility_tree=trimmed_tree,
        interactive_regions=interactive_regions,
    )


async def _execute_action_with_fallbacks(
    page: Page,
    action: BrowserAction,
    *,
    dry_run_mode: bool,
) -> ActionExecutionResult:
    """Execute one browser action with A/B/C fallback flow and dry-run guardrails."""
    if dry_run_mode and _is_destructive_action(action):
        await _preview_action_target(page, action)
        return ActionExecutionResult(
            details={
                "action": action.action,
                "dry_run_mode": True,
                "dry_run_skip_reason": "Potentially destructive action skipped.",
                "selector": action.selector,
                "x": action.x,
                "y": action.y,
                "fallback_stage": "A",
            },
            vision_calls_used=0,
        )

    executor = _get_zero_token_executor()
    return await executor.execute_action(page=page, action=action)


def _is_destructive_action(action: BrowserAction) -> bool:
    """Detect likely post/reply submission actions that should be dry-run protected."""
    if action.action not in {"click", "type", "type_and_enter"}:
        return False

    haystack = " ".join([action.selector, action.text, action.url]).lower()
    destructive_keywords = (
        "post",
        "reply",
        "comment",
        "submit",
        "send",
        "tweet",
        "publish",
        "create",
        "share",
        "commit",
        "merge",
        "new discussion",
    )
    return any(keyword in haystack for keyword in destructive_keywords)


async def _preview_action_target(page: Page, action: BrowserAction) -> None:
    """Highlight intended target in dry-run mode instead of executing destructive action."""
    if action.selector:
        await page.evaluate(
            """
(selector) => {
  const element = document.querySelector(selector);
  if (!element) return false;

  element.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' });
  const previous = element.style.outline;
  element.style.outline = '3px solid #ff9800';
  setTimeout(() => { element.style.outline = previous; }, 1600);
  return true;
}
""",
            action.selector,
        )
        await page.wait_for_timeout(400)
        return

    if action.x is not None and action.y is not None:
        await page.evaluate(
            """
([x, y]) => {
  const marker = document.createElement('div');
  marker.style.position = 'absolute';
  marker.style.left = `${x - 8}px`;
  marker.style.top = `${y - 8}px`;
  marker.style.width = '16px';
  marker.style.height = '16px';
  marker.style.borderRadius = '50%';
  marker.style.border = '2px solid #ff9800';
  marker.style.background = 'rgba(255, 152, 0, 0.25)';
  marker.style.zIndex = '2147483647';
  marker.id = '__agent_dry_run_marker';
  document.body.appendChild(marker);
  setTimeout(() => marker.remove(), 1600);
}
""",
            [action.x, action.y],
        )
        await page.wait_for_timeout(400)


async def browser_controller_node(state: AgentState) -> dict[str, Any]:
    """
    Execute queued worker browser actions on a persistent stealth Playwright page.

    This node updates compressed screenshot payload, accessibility metadata,
    and interactive bounding boxes each cycle.
    """
    pending_actions = list(state.worker_action_queue)
    screenshot_history = list(state.screenshot_history)
    feedback_history = list(state.worker_feedback)

    try:
        page = await BrowserRuntime.ensure_page(
            browser_config=state.browser_config,
            platform_name=resolve_platform_name(state),
            thread_id=state.thread_id,
        )
    except Exception as exc:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"ensure_page FAILED EXCEPTION: {exc}")
        feedback = WorkerFeedback(
            command_id="browser_runtime",
            status="failed",
            message="Browser runtime initialization failed.",
            details={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "mode": state.browser_config.mode.value,
            },
        )
        feedback_history = [*feedback_history, feedback][-state.feedback_window_size :]

        updated_routing = state.routing.model_copy(
            update={
                "next_hop": "router",
                "requires_browser_action": False,
                "requires_supervisor_review": True,
            }
        )
        return {
            "worker_action_queue": [],
            "worker_feedback": feedback_history,
            "worker_last_confused": True,
            "worker_last_confusion_reason": "Browser runtime initialization failed.",
            "routing": updated_routing,
        }

    if not pending_actions:
        snapshot = await _capture_snapshot(page)
        screenshot_history = [*screenshot_history, snapshot][-state.screenshot_window_size :]

        feedback = WorkerFeedback(
            command_id="none",
            status="completed",
            message="No worker action queued. Browser context snapshot refreshed.",
            details={"url": page.url},
        )
        feedback_history = [*feedback_history, feedback][-state.feedback_window_size :]

        updated_routing = state.routing.model_copy(
            update={
                "next_hop": "router",
                "requires_browser_action": False,
                "requires_supervisor_review": False,
            }
        )
        return {
            "screenshot_history": screenshot_history,
            "worker_feedback": feedback_history,
            "current_url": snapshot.url,
            "current_screenshot_base64": snapshot.screenshot_base64,
            "current_screenshot_encoding": snapshot.screenshot_encoding,
            "current_accessibility_tree": snapshot.accessibility_tree,
            "current_interactive_regions": snapshot.interactive_regions,
            "routing": updated_routing,
        }

    action = pending_actions.pop(0)
    status: Literal["completed", "failed"] = "completed"
    details: dict[str, Any]
    vision_calls_used = 0

    try:
        execution: ActionExecutionResult = await _execute_action_with_fallbacks(
            page,
            action,
            dry_run_mode=state.dry_run_mode,
        )
        details = execution.details
        vision_calls_used = execution.vision_calls_used
    except ActionExecutionError as exc:
        status = "failed"
        vision_calls_used = exc.vision_calls_used
        details = {
            "action": action.action,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    except Exception as exc:
        status = "failed"
        details = {
            "action": action.action,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    snapshot = await _capture_snapshot(page)
    screenshot_history = [*screenshot_history, snapshot][-state.screenshot_window_size :]

    feedback = WorkerFeedback(
        command_id=action.action_id,
        status=status,
        message=f"browser_controller processed action={action.action} status={status}",
        details=details,
    )
    feedback_history = [*feedback_history, feedback][-state.feedback_window_size :]

    updated_routing = state.routing.model_copy(
        update={
            "next_hop": "router",
            "requires_browser_action": bool(pending_actions),
            "requires_supervisor_review": status == "failed",
        }
    )

    return {
        "worker_action_queue": pending_actions,
        "screenshot_history": screenshot_history,
        "worker_feedback": feedback_history,
        "current_url": snapshot.url,
        "current_screenshot_base64": snapshot.screenshot_base64,
        "current_screenshot_encoding": snapshot.screenshot_encoding,
        "current_accessibility_tree": snapshot.accessibility_tree,
        "current_interactive_regions": snapshot.interactive_regions,
        "vision_calls": state.vision_calls + vision_calls_used,
        "routing": updated_routing,
    }


async def task_logging_node(state: AgentState) -> dict[str, Any]:
    """Summarize final state and push to completed_tasks table."""
    logger.info("task_logging_node: Summarizing execution and logging to completed_tasks")
    
    import json
    from app.browser_promoter.database import initialize_persistence_database
    import sqlite3
    
    summary = f"Completed task '{state.high_level_command.behavior_plan if state.high_level_command else 'Unknown'}' after {state.cycle_count} cycles."
    
    final_state_data = {
        "thread_id": state.thread_id,
        "cycles": state.cycle_count,
        "final_url": state.current_url,
        "actions_taken": len(state.action_history),
        "db_accesses": len(state.db_access_logs)
    }
    
    db_path = initialize_persistence_database()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO completed_tasks(thread_id, summary, final_state_json) VALUES(?, ?, ?)",
                (state.thread_id, summary, json.dumps(final_state_data))
            )
            conn.commit()
    except Exception as e:
        log.error(f"Failed to write to completed_tasks: {e}")

    updated_routing = state.routing.model_copy(update={"next_hop": "housekeeping_node"})
    return {"routing": updated_routing}


async def housekeeping_node(state: AgentState) -> dict[str, Any]:
    """Clean up temp files, drop dynamic tables, and purge ephemeral state."""
    log.info("housekeeping_node: Beginning system cleanup")
    import os
    import sqlite3
    from app.browser_promoter.database import initialize_persistence_database
    
    # 1. Temp file cleanup
    files_removed = 0
    for fpath in state.temp_files:
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
                files_removed += 1
        except Exception as e:
            log.warning(f"Failed to remove temp file {fpath}: {e}")
            
    # 2. Drop dynamic schemas
    tables_dropped = 0
    if state.dynamic_schemas_created:
        db_path = initialize_persistence_database()
        with sqlite3.connect(db_path) as conn:
            for table_name in state.dynamic_schemas_created:
                try:
                    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    tables_dropped += 1
                except Exception as e:
                    log.warning(f"Failed to drop table {table_name}: {e}")
            conn.commit()
            
    log.info(f"housekeeping_node: Cleaned up {files_removed} files and {tables_dropped} dynamic tables.")
    
    updated_routing = state.routing.model_copy(update={"next_hop": "end"})
    return {
        "routing": updated_routing,
        "temp_files": [],
        "db_access_logs": [],
        "dynamic_schemas_created": []
    }
