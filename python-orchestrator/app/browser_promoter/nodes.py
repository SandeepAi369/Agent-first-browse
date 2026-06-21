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
from app.browser_promoter.state import AgentState, BrowserAction, ScreenshotFrame, WorkerFeedback
from app.browser_promoter.supervisor_subgraph import run_supervisor_subgraph
from app.browser_promoter.worker_planner import ReasoningAgent, VisionAgent
from app.browser_promoter.zero_token_executor import (
    ActionExecutionError,
    ActionExecutionResult,
    ZeroTokenActionExecutor,
)
from app.logger import get_logger

logger = get_logger(__name__)


_VISION_AGENT: VisionAgent | None = None
_REASONING_AGENT: ReasoningAgent | None = None
_ZERO_TOKEN_EXECUTOR: ZeroTokenActionExecutor | None = None


async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Run hierarchical Supervisor subgraph and publish only final HighLevelCommand."""
    logger.info("Running hierarchical supervisor subgraph")
    command = await run_supervisor_subgraph(state)
    return {"high_level_command": command}


def _get_vision_agent() -> VisionAgent:
    global _VISION_AGENT
    if _VISION_AGENT is None:
        _VISION_AGENT = VisionAgent()
    return _VISION_AGENT


def _get_reasoning_agent() -> ReasoningAgent:
    global _REASONING_AGENT
    if _REASONING_AGENT is None:
        _REASONING_AGENT = ReasoningAgent()
    return _REASONING_AGENT


async def vision_agent_node(state: AgentState) -> dict[str, Any]:
    """Step 1: Vision agent returns strict JSON mapping of screen coordinates."""
    if not state.current_screenshot_base64:
        return {}

    logger.info("Vision Agent mapping interactive elements")
    vision_agent = _get_vision_agent()

    llm_config = build_llm_config(
        run_name="swarm.vision_agent",
        tags=["vision_agent"],
        metadata={"thread_id": state.thread_id},
    )

    vision_map_json = await vision_agent.detect_elements(
        screenshot_base64=state.current_screenshot_base64,
        screenshot_encoding=state.current_screenshot_encoding,
        llm_config=llm_config,
    )

    updated_routing = state.routing.model_copy(
        update={
            "next_hop": "reasoning_agent",
        }
    )

    return {
        "current_vision_map_json": vision_map_json,
        "vision_calls": state.vision_calls + 1,
        "routing": updated_routing,
    }


async def reasoning_agent_node(state: AgentState) -> dict[str, Any]:
    """Step 2: Reasoning agent decides the next BrowserAction from vision JSON."""
    logger.info("Reasoning Agent selecting action from vision map")
    reasoning_agent = _get_reasoning_agent()
    high_level_command = _resolve_high_level_command(state)

    llm_config = build_llm_config(
        run_name="swarm.reasoning_agent",
        tags=["reasoning_agent"],
        metadata={"thread_id": state.thread_id},
    )

    decision = await reasoning_agent.decide_action(
        high_level_command=high_level_command,
        current_url=state.current_url,
        vision_map_json=state.current_vision_map_json,
        action_history=list(state.action_history),
        llm_config=llm_config,
    )

    if decision.action and decision.action.action in {"click", "type", "type_and_enter"}:
        if decision.action.x is None or decision.action.y is None:
            decision = decision.model_copy(
                update={
                    "confused": True,
                    "confusion_reason": "Missing coordinates for targeted action.",
                }
            )

    summary = decision.scene_summary.strip() or "Reasoning action selected."
    worker_feedback = list(state.worker_feedback)
    worker_action_queue = list(state.worker_action_queue)

    reasoning_log = list(state.shared_reasoning_log)
    if decision.reasoning:
        reasoning_log.append(f"[Reasoning] {decision.reasoning}")

    if decision.confused or decision.action is None:
        feedback = WorkerFeedback(
            command_id=_resolve_command_id(state),
            status="failed",
            message="Reasoning agent could not produce a valid action.",
            details={"confusion_reason": decision.confusion_reason},
        )
        worker_feedback = [*worker_feedback, feedback][-state.feedback_window_size :]

        updated_routing = state.routing.model_copy(
            update={
                "next_hop": "router",
                "requires_browser_action": False,
                "requires_supervisor_review": True,
            }
        )

        return {
            "worker_feedback": worker_feedback,
            "worker_last_confidence": decision.confidence,
            "worker_last_confused": True,
            "worker_last_confusion_reason": decision.confusion_reason or "Reasoning model was uncertain.",
            "current_scene_summary": summary,
            "shared_reasoning_log": reasoning_log,
            "routing": updated_routing,
        }

    feedback = WorkerFeedback(
        command_id=decision.action.action_id,
        status="completed",
        message="Reasoning action mapped successfully.",
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
        "shared_reasoning_log": reasoning_log,
        "routing": updated_routing,
        **_garbage_collect_visual_state(state, summary),
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

def auth_check_node(state: AgentState) -> dict[str, Any]:
    """Detect login walls and pause execution for human authentication.

    Checks two signals:
      1. URL pattern: If the current URL matches known auth endpoints.

    When a login wall is detected, prints a prominent terminal message and
    blocks on `input()` so the human can log in manually in the open
    Playwright browser window. Once the human presses Enter, the agent
    resumes with the authenticated session.
    """
    logger.info("Checking for authentication walls")

    url = state.current_url.lower()
    needs_login = any(sig in url for sig in _LOGIN_URL_SIGNALS)

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
                    "scene_summary": frame.scene_summary or scene_summary,
                }
            )
        )

    return {
        "screenshot_history": compact_history,
        "current_screenshot_base64": "",
        "current_screenshot_encoding": "purged",
    }


# _resolve_platform_name is now imported from app.browser_promoter.dashboard
# as resolve_platform_name and used across the codebase.


async def _capture_snapshot(page: Page) -> ScreenshotFrame:
    """
    Capture a compressed screenshot for the vision agent.

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

    return ScreenshotFrame(
        captured_at=datetime.now(UTC),
        url=page.url,
        screenshot_base64=encoded_image,
        screenshot_encoding="zlib+base64:jpeg",
        original_bytes=len(screenshot_bytes),
        compressed_bytes=len(compressed_bytes),
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
    """Highlight intended target in dry-run mode using coordinates only."""
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

    This node updates compressed screenshot payload each cycle.
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
            "current_vision_map_json": "",
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
        "current_vision_map_json": "",
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
        logger.error(f"Failed to write to completed_tasks: {e}")

    updated_routing = state.routing.model_copy(update={"next_hop": "housekeeping_node"})
    return {"routing": updated_routing}


async def housekeeping_node(state: AgentState) -> dict[str, Any]:
    """Clean up temp files, drop dynamic tables, and purge ephemeral state."""
    logger.info("housekeeping_node: Beginning system cleanup")
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
            logger.warning(f"Failed to remove temp file {fpath}: {e}")
            
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
                    logger.warning(f"Failed to drop table {table_name}: {e}")
            conn.commit()
            
    logger.info(f"housekeeping_node: Cleaned up {files_removed} files and {tables_dropped} dynamic tables.")
    
    updated_routing = state.routing.model_copy(update={"next_hop": "end"})
    return {
        "routing": updated_routing,
        "temp_files": [],
        "db_access_logs": [],
        "dynamic_schemas_created": []
    }
