"""Overwatch — Multi-Layered Verification Subgraph for True Brain v16.0.

The Overwatch is the ONLY node that can commit state. Every worker
proposes an action; Overwatch validates it before it becomes permanent.

Verification layers (ordered cheap → expensive):
  Layer 1: Deterministic state validation (~0ms)
  Layer 2: Grounding validation (~5ms)
  Layer 3: Action execution + DOM ground-truth (CriticV12) (~200ms)
  Layer 4: CoVe action-trail gate (for 'done' actions) (~0ms)
  Layer 5: Loop detection + circuit breaker (~0ms)

Design:
  - Layers are ordered by cost: 80% of failures caught at zero LLM cost
  - Each layer can short-circuit with a verdict (retry/rollback/escalate)
  - Only Layer 3 actually executes the action on the live browser
  - The Overwatch NEVER calls the LLM for decisions — it uses
    deterministic checks + existing CriticV12 for post-action analysis

References:
  - Universal Verifier (arXiv 2604.06240): layered verification cuts
    false positives from 30%+ to 1–8%
  - SAGE-32B (arXiv 2601.04237): verification sensitivity formula
    ε' = ε(1−α) + ε·α·ε_retry
  - Six Sigma Agent (arXiv 2601.22290): consensus voting for irreversible actions
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

try:
    from app.logger import get_logger
    logger = get_logger("overwatch")
except ImportError:
    logger = logging.getLogger("overwatch")


# ═══════════════════════════════════════════════════════════════════════════════
#  V18 Cognition — escalation ladder driver
# ═══════════════════════════════════════════════════════════════════════════════

def _escalate(state: dict, reason: str, updates: dict) -> dict:
    """Advance the deterministic escalation ladder for a stuck state.

    Writes the next DISTINCT tactic's directive into `correction_context`
    (reusing the existing worker-prompt injection slot), advances the ladder,
    and decays strategy confidence. When the ladder is exhausted it sets the
    verdict to 'rollback' so the rollback node re-strategizes. Returns the
    ladder dict so the caller can see whether a re-strategy was requested.
    """
    from cognition import advance_ladder, update_confidence, obstacle_key

    plan_steps = state.get("plan_steps", [])
    active = next((s for s in plan_steps
                   if s.get("status") in ("active", "in_progress")), None)
    plan_step_desc = active.get("desc", "") if active else ""
    obk = obstacle_key(state.get("current_url", ""), plan_step_desc)

    lad = advance_ladder(
        state.get("current_obstacle", ""), obk,
        state.get("ladder_rung", 0), state.get("tried_tactics", []),
    )
    updates["current_obstacle"] = lad["obstacle"]
    updates["ladder_rung"] = lad["rung"]
    updates["tried_tactics"] = lad["tried"]
    updates["strategy_confidence"] = update_confidence(
        state.get("strategy_confidence", 1.0), progress=False)

    if lad["restrategize"]:
        updates["correction_context"] = (
            f"\n\n⚠️ {reason} Prior tactics did not work — forming a NEW strategy."
        )
        updates["overwatch_verdict"] = "rollback"
    else:
        updates["correction_context"] = (
            f"\n\n🧗 ADAPTIVE TACTIC [{lad['tactic']}]: {lad['directive']}"
        )
    logger.info("🧗 Escalation [%s] rung→%d confidence=%.2f (%s)",
                lad["tactic"], lad["rung"], updates["strategy_confidence"], reason)
    return lad


# ═══════════════════════════════════════════════════════════════════════════════
#  Overwatch Node — The verification gate
# ═══════════════════════════════════════════════════════════════════════════════

async def overwatch_node(state: dict[str, Any], page, critic, action_verifier=None) -> dict:
    """Multi-layered verification before committing state.

    Args:
        state: Current BrainState dict
        page: Live Playwright page reference
        critic: CriticV12 instance
        action_verifier: ActionVerifier instance (optional)

    Returns:
        Dict of state updates including overwatch_verdict
    """
    proposed = state.get("proposed_action")
    if not proposed:
        logger.warning("Overwatch: no proposed action")
        return {"overwatch_verdict": "pass"}

    verb = proposed.get("verb", "")
    element_id = proposed.get("element_id")
    selector_map = state.get("selector_map", {})
    step_number = state.get("step_number", 0)

    updates: dict[str, Any] = {}

    # ═══════════════════════════════════════════════════════════════════
    #  Layer 1: Deterministic State Validation (~0ms)
    # ═══════════════════════════════════════════════════════════════════

    page_fsm = state.get("page_fsm", "READY")
    if page_fsm == "ERROR":
        logger.warning("⚠️ Overwatch L1: page FSM in ERROR state")
        updates["overwatch_verdict"] = "rollback"
        updates["error_count"] = state.get("error_count", 0) + 1
        return updates

    if verb == "done":
        # Done actions skip to Layer 4 (CoVe check)
        return await _layer_4_cove_check(state, page, updates)

    # ═══════════════════════════════════════════════════════════════════
    #  Layer 2: Grounding Validation (~5ms)
    # ═══════════════════════════════════════════════════════════════════

    if verb in ("click", "type"):
        from mcp_tools import mcp_ground_action
        ground_result = await mcp_ground_action(
            element_id=element_id,
            x=proposed.get("x"),
            y=proposed.get("y"),
            selector_map=selector_map,
            elements_list=state.get("elements_list", []),
        )

        if not ground_result["grounded"]:
            logger.warning(
                "🎯 Overwatch L2: grounding REJECT — %s",
                ground_result["reason"]
            )
            updates["overwatch_verdict"] = "retry"
            updates["error_count"] = state.get("error_count", 0) + 1
            updates["grounding_rejects"] = state.get("grounding_rejects", 0) + 1
            updates["failures"] = state.get("failures", []) + [{
                "action": f"{verb}({element_id or ''})",
                "why": ground_result["reason"],
                "count": 1,
            }]
            return updates

        # Update coordinates from grounding
        proposed["x"] = ground_result["x"]
        proposed["y"] = ground_result["y"]
        updates["proposed_action"] = proposed

    # ═══════════════════════════════════════════════════════════════════
    #  Layer 3: Execute + DOM Ground-Truth Check (CriticV12)
    # ═══════════════════════════════════════════════════════════════════

    # Pre-action snapshot — V19: REUSE the perceive snapshot already in state
    # instead of re-extracting. This (a) saves one full DOM extraction per step,
    # and (b) crucially keeps window.__aid (the element-handle registry) exactly
    # as the LLM saw it, so the action resolves the SAME node the LLM chose
    # (a re-extract here would rebuild the registry right before the click).
    try:
        from cognitive_core import dom_data_to_a11y_format
        pre_dom = {
            "elements": state.get("elements_list", []),
            "element_count": state.get("element_count", 0),
            "markdown": state.get("dom_markdown", ""),
        }
        pre_a11y = dom_data_to_a11y_format(pre_dom)
        await critic.snapshot_before(pre_a11y)
    except Exception as snap_err:
        logger.warning("Overwatch L3: pre-snapshot failed: %s", snap_err)
        try:
            await critic.snapshot_before(None)
        except Exception:
            pass

    # Execute the action via MCP tools
    action_outcome = await _execute_action(proposed, page)
    updates["action_outcome"] = action_outcome
    updates["page_fsm"] = "ACTION_PENDING"
    updates["total_actions"] = state.get("total_actions", 0) + 1

    # Post-action DOM check
    try:
        import dom_parser
        from cognitive_core import dom_data_to_a11y_format
        post_dom = await dom_parser.extract(page, timeout=3.0)
        post_a11y = dom_data_to_a11y_format(post_dom)
    except Exception:
        post_a11y = None

    px = proposed.get("x")
    py = proposed.get("y")
    x_val = int(px) if px is not None else 0
    y_val = int(py) if py is not None else 0
    target_ref = element_id or f"({x_val},{y_val})"
    verdict = await critic.evaluate(
        action=verb,
        target_ref=target_ref,
        target_name=proposed.get("text", "")[:30] if proposed.get("text") else proposed.get("target_name", ""),
        a11y_data_after=post_a11y,
    )

    if verdict.success:
        updates["correction_failures"] = 0
        updates["correction_context"] = ""
        updates["critic_progress"] = state.get("critic_progress", 0) + 1
        logger.info("✓ Overwatch L3: %s [%.0f%%]", verdict.reason[:100], verdict.confidence * 100)
    else:
        cf = state.get("correction_failures", 0) + 1
        updates["correction_failures"] = cf
        updates["critic_no_progress"] = state.get("critic_no_progress", 0) + 1

        if verdict.circuit_breaker_triggered:
            logger.warning("🛑 Overwatch L3: circuit breaker — %s", verdict.reason[:100])
            _escalate(state, "Critic circuit breaker.", updates)
            updates["overwatch_verdict"] = "rollback"
            updates["reflexion_triggers"] = state.get("reflexion_triggers", 0) + 1
            return updates

        if cf >= 2:
            # Escalate through the ladder (distinct tactic each time; never repeats).
            _escalate(state, f"Last {cf} actions had no visible effect.", updates)
            updates["reflexion_triggers"] = state.get("reflexion_triggers", 0) + 1
            # Honor a ladder-forced rollback; otherwise retry until cf exhausts.
            if updates.get("overwatch_verdict") != "rollback":
                updates["overwatch_verdict"] = "retry" if cf < 3 else "rollback"
            return updates

        # cf == 1: one free re-decision before we start escalating.
        updates["overwatch_verdict"] = "retry"
        return updates

    # ═══════════════════════════════════════════════════════════════════
    #  Layer 5: Loop Detection + Circuit Breaker (~0ms)
    # ═══════════════════════════════════════════════════════════════════

    sig = f"{verb}|{element_id or ''}|{state.get('current_url', '')}"
    loop_sigs = state.get("loop_signatures", [])
    sig_count = loop_sigs[-12:].count(sig) if loop_sigs else 0  # sliding window

    if sig_count >= 3:
        logger.warning("🔄 Overwatch L5: loop detected — sig '%s' x%d", sig[:60], sig_count)
        _escalate(state, "Loop detected (same action repeating).", updates)
        updates["overwatch_verdict"] = "rollback"
        updates["loop_signatures"] = [sig]
        return updates

    # ── All layers passed ──
    updates["overwatch_verdict"] = "pass"
    updates["page_fsm"] = "VALIDATED"
    updates["loop_signatures"] = [sig]

    # Record step in history
    history_entry = {
        "step": step_number + 1,
        "action": f"{verb}" + (f"({x_val},{y_val})" if px is not None else ""),
        "outcome": action_outcome,
        "screen": proposed.get("screen_state", "")[:80],
        "url": page.url if page else "",
    }
    updates["history"] = [history_entry]

    return updates


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 4: CoVe Action-Trail Verification (for 'done' actions)
# ═══════════════════════════════════════════════════════════════════════════════

async def _layer_4_cove_check(state: dict, page, updates: dict) -> dict:
    """Chain-of-Verification before declaring done."""
    from execution_safety import cove_pre_done_check
    from cognitive_core import PlanState, WorkingMemory

    # Reconstruct PlanState and WorkingMemory from BrainState
    plan = PlanState(
        mission=state.get("objective", ""),
        steps=state.get("plan_steps", []),
        current_step_id=state.get("plan_cursor", 0),
    )

    working_mem = WorkingMemory()
    # Populate working_mem.history from state for action-trail check
    for h in state.get("history", []):
        working_mem.episodic.append(h)

    objective = state.get("objective", "")
    cove_ok, cove_reason = await cove_pre_done_check(page, plan, working_mem, objective)

    if not cove_ok:
        logger.warning("🛡️ Overwatch L4: CoVe BLOCKED premature done — %s", cove_reason)
        updates["overwatch_verdict"] = "retry"
        updates["done_blocked"] = state.get("done_blocked", 0) + 1
        updates["history"] = [{
            "step": state.get("step_number", 0) + 1,
            "action": "done",
            "outcome": f"BLOCKED: {cove_reason[:60]}",
        }]
        return updates

    logger.info("✅ Overwatch L4: CoVe verified done — %s", cove_reason[:60])
    updates["overwatch_verdict"] = "pass"
    updates["mission_success"] = True
    return updates


# ═══════════════════════════════════════════════════════════════════════════════
#  Action Executor — delegates to MCP tools
# ═══════════════════════════════════════════════════════════════════════════════

async def _execute_action(proposed: dict, page) -> str:
    """Execute a proposed action via MCP tools. Returns outcome string."""
    verb = proposed.get("verb", "wait")

    try:
        if verb == "goto":
            from mcp_tools import mcp_navigate
            result = await mcp_navigate(proposed.get("url", ""))
            if result["success"]:
                return f"→ OK (navigated to {result['url'][:60]})"
            return f"→ FAILED: {result['error']}"

        elif verb == "click":
            from mcp_tools import mcp_click
            result = await mcp_click(
                x=proposed.get("x", 0),
                y=proposed.get("y", 0),
                element_id=proposed.get("element_id"),
            )
            if result["success"]:
                return (
                    f"→ OK (click via {result['strategy']}"
                    f"{',' + ' navigated' if result['navigated'] else ''}"
                    f"{',' + ' DOM changed' if result['dom_changed'] else ''})"
                )
            return f"→ CLICK INEFFECTIVE: {result['error']}"

        elif verb == "type":
            from mcp_tools import mcp_type
            result = await mcp_type(
                text=proposed.get("text", ""),
                x=proposed.get("x", 0),
                y=proposed.get("y", 0),
                element_id=proposed.get("element_id"),
            )
            if result["success"]:
                return f"→ OK (typed {result['actual_length']} chars via {result['strategy']})"
            return f"→ TYPE FAILED: {result['error']}"

        elif verb == "scroll":
            from mcp_tools import mcp_scroll
            result = await mcp_scroll(600)
            return "→ OK" if result["success"] else f"→ FAILED: {result['error']}"

        elif verb == "press_enter":
            from mcp_tools import mcp_press_enter
            result = await mcp_press_enter()
            return "→ OK" if result["success"] else f"→ FAILED: {result['error']}"

        elif verb == "wait":
            from mcp_tools import mcp_wait
            result = await mcp_wait(proposed.get("wait_ms", 800))
            return f"→ OK (waited {result.get('waited_ms', 800)}ms)"

        else:
            return f"→ UNKNOWN verb: {verb}"

    except Exception as e:
        logger.warning("Action execution failed: %s", e)
        return f"→ CRASHED: {str(e)[:100]}"
