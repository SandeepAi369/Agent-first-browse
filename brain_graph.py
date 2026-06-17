"""Brain Graph — LangGraph StateGraph Orchestration Spine for True Brain v16.0.

This is the replacement for the monolithic for-loop in advanced_agent.py.
It wires together: goal compiler, planner, perceiver, router, worker nodes,
overwatch verifier, commit/rollback/recovery nodes, and the finalizer.

Key architectural properties:
  1. TYPED STATE: BrainState flows through every node
  2. CHECKPOINTING: SqliteSaver persists state at every super-step
  3. CONDITIONAL EDGES: MoE routing + verdict-based branching
  4. CYCLES: retry/rollback/replan loops with bounded iteration
  5. AUDIT TRAIL: every step is logged and replayable

The graph does NOT spawn multiple agents. It uses a single LLM failover
chain with SPECIALIZED PROMPTS per worker node. This follows the "earn
your cost" principle: multi-agent complexity only when genuinely needed.

References:
  - LangGraph docs: StateGraph, conditional edges, checkpointing
  - MAST (arXiv 2503.13657): typed state defeats system-design failures
  - Six Sigma Agent (arXiv 2601.22290): checkpoint-every-k-steps strategy
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Ensure app imports work
sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

from brain_state import BrainState
from moe_router import route_to_worker, verdict_router
import mcp_tools

try:
    from app.logger import get_logger
    logger = get_logger("brain_graph")
except ImportError:
    logger = logging.getLogger("brain_graph")


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-Level State (initialized at runtime)
# ═══════════════════════════════════════════════════════════════════════════════

_PAGE = None
_CONTEXT = None
_FAILOVER_CHAIN = []   # full text chain — auxiliary calls (planner, PRM, judge)
_WORKER_CHAIN = []     # V24 role separation — worker action decisions (top tier only)
_VISION_CHAIN = []
_BREAKER = None
_HEALTH_TRACKER = None
_INVOKE_FN = None
_CRITIC = None
_DREAMER = None
_PRM_CRITIC = None
_SKILL_MEM = None


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Goal Compiler
# ═══════════════════════════════════════════════════════════════════════════════

async def goal_compiler_node(state: BrainState) -> dict:
    """Parse objective, set up authorization gates, extract domain."""
    objective = state.objective

    # Auto-allow domains from objective
    from execution_safety import auto_allow_from_objective
    auto_allow_from_objective(objective)

    # Extract target domain
    domain = ""
    url_match = re.search(r'https?://(?:www\.)?([\\w.-]+)', objective)
    if url_match:
        domain = url_match.group(1)

    logger.info("🎯 Goal compiled: %s", objective[:120])
    logger.info("🌐 Target domain: %s", domain or "(none)")

    return {"task_domain": domain}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Planner
# ═══════════════════════════════════════════════════════════════════════════════

async def planner_node(state: BrainState) -> dict:
    """Strategic planner — forms an APPROACH + finish-line + steps in one call.

    V18: instead of just a list of checkpoints, the planner now produces a
    StrategicPlan{strategy, success_criteria, assumptions, steps}. The strategy
    and assumptions seed the agent's persistent cognition (so it reasons WITH a
    theory rather than re-deriving one every step), and success_criteria gives
    the agent an explicit finish line (anti done-confusion).
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from cognition import StrategicPlan, merge_beliefs

    objective = state.objective

    sys_msg = SystemMessage(content=(
        "You are a senior web-automation strategist. Given an objective, produce: "
        "(1) a short overall STRATEGY/approach, (2) the SUCCESS CRITERIA describing "
        "exactly what the page shows when the task is done, (3) 2-4 testable "
        "ASSUMPTIONS, and (4) 3-5 sequential STEPS (each one browser action)."
    ))
    user_msg = HumanMessage(content=f"Objective: {objective}")

    strategy = ""
    success_criteria = ""
    assumptions: list[str] = []
    checkpoints: list[str] = []
    try:
        plan, _ = await _INVOKE_FN(
            _FAILOVER_CHAIN, [sys_msg, user_msg], StrategicPlan,
            _BREAKER, health_tracker=_HEALTH_TRACKER,
        )
        strategy = (getattr(plan, "strategy", "") or "").strip()
        success_criteria = (getattr(plan, "success_criteria", "") or "").strip()
        assumptions = [a.strip() for a in (getattr(plan, "assumptions", []) or []) if a.strip()]
        checkpoints = [s.strip() for s in (getattr(plan, "steps", []) or []) if s.strip()]
    except Exception as e:
        logger.warning("Strategic planner failed: %s — using objective as single step", e)
    if not checkpoints:
        checkpoints = [objective]

    # Build plan steps
    plan_steps = []
    for i, desc in enumerate(checkpoints[:7]):
        plan_steps.append({
            "id": i,
            "desc": desc.strip(),
            "status": "active" if i == 0 else "pending",
        })

    # Seed cognition: strategy + bounded initial beliefs (from assumptions)
    beliefs = merge_beliefs([], assumptions)
    if strategy:
        logger.info("🧭 Strategy: %s", strategy[:160])
    if success_criteria:
        logger.info("🏁 Done when: %s", success_criteria[:120])

    # Generate PRM checklist
    prm_checklist = []
    if _PRM_CRITIC:
        try:
            prm_checklist_items = await _PRM_CRITIC.generate_checklist(objective)
            prm_checklist = [{"desc": c.description, "status": "pending"}
                            for c in prm_checklist_items]
            logger.info("📋 PRM Checklist (%d items) generated", len(prm_checklist))
        except Exception as e:
            logger.warning("PRM checklist generation failed: %s", e)

    # Retrieve skill context
    skill_context = ""
    if _SKILL_MEM:
        try:
            domain = state.task_domain
            relevant = _SKILL_MEM.retrieve_relevant(objective, domain=domain)
            skill_context = _SKILL_MEM.inject_into_prompt(relevant)
            if skill_context:
                logger.info("🧠 SkillMemory: %d workflows injected", len(relevant))
        except Exception as e:
            logger.warning("SkillMemory retrieval failed: %s", e)

    logger.info("📋 Plan (%d steps): %s", len(plan_steps),
                " → ".join(s["desc"][:40] for s in plan_steps))

    return {
        "plan_steps": plan_steps,
        "plan_cursor": 0,
        "plan_progress_pct": 0,
        "prm_checklist": prm_checklist,
        "skill_context": skill_context,
        # ── Cognition seed ──
        "strategy": strategy,
        "success_criteria": success_criteria,
        "beliefs": beliefs,
        "strategy_confidence": 1.0,
        "current_obstacle": "",
        "ladder_rung": 0,
        "tried_tactics": [],
        "goal_score_window": [],
        "restrategize_count": 0,
        "goal_complete_hint": "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Perceive — DOM extraction + page FSM update
# ═══════════════════════════════════════════════════════════════════════════════

async def perceive_node(state: BrainState) -> dict:
    """Extract DOM, detect login state, update page FSM."""
    page = _PAGE

    # Check step budget
    if state.step_number >= state.max_steps:
        return {"next_node": "finalize"}

    # Wait for page load
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=10000)
        await page.wait_for_timeout(1500)
    except Exception:
        pass

    current_url = page.url
    logger.info("━━━ Step %d/%d ━━━", state.step_number + 1, state.max_steps)
    logger.info("URL: %s", current_url)

    # DOM snapshot — routed through the Adaptive Perception Engine when enabled.
    # P0: the engine is a behavior-identical Tier-1 passthrough over this same
    # mcp_snapshot call; it adds the routing seam for later deep-sweep / vision
    # tiers without changing this node. A hard fallback to the direct snapshot
    # guarantees perception can never be broken by the engine layer.
    async def _direct_snapshot() -> dict:
        return await mcp_tools.mcp_snapshot()

    snapshot = None
    try:
        from feature_flags import adaptive_perception_enabled
        if adaptive_perception_enabled():
            from perception_engine import perceive as _perceive
            pr = await _perceive(page, ctx={
                "objective": state.objective,
                "bound_target": state.bound_target,
                "step_number": state.step_number,
            })
            snapshot = {
                "elements": pr.elements, "markdown": pr.markdown,
                "selector_map": pr.selector_map, "element_count": pr.element_count,
            }
    except Exception as e:
        logger.warning("Adaptive perception failed (%s) — using direct snapshot", e)
        snapshot = None
    if snapshot is None:
        snapshot = await _direct_snapshot()

    elements_list = snapshot.get("elements", [])
    dom_markdown = snapshot.get("markdown", "")
    selector_map = snapshot.get("selector_map", {})
    element_count = snapshot.get("element_count", 0)

    # Login detection
    login_result = await mcp_tools.mcp_detect_login()
    login_detected = login_result.get("logged_in", False)
    if login_detected:
        logger.info("🔑 Login detected: profile=True")

    # URL streak tracking
    same_url_streak = state.same_url_streak
    last_url = state.last_url_for_streak
    if current_url == last_url:
        same_url_streak += 1
    else:
        same_url_streak = 0
        if _DREAMER:
            _DREAMER.clear_cache()

    # Pre-compute plan/facts/history renders
    plan_render = state.get_plan_render()
    facts_render = state.render_facts()
    history_compressed = state.compress_history()

    # ── V29 Phase 3: progress-aware stagnation (revives the previously-DEAD
    #    same_url_streak signal). "Busy but not progressing" → break the loop. ──
    stagnation_level = 0
    stagnation_note = ""
    try:
        from feature_flags import stagnation_enabled
        if stagnation_enabled():
            from stagnation import detect_stagnation
            sig = detect_stagnation({
                "same_url_streak": same_url_streak,
                "goal_score_window": state.goal_score_window,
                "loop_signatures": state.loop_signatures,
            })
            stagnation_level = sig.level
            stagnation_note = sig.note
            if sig.stuck:
                logger.warning("🔁 STAGNATION (level %d): %s", sig.level,
                               "; ".join(sig.reasons))
    except Exception as e:
        logger.debug("Stagnation check skipped (non-fatal): %s", e)

    return {
        "current_url": current_url,
        "dom_markdown": dom_markdown,
        "selector_map": selector_map,
        "elements_list": elements_list,
        "element_count": element_count,
        "login_detected": login_detected,
        "page_fsm": "READY",
        "same_url_streak": same_url_streak,
        "last_url_for_streak": current_url,
        # V29 stagnation signals (read by the guidance bus next worker step)
        "stagnation_level": stagnation_level,
        "stagnation_note": stagnation_note,
        # Pre-computed renders (avoids re-computing in workers)
        "plan_render": plan_render,
        "facts_render": facts_render,
        "history_compressed": history_compressed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Router (MoE)
# ═══════════════════════════════════════════════════════════════════════════════

def router_node(state: BrainState) -> dict:
    """MoE router — classifies the next worker to invoke."""
    # Check circuit breaker
    if _BREAKER and _BREAKER.tripped:
        logger.critical("🔌 CIRCUIT BREAKER: %s", _BREAKER.reason)
        return {"next_node": "finalize"}

    target = route_to_worker(state.model_dump())
    return {"next_node": target}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Workers (Navigator, Interactor, Extractor)
# ═══════════════════════════════════════════════════════════════════════════════

async def navigator_worker(state: BrainState) -> dict:
    """Navigation specialist — goto, scroll, wait."""
    from workers.base_worker import invoke_worker, build_system_prompt
    prompt = build_system_prompt(
        "navigator",
        plan_context=state.get_plan_render(),
        facts_context=state.render_facts(),
        skill_context=state.skill_context,
    )
    return await invoke_worker(
        state.model_dump(), prompt,
        _WORKER_CHAIN or _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER, _INVOKE_FN,
        vision_chain=_VISION_CHAIN,
        dreamer=_DREAMER,
    )


async def interactor_worker(state: BrainState) -> dict:
    """Interaction specialist — click, type, form fill."""
    from workers.base_worker import invoke_worker, build_system_prompt
    prompt = build_system_prompt(
        "interactor",
        plan_context=state.get_plan_render(),
        facts_context=state.render_facts(),
        skill_context=state.skill_context,
    )
    return await invoke_worker(
        state.model_dump(), prompt,
        _WORKER_CHAIN or _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER, _INVOKE_FN,
        vision_chain=_VISION_CHAIN,
        dreamer=_DREAMER,
    )


async def extractor_worker(state: BrainState) -> dict:
    """Data extraction specialist — read, find, capture."""
    from workers.base_worker import invoke_worker, build_system_prompt
    prompt = build_system_prompt(
        "extractor",
        plan_context=state.get_plan_render(),
        facts_context=state.render_facts(),
        skill_context=state.skill_context,
    )
    return await invoke_worker(
        state.model_dump(), prompt,
        _WORKER_CHAIN or _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER, _INVOKE_FN,
        vision_chain=_VISION_CHAIN,
        dreamer=_DREAMER,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Overwatch
# ═══════════════════════════════════════════════════════════════════════════════

async def overwatch_wrapper(state: BrainState) -> dict:
    """Wrap the Overwatch verifier subgraph into a LangGraph node."""
    from overwatch import overwatch_node
    return await overwatch_node(state.model_dump(), _PAGE, _CRITIC)


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Done Check (runs Overwatch in done-mode)
# ═══════════════════════════════════════════════════════════════════════════════

async def done_check_node(state: BrainState) -> dict:
    """Special Overwatch path for 'done' actions."""
    from overwatch import overwatch_node
    return await overwatch_node(state.model_dump(), _PAGE, _CRITIC)


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Commit — Checkpoint + Advance Plan
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_prm_audit(state: BrainState) -> tuple[float, list[dict], list[str]]:
    """Revived PRM goal-audit: score the live page against the checklist.

    Reconstructs ChecklistItem objects from the persisted dicts, scores them
    (PRMCritic mutates statuses in place), and writes the statuses back.
    Returns (goal_score 0..1, updated_checklist_dicts, newly_completed_descs).
    """
    from prm_critic import ChecklistItem

    items = [
        ChecklistItem(
            id=i,
            description=d.get("desc", ""),
            status=d.get("status", "pending"),
            confidence=float(d.get("confidence", 0.0)),
            step_completed=int(d.get("step_completed", -1)),
            verified=bool(d.get("verified", False)),   # V26: persist stickiness
            evidence=d.get("evidence", ""),
        )
        for i, d in enumerate(state.prm_checklist)
    ]
    step_score = await _PRM_CRITIC.score_step(
        checklist=items,
        dom_markdown=state.dom_markdown,
        current_url=state.current_url,
        step_number=state.step_number + 1,
    )
    updated = [
        {"desc": it.description, "status": it.status,
         "confidence": it.confidence, "step_completed": it.step_completed,
         "verified": it.verified, "evidence": it.evidence}
        for it in items
    ]
    return step_score.total_score, updated, step_score.newly_completed


async def commit_node(state: BrainState) -> dict:
    """Commit after Overwatch passes. Advance plan, extend budget if needed.

    V18: a committed step is a VERIFIED-PROGRESS step, so it reinforces strategy
    confidence and clears the transient 'stuck' state (the obstacle is resolved —
    no stale escalation directive may bleed forward). A throttled PRM goal-audit
    gives a goal-aware progress signal, detects stalls, and nudges the agent to
    finish once the goal looks achieved (anti done-confusion).
    """
    from cognition import (
        update_confidence, prm_should_audit, push_goal_score, detect_stall,
        clear_transient, DONE_CEILING,
    )

    updates = {
        "page_fsm": "READY",
        "step_number": state.step_number + 1,
        "error_count": 0,
        "retry_count": 0,
        "session_checkpoint_id": f"step_{state.step_number}",
        "proposed_action": None,
        "overwatch_verdict": "",
    }

    # Reinforce confidence (verified progress) + clear transient stuck state.
    updates["strategy_confidence"] = update_confidence(state.strategy_confidence, progress=True)
    updates.update(clear_transient())

    # Check mission success
    if state.mission_success:
        updates["next_node"] = "finalize"
        return updates

    # Plan advancement — check if current action indicates progress
    plan_steps = list(state.plan_steps)
    plan_cursor = state.plan_cursor
    advanced = False

    for i, step in enumerate(plan_steps):
        if step.get("status") in ("active", "in_progress"):
            # V17: semantic plan advancement — the action outcome must indicate
            # success AND the action should relate to the current plan step.
            # Old heuristic (`"OK" in outcome`) advanced on ANY success, causing
            # critical steps to be skipped prematurely.
            outcome = state.action_outcome or ""
            action_ok = ("OK" in outcome or "navigated" in outcome.lower()
                         or "SUCCESS" in outcome.upper())
            if action_ok:
                # Only advance if the action plausibly matches this plan step.
                # We check overlap between action context and step description.
                step_desc = step.get("desc", "").lower()
                action_ctx = " ".join([
                    (state.proposed_action or {}).get("verb", ""),
                    (state.proposed_action or {}).get("target_name", ""),
                    (state.proposed_action or {}).get("text", "") or "",
                    outcome,
                ]).lower()

                # Advance if any meaningful keyword from the step appears in
                # the action context, OR if the step is a generic navigation
                # step that any successful action can satisfy.
                step_words = {w for w in step_desc.split() if len(w) > 3}
                overlap = step_words & {w for w in action_ctx.split() if len(w) > 3}
                is_generic = any(kw in step_desc for kw in
                                 ("navigate", "open", "go to", "visit", "load"))

                if overlap or is_generic:
                    plan_steps[i] = {**step, "status": "done"}
                    # Activate next pending step
                    for j in range(i + 1, len(plan_steps)):
                        if plan_steps[j].get("status") == "pending":
                            plan_steps[j] = {**plan_steps[j], "status": "active"}
                            plan_cursor = j
                            break
                    advanced = True
            break

    if advanced:
        done_count = sum(1 for s in plan_steps if s.get("status") == "done")
        progress_pct = int(done_count / len(plan_steps) * 100) if plan_steps else 0
        updates["plan_steps"] = plan_steps
        updates["plan_cursor"] = plan_cursor
        updates["plan_progress_pct"] = progress_pct

        # Dynamic budget extension
        if state.step_number >= state.max_steps - 3 and progress_pct >= 60:
            updates["max_steps"] = state.max_steps + 5
            updates["budget_extended"] = True
            logger.info("📊 Budget extended to %d (progress=%d%%)",
                        updates["max_steps"], progress_pct)

    # ── Throttled PRM goal-audit (revived): goal-aware progress + stall + done-nudge ──
    if _PRM_CRITIC and state.prm_checklist and prm_should_audit(state.step_number, True):
        try:
            goal_score, updated_checklist, newly = await _run_prm_audit(state)
            updates["prm_checklist"] = updated_checklist
            window = push_goal_score(state.goal_score_window, goal_score)
            updates["goal_score_window"] = window

            # V26: make the SUCCESS verification first-class in the logs (was only
            # the struggle that showed up). Each newly-verified sub-goal is logged
            # with its captured on-page proof and marked sticky.
            prev_verified = {d.get("desc") for d in state.prm_checklist if d.get("verified")}
            for d in updated_checklist:
                if d.get("verified") and d.get("desc") not in prev_verified:
                    logger.info("✅ VERIFIED [%s] — %s",
                                (d.get("desc") or "")[:50],
                                (d.get("evidence") or "state change confirmed")[:120])

            # Done-ceiling nudge — reduce post-completion wandering.
            # CoVe (Overwatch L4) still guards against a PREMATURE done.
            if goal_score >= DONE_CEILING:
                updates["goal_complete_hint"] = (
                    "\n\n✅ GOAL APPEARS COMPLETE (checklist ~satisfied). Verify your "
                    "DONE WHEN criteria on screen and output action_type='done' unless "
                    "something is clearly still missing. Do NOT keep acting once done."
                )
            else:
                updates["goal_complete_hint"] = ""

            # Stall: DOM keeps changing (we committed) but goal-score is flat →
            # busy-but-not-progressing → penalize confidence to provoke a revision.
            if detect_stall(window):
                penalized = update_confidence(updates["strategy_confidence"], progress=False)
                logger.info("📉 Goal-progress STALL (score window flat) — confidence %.2f→%.2f",
                            updates["strategy_confidence"], penalized)
                updates["strategy_confidence"] = penalized
        except Exception as e:
            logger.debug("PRM audit error (non-fatal): %s", e)

    # ── V17 Win-State Recognizer (generalized — zero hardcoded platforms) ──
    # If the agent just executed an IRREVERSIBLE action (as classified by
    # action_classifier: add-to-cart, submit, fork, star, post, etc.) AND the
    # page responded with a significant state change (URL changed or large DOM
    # delta), this is the cognitive signal that the CRITICAL task action just
    # completed. Inject a strong nudge to call 'done' immediately — preventing
    # the agent from wandering into post-action pages (cart recommendations,
    # confirmation screens, etc.) and clicking similar-looking buttons.
    #
    # This is NOT hardcoding any platform. The action_classifier already knows
    # what "critical" looks like from its regex patterns (buy, submit, add to
    # cart, delete, post, etc.) — all task-agnostic. The URL change is a
    # universal signal that the action had a real effect.
    action = state.proposed_action or {}
    if (action.get("risk_level") == "IRREVERSIBLE"
            and state.action_outcome
            and "OK" in state.action_outcome):
        # The critical action succeeded — check if the page reacted
        outcome_lower = state.action_outcome.lower()
        page_reacted = ("navigated" in outcome_lower
                        or "structure changed" in outcome_lower
                        or "element count changed" in outcome_lower)
        if page_reacted:
            logger.info(
                "🏆 WIN-STATE: IRREVERSIBLE action '%s' on '%s' succeeded with "
                "page reaction — nudging agent to verify and finish.",
                action.get("verb", "?"), action.get("target_name", "?")[:40],
            )
            updates["goal_complete_hint"] = (
                "\n\n🏆 CRITICAL ACTION COMPLETED: You just performed the KEY "
                "irreversible action for this task and the page confirmed it "
                "(URL/DOM changed). Your task is very likely DONE. Verify the "
                "result on screen and output action_type='done' IMMEDIATELY. "
                "Do NOT click any other buttons, links, or recommendations. "
                "Do NOT navigate away. Just verify and finish."
            )

    return updates


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Retry
# ═══════════════════════════════════════════════════════════════════════════════

async def retry_node(state: BrainState) -> dict:
    """Retry after a soft failure (grounding reject, no progress)."""
    return {
        "retry_count": state.retry_count + 1,
        "proposed_action": None,  # Clear for re-evaluation
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Rollback
# ═══════════════════════════════════════════════════════════════════════════════

async def rollback_node(state: BrainState) -> dict:
    """Hard-stuck handler. The escalation ladder (in Overwatch) has already tried
    the cheap tactics; rollback decides whether to RE-STRATEGIZE.

    V18: instead of a full replan (which wiped progress), rollback either
      (a) RE-STRATEGIZES — one LLM call for a genuinely different approach,
          updating `strategy` + adding the lesson as a belief, resetting the
          ladder, then continuing (→ perceive); or
      (b) lets the ladder's current directive stand and continues (→ perceive).
    Bounded by MAX_RESTRATEGIZE; the existing error_count→recovery→finalize path
    still guarantees termination.
    """
    from cognition import (
        restrategize, needs_restrategy, merge_beliefs,
        RESTRATEGIZE_TACTIC, LADDER,
    )

    updates: dict = {
        "error_count": state.error_count + 1,
        "page_fsm": "READY",
        "proposed_action": None,
        "retry_count": 0,
        "next_node": "perceive",  # continue — no destructive full replan
    }

    ladder_exhausted = (
        RESTRATEGIZE_TACTIC in state.tried_tactics
        or state.ladder_rung >= len(LADDER)
    )
    do_restrategy = ladder_exhausted or needs_restrategy(
        state.strategy_confidence, state.restrategize_count
    )

    if do_restrategy:
        logger.warning("🔄 ROLLBACK → RE-STRATEGIZE (#%d, confidence=%.2f)",
                       state.restrategize_count + 1, state.strategy_confidence)
        new_strategy, lesson = await restrategize(
            _INVOKE_FN, _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER,
            objective=state.objective,
            current_strategy=state.strategy,
            beliefs=state.beliefs,
            current_url=state.current_url,
            recent_failure=state.action_outcome,
            plan_render=state.get_plan_render(),
        )
        if new_strategy:
            updates["strategy"] = new_strategy
            updates["strategy_confidence"] = 0.6  # fresh hypothesis — moderate prior
            updates["restrategize_count"] = state.restrategize_count + 1
            if lesson:
                updates["beliefs"] = merge_beliefs(state.beliefs, [lesson])
                updates["reflections"] = [f"Re-strategized: {lesson[:120]}"]
        # Reset the ladder for the new strategy + clear stale directives
        updates["current_obstacle"] = ""
        updates["ladder_rung"] = 0
        updates["tried_tactics"] = []
        updates["correction_context"] = ""
        updates["recovery_advice"] = ""
    else:
        # Ladder still has untried rungs — keep Overwatch's directive, continue.
        logger.warning("🔄 ROLLBACK at step %d — continuing with next escalation tactic",
                       state.step_number)
        updates["reflections"] = [
            f"Step {state.step_number} stuck (verdict={state.overwatch_verdict}); "
            "escalating tactic."
        ]

    return updates


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Recovery
# ═══════════════════════════════════════════════════════════════════════════════

async def recovery_node(state: BrainState) -> dict:
    """Recovery after too many errors — reset counters and continue."""
    logger.warning("🔧 Recovery node: resetting error state after %d errors", state.error_count)
    return {
        "error_count": 0,
        "correction_failures": 0,
        "retry_count": 0,
        "recovery_advice": "Agent recovered. Trying fresh approach.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Finalize
# ═══════════════════════════════════════════════════════════════════════════════

async def finalize_node(state: BrainState) -> dict:
    """Clean shutdown — record workflow, close browser."""
    mission_success = state.mission_success
    done_evidence = state.done_evidence

    # ── V20 last-chance outcome audit: shutting down WITHOUT a verified done
    #    (budget exhausted etc.) — check the final page once before reporting
    #    failure. The goal may be achieved even though 'done' was never accepted.
    if not mission_success and not done_evidence and _PAGE is not None:
        try:
            from overwatch import final_outcome_audit
            achieved, evidence = await final_outcome_audit(state.model_dump(), _PAGE)
            done_evidence = evidence
            if achieved:
                mission_success = True
                logger.info("⚖️ Final audit: goal evidence found on shutdown — %s",
                            evidence[:160])
            else:
                logger.info("⚖️ Final audit: goal NOT confirmed — %s", evidence[:160])
        except Exception as e:
            logger.debug("Final outcome audit failed (non-fatal): %s", e)

    # Record in SkillMemory
    if _SKILL_MEM and state.history:
        try:
            _SKILL_MEM.record_workflow(
                objective=state.objective,
                steps=[dict(h) for h in state.history],
                success=mission_success,
                total_steps=state.step_number,
            )
            logger.info("📝 Workflow recorded in SkillMemory (success=%s)", mission_success)
        except Exception as e:
            logger.warning("SkillMemory record failed: %s", e)

    # Record in CampaignMemory
    try:
        from campaign_memory import CampaignMemory
        memory = CampaignMemory()
        memory.record_post(
            platform="web",
            title=state.objective[:120],
            content=state.objective,
            url=state.current_url,
            agent_model="brain_v16",
            steps_taken=state.step_number,
        )
    except Exception as e:
        logger.warning("CampaignMemory record failed: %s", e)

    if mission_success:
        logger.info("✅ MISSION COMPLETE — Task finished successfully!")
        if done_evidence:
            logger.info("⚖️ Verified outcome: %s", done_evidence[:200])
    else:
        logger.info("⚠️ Agent shutting down (mission incomplete)")
        if done_evidence:
            logger.info("⚖️ Outcome status: %s", done_evidence[:200])

    # ── Clean handoff: reset stateful singletons so a fresh task starts clean ──
    # (BrainState is per-task already; these objects are reused if the process is.)
    try:
        if _CRITIC and hasattr(_CRITIC, "reset_for_task"):
            _CRITIC.reset_for_task()
        if _PRM_CRITIC is not None:
            _PRM_CRITIC._prev_score = 0.0
    except Exception as e:
        logger.debug("Cognitive reset on finalize failed (non-fatal): %s", e)

    # Clear cognitive state in the returned state for a tidy final snapshot.
    from cognition import clear_all
    # Persist the (possibly audit-corrected) outcome into the final state so the
    # runner's summary reports the verified result, not the pre-audit one.
    return {**clear_all(),
            "mission_success": mission_success,
            "done_evidence": done_evidence}


# ═══════════════════════════════════════════════════════════════════════════════
#  Graph Construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_brain_graph() -> StateGraph:
    """Construct and return the compiled LangGraph StateGraph."""
    g = StateGraph(BrainState)

    # ── Add Nodes ──
    g.add_node("goal_compiler", goal_compiler_node)
    g.add_node("planner", planner_node)
    g.add_node("perceive", perceive_node)
    g.add_node("router", router_node)
    g.add_node("navigator", navigator_worker)
    g.add_node("interactor", interactor_worker)
    g.add_node("extractor", extractor_worker)
    g.add_node("overwatch", overwatch_wrapper)
    g.add_node("done_check", done_check_node)
    g.add_node("commit", commit_node)
    g.add_node("retry", retry_node)
    g.add_node("rollback", rollback_node)
    g.add_node("recovery", recovery_node)
    g.add_node("finalize", finalize_node)

    # ── Linear Path: START → goal_compiler → planner → perceive ──
    g.add_edge(START, "goal_compiler")
    g.add_edge("goal_compiler", "planner")
    g.add_edge("planner", "perceive")

    # ── perceive → router ──
    g.add_edge("perceive", "router")

    # ── Router → workers (conditional) ──
    g.add_conditional_edges("router", lambda s: s.next_node, {
        "navigator": "navigator",
        "interactor": "interactor",
        "extractor": "extractor",
        "done_check": "done_check",
        "recovery": "recovery",
        "finalize": "finalize",
    })

    # ── All workers → overwatch ──
    g.add_edge("navigator", "overwatch")
    g.add_edge("interactor", "overwatch")
    g.add_edge("extractor", "overwatch")

    # ── Overwatch → conditional verdict ──
    g.add_conditional_edges("overwatch", lambda s: verdict_router(s.model_dump()), {
        "commit": "commit",
        "retry": "retry",
        "rollback": "rollback",
        "finalize": "finalize",
    })

    # ── Done check → conditional verdict ──
    g.add_conditional_edges("done_check", lambda s: verdict_router(s.model_dump()), {
        "commit": "commit",
        "retry": "retry",
        "rollback": "rollback",
        "finalize": "finalize",
    })

    # ── Commit → perceive (next step cycle) ──
    def commit_router(state: BrainState) -> str:
        if state.mission_success:
            return "finalize"
        if state.step_number >= state.max_steps:
            return "finalize"
        return "perceive"

    g.add_conditional_edges("commit", commit_router, {
        "perceive": "perceive",
        "finalize": "finalize",
    })

    # ── Retry → perceive (re-perceive and re-route) ──
    g.add_edge("retry", "perceive")

    # ── Rollback → perceive (V18: continue with the re-strategy / next tactic;
    #    no destructive full replan that wipes plan progress) ──
    g.add_conditional_edges("rollback", lambda s: s.next_node or "perceive", {
        "perceive": "perceive",
        "finalize": "finalize",
    })

    # ── Recovery → perceive ──
    g.add_edge("recovery", "perceive")

    # ── Finalize → END ──
    g.add_edge("finalize", END)

    return g


# ═══════════════════════════════════════════════════════════════════════════════
#  Runner — Initialize modules and execute the graph
# ═══════════════════════════════════════════════════════════════════════════════

async def run_brain(objective: str, headless: bool = False):
    """Initialize all modules and execute the brain graph."""
    global _PAGE, _CONTEXT, _FAILOVER_CHAIN, _WORKER_CHAIN, _VISION_CHAIN, _BREAKER, _HEALTH_TRACKER
    global _INVOKE_FN, _CRITIC, _DREAMER, _PRM_CRITIC, _SKILL_MEM

    # ── Import existing modules ──
    from advanced_agent import (
        launch_browser, SessionGuard, _invoke_with_failover,
    )
    from model_registry import ModelRegistry
    from app.browser_promoter.browser_warmup import run_warmup, extract_target_url_from_objective
    from app.browser_promoter.worker_planner import ReasoningAgent
    from orchestrator.critic_v12 import CriticV12
    from prm_critic import PRMCritic
    from web_dreamer import WebDreamer
    from skill_memory import SkillMemory

    # ── Display mode: prefer a REAL headed browser under a (virtual) display —
    #    far less bot-detectable than --headless (no "HeadlessChrome" UA, real
    #    window/compositor). Falls back to --headless if Xvfb isn't available.
    #    Pass --headless to force true headless. ──
    from virtual_display import start_virtual_display
    launch_headless = headless
    if not headless:
        if start_virtual_display(1920, 1080):
            launch_headless = False
        else:
            launch_headless = True  # no display available → headless fallback
    logger.info("🌐 Browser mode: %s", "HEADLESS" if launch_headless else "HEADED (stealth)")

    # ── Launch Browser ──
    _CONTEXT, _PAGE = await launch_browser(headless=launch_headless)
    guard = SessionGuard.get()

    # ── Warmup ──
    target_hint = extract_target_url_from_objective(objective)
    try:
        await run_warmup(_PAGE, target_url=target_hint)
    except Exception as e:
        logger.warning("Warmup failed (non-fatal): %s", e)

    # ── ModelRegistry ──
    registry = ModelRegistry.get_instance()
    _BREAKER = registry.breaker
    _HEALTH_TRACKER = registry.health

    # V17.0: Probe once — prune dead models (404/401), seed latency estimates.
    # Must run BEFORE ReasoningAgent() snapshots the chain.
    try:
        await registry.probe_and_prune()
    except Exception as e:
        logger.warning("Model probe failed (non-fatal): %s", e)

    reasoning_agent = ReasoningAgent()
    _FAILOVER_CHAIN = reasoning_agent.get_failover_chain()
    chain_names = reasoning_agent.get_chain_names()
    _INVOKE_FN = _invoke_with_failover

    if not _FAILOVER_CHAIN:
        logger.error("No LLM clients available. Check API keys in .env.")
        await _CONTEXT.close()
        guard.detach()
        return

    logger.info("LLM Failover Chain (%d models): %s",
                len(_FAILOVER_CHAIN), " → ".join(chain_names))

    # ── V24 role separation: the worker (action decisions, the critical path)
    #    uses only the top-tier capable models; auxiliary calls use the full chain.
    _WORKER_CHAIN = registry.get_worker_chain()
    logger.info("🎯 [%s mode] Worker chain (%d top models): %s",
                registry.mode, len(_WORKER_CHAIN),
                " → ".join(registry.get_worker_chain_names()))

    # ── V29 Cognitive Overhaul — log the active feature switches (auditability) ──
    try:
        from feature_flags import active_flags, v29_enabled
        flags = active_flags()
        if v29_enabled():
            on = [k for k, v in flags.items() if v and k != "V29_ENABLED"]
            logger.info("🧬 V29 Cognitive Overhaul ACTIVE — %s", ", ".join(on) or "(master only)")
        else:
            logger.info("🧬 V29 disabled (V29_ENABLED=0) — running pure V28 behavior")
    except Exception as e:
        logger.debug("V29 flag log skipped (non-fatal): %s", e)

    # Clear any durable intent ledger left behind by a previously-crashed run, so a
    # stale write-ahead record can never contaminate this fresh task's audit trail.
    try:
        from intent_journal import resolve_intent
        resolve_intent()
    except Exception:
        pass

    # ── V21 Vision chain (for on-demand consults; a11y DOM stays the default) ──
    try:
        _VISION_CHAIN = registry.get_vision_chain()
        if _VISION_CHAIN:
            logger.info("👁️ Vision chain ready for on-demand consults (%d models): %s",
                        len(_VISION_CHAIN), " → ".join(registry.get_vision_chain_names()))
        else:
            logger.info("👁️ No vision models configured — agent runs a11y-DOM only")
    except Exception as e:
        _VISION_CHAIN = []
        logger.warning("Vision chain unavailable (a11y-DOM only): %s", e)

    # ── MCP Tools ──
    mcp_tools.set_page(_PAGE)

    # ── CriticV12 ──
    _CRITIC = CriticV12(_PAGE)
    logger.info("🧠 CriticV12 initialized")

    # ── WebDreamer ──
    try:
        _DREAMER = WebDreamer(
            invoke_fn=_INVOKE_FN,
            failover_chain=_FAILOVER_CHAIN,
            breaker=_BREAKER,
            health_tracker=_HEALTH_TRACKER,
            num_candidates=3,
            num_simulations=1,
        )
        logger.info("🌙 WebDreamer initialized")
    except Exception as e:
        logger.warning("WebDreamer init failed: %s", e)

    # ── PRM Critic ──
    try:
        _PRM_CRITIC = PRMCritic(_INVOKE_FN, _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER)
        logger.info("📊 PRMCritic initialized")
    except Exception as e:
        logger.warning("PRMCritic init failed: %s", e)

    # ── V20 Outcome Judge (evidence-grounded done-gate) ──
    try:
        from overwatch import configure_outcome_judge
        configure_outcome_judge(_INVOKE_FN, _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER)
        logger.info("⚖️ Outcome judge wired into the done-gate")
    except Exception as e:
        logger.warning("Outcome judge config failed (heuristic fallback): %s", e)

    # ── SkillMemory ──
    try:
        _SKILL_MEM = SkillMemory()
        stats = _SKILL_MEM.get_stats()
        logger.info("🧠 SkillMemory: %d workflows, %d reliable",
                    stats['total_workflows'], stats['reliable_workflows'])
    except Exception as e:
        logger.warning("SkillMemory init failed: %s", e)

    # ── Build Graph ──
    graph_builder = build_brain_graph()

    # ── Compile with Checkpointer ──
    db_path = Path(__file__).parent / "persistence" / "brain_checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        brain = graph_builder.compile(checkpointer=checkpointer)

        logger.info("🧠 Brain Graph compiled with %d nodes, checkpointer=%s",
                    len(graph_builder.nodes), type(checkpointer).__name__)

        # ── Execute ──
        config = {"configurable": {"thread_id": f"brain_{int(time.time())}"}}
        initial_state = BrainState(objective=objective)

        try:
            # Latch mission_success across the stream: each event is ONE node's
            # partial output (finalize returns {}), so reading only the last
            # event misses the success flag set earlier by overwatch/commit.
            mission_success = False
            async for event in brain.astream(initial_state, config):
                # Log each node execution
                for node_name, node_output in event.items():
                    if node_name != "__end__":
                        logger.debug("Node '%s' → %d updates",
                                    node_name, len(node_output) if isinstance(node_output, dict) else 0)
                    if isinstance(node_output, dict) and node_output.get("mission_success"):
                        mission_success = True

            if mission_success:
                print("\n" + "═" * 60)
                print("  ✅  MISSION COMPLETE — Task finished successfully!")
                print("═" * 60 + "\n")
            else:
                print("\n" + "═" * 60)
                print("  ⚠️  Mission incomplete — ran out of steps or was interrupted.")
                print("═" * 60 + "\n")

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt — saving session...")
        except Exception as e:
            import traceback
            logger.error("💥 Brain crash: %s", e)
            logger.error(traceback.format_exc())
        finally:
            try:
                await _CONTEXT.close()
                logger.info("Browser closed. Profile state persisted.")
            except Exception as e:
                logger.warning("Browser close failed: %s", e)
            guard.detach()
            try:
                from virtual_display import stop_virtual_display
                stop_virtual_display()
            except Exception:
                pass
