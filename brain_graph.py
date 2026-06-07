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

logger = logging.getLogger("brain_graph")


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-Level State (initialized at runtime)
# ═══════════════════════════════════════════════════════════════════════════════

_PAGE = None
_CONTEXT = None
_FAILOVER_CHAIN = []
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
    """Decompose objective into sequential checkpoints using LLM."""
    objective = state.objective

    prompt = (
        "Break this browser automation objective into 3-5 short sequential checkpoints. "
        "Each checkpoint is ONE specific browser action (navigate, click, type, verify). "
        "Return ONLY a JSON array of strings. No markdown, no explanation.\n\n"
        f"Objective: {objective}"
    )

    from langchain_core.messages import HumanMessage
    try:
        response, _ = await _INVOKE_FN(
            _FAILOVER_CHAIN,
            [HumanMessage(content=prompt)],
            None, _BREAKER, health_tracker=_HEALTH_TRACKER,
        )
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        checkpoints = json.loads(text)
        if not isinstance(checkpoints, list):
            checkpoints = [objective]
    except Exception as e:
        logger.warning("Planner failed: %s — using objective as single step", e)
        checkpoints = [objective]

    # Build plan steps
    plan_steps = []
    for i, desc in enumerate(checkpoints[:7]):
        plan_steps.append({
            "id": i,
            "desc": desc.strip(),
            "status": "active" if i == 0 else "pending",
        })

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

    # DOM snapshot via MCP
    snapshot = await mcp_tools.mcp_snapshot()
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
        _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER, _INVOKE_FN,
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
        _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER, _INVOKE_FN,
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
        _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER, _INVOKE_FN,
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

async def commit_node(state: BrainState) -> dict:
    """Commit after Overwatch passes. Advance plan, extend budget if needed."""
    updates = {
        "page_fsm": "READY",
        "step_number": state.step_number + 1,
        "error_count": 0,
        "retry_count": 0,
        "session_checkpoint_id": f"step_{state.step_number}",
        "proposed_action": None,
        "overwatch_verdict": "",
    }

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
            # Check if step is done based on action outcome
            outcome = state.action_outcome or ""
            if "OK" in outcome or "navigated" in outcome.lower():
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
    """Rollback forces a replan via the planner node.

    Adds a reflexion note explaining why the rollback occurred.
    """
    reflection = (
        f"Step {state.step_number} failed (verdict={state.overwatch_verdict}). "
        f"Outcome: {state.action_outcome or 'unknown'}. "
        "Need a completely different approach."
    )

    logger.warning("🔄 ROLLBACK: %s", reflection[:100])

    # Call Recovery Advisor for guidance
    recovery_advice = ""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        prompt = (
            "You are a browser automation RECOVERY CONSULTANT. The agent is STUCK. "
            "Analyze the situation and give ONE specific actionable instruction.\n\n"
            f"OBJECTIVE: {state.objective}\nURL: {state.current_url}\n"
            f"LAST FAILURE: {state.action_outcome}\n\n"
            "Respond in 1-2 sentences ONLY."
        )
        response, _ = await _INVOKE_FN(
            _FAILOVER_CHAIN,
            [HumanMessage(content=prompt)],
            None, _BREAKER, health_tracker=_HEALTH_TRACKER,
        )
        content_val = response.content if hasattr(response, "content") else ""
        if isinstance(content_val, list):
            texts = [b.get("text", "") for b in content_val if isinstance(b, dict) and b.get("type") == "text"]
            content_val = " ".join(texts)
        elif not isinstance(content_val, str):
            content_val = str(content_val)
        recovery_advice = content_val[:200]
        logger.info("🛠️ Recovery advice: %s", recovery_advice[:150])
    except Exception as e:
        logger.warning("Recovery advisor failed: %s", e)
        recovery_advice = "Try scrolling down or dismissing any visible popups."

    return {
        "reflections": [reflection],
        "error_count": state.error_count + 1,
        "page_fsm": "READY",
        "proposed_action": None,
        "retry_count": 0,
        "recovery_advice": recovery_advice,
    }


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
    else:
        logger.info("⚠️ Agent shutting down (mission incomplete)")

    return {}


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

    # ── Rollback → planner (replan) ──
    g.add_edge("rollback", "planner")

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
    global _PAGE, _CONTEXT, _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER
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

    # ── Launch Browser ──
    _CONTEXT, _PAGE = await launch_browser(headless=headless)
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
            final_state = None
            async for event in brain.astream(initial_state, config):
                # Log each node execution
                for node_name, node_output in event.items():
                    if node_name != "__end__":
                        logger.debug("Node '%s' → %d updates",
                                    node_name, len(node_output) if isinstance(node_output, dict) else 0)
                    if isinstance(node_output, dict):
                        final_state = node_output

            # ── Final Report ──
            mission_success = False
            if final_state:
                mission_success = final_state.get("mission_success", False)

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
