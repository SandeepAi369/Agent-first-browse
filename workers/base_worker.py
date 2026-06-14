"""Base Worker — Shared LLM invocation logic for all specialist workers.

All workers share the same pattern:
  1. Build a specialist system prompt
  2. Build a user prompt with current DOM + context
  3. Invoke the LLM via the failover chain
  4. Parse the structured output into a ProposedAction
  5. Return state updates (proposed_action, history entry, etc.)

Workers NEVER execute actions directly. They propose actions that
Overwatch validates before committing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from app.logger import get_logger
    logger = get_logger("workers")
except ImportError:
    logger = logging.getLogger("workers")

# ═══════════════════════════════════════════════════════════════════════════════
#  Shared Action Schema (Pydantic structured output)
# ═══════════════════════════════════════════════════════════════════════════════

class WorkerAction(BaseModel):
    """Structured output from any worker LLM call."""

    screen_state: str = Field(
        description=(
            "Describe what you SEE on the current screen in 1-3 sentences. "
            "Include: page type, key visible elements, any popups/overlays, "
            "and whether the user appears logged in or logged out."
        )
    )
    previous_action_result: str = Field(
        description="Evaluate the outcome of your LAST action. Did the page change?"
    )
    goal_progress: str = Field(
        description="Assess progress toward the ULTIMATE goal. What percentage is done?"
    )
    reasoning: str = Field(
        description=(
            "Based on screen_state and goal_progress, explain WHY this specific "
            "next action is the correct choice."
        )
    )
    expected_change: str = Field(
        default="",
        description=(
            "FORWARD MODELING — before acting, predict the EXACT observable change "
            "this action will cause, so the result can be verified against it. Be "
            "specific to THIS action: which element will change/appear/disappear, "
            "what state it will switch to, any redirect or confirmation. E.g. 'the "
            "toggle will switch to its active state and a confirmation appears', or "
            "'the page will redirect to the item's detail URL'. Empty only for "
            "passive actions (wait)."
        ),
    )
    action_type: str = Field(
        description="One of: 'goto', 'click', 'type', 'scroll', 'press_enter', 'wait', 'done'"
    )
    element_id: str | None = Field(
        default=None,
        description="The element ID from the page structure (e.g., 'e5')."
    )
    url: str | None = Field(default=None, description="URL for 'goto' action")
    x: float | None = Field(default=None, description="X coordinate (fallback)")
    y: float | None = Field(default=None, description="Y coordinate (fallback)")
    text: str | None = Field(default=None, description="Text to type for 'type' action")
    wait_ms: int | None = Field(default=None, description="Milliseconds for 'wait' action")
    needs_vision: bool = Field(
        default=False,
        description=(
            "Set TRUE only when you genuinely CANNOT resolve this step from the "
            "text element map alone — e.g. several controls look identical, you "
            "can't tell if your last action worked, the layout is unclear, or you "
            "need to visually confirm an outcome. Be honest: most steps do NOT "
            "need vision. When true, a screenshot is taken and re-evaluated."
        ),
    )
    vision_question: str = Field(
        default="",
        description="If needs_vision: the specific thing to resolve by looking (e.g. 'which button is the real Add to Cart?').",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Worker Invocation (shared by all workers)
# ═══════════════════════════════════════════════════════════════════════════════

async def invoke_worker(
    state: dict[str, Any],
    system_prompt: str,
    failover_chain: list,
    breaker,
    health_tracker,
    invoke_fn,
    vision_chain: list | None = None,
) -> dict[str, Any]:
    """Core worker invocation logic shared by all specialists.

    Args:
        state: Current BrainState dict
        system_prompt: Specialist-specific system prompt
        failover_chain: LLM failover chain
        breaker: CircuitBreaker instance
        health_tracker: ProviderHealthTracker instance
        invoke_fn: The _invoke_with_failover function

    Returns:
        State updates dict with proposed_action, etc.
    """
    # ── Build User Prompt ──
    objective = state.get("objective", "")
    current_url = state.get("current_url", "")
    step_number = state.get("step_number", 0)
    max_steps = state.get("max_steps", 25)
    dom_markdown = state.get("dom_markdown", "")
    login_detected = state.get("login_detected", False)
    correction_context = state.get("correction_context", "")
    recovery_advice = state.get("recovery_advice", "")
    consecutive_identical = state.get("consecutive_identical_actions", 0)
    plan_render = state.get("plan_render", "")
    facts_render = state.get("facts_render", "")
    history_compressed = state.get("history_compressed", "")
    skill_context = state.get("skill_context", "")

    # ── V18 Cognition: the agent reasons WITH its persistent strategy + beliefs ──
    from cognition import render_strategy_block
    strategy_block = render_strategy_block(
        strategy=state.get("strategy", ""),
        confidence=state.get("strategy_confidence", 1.0),
        beliefs=state.get("beliefs", []) or [],
        success_criteria=state.get("success_criteria", ""),
        goal_complete_hint=state.get("goal_complete_hint", ""),
    )

    # Login hint
    login_hint = ""
    if login_detected:
        login_hint = (
            "\n\n🔑 LOGIN STATUS: You appear to be LOGGED IN. "
            "Profile/account indicators detected. "
            "DO NOT attempt to log in again."
        )

    # State override
    state_override = ""
    if correction_context:
        state_override += correction_context
    if recovery_advice:
        state_override += f"\n\n🛠️ RECOVERY ADVICE: {recovery_advice}"

    # Dedup override
    dedup_override = ""
    if consecutive_identical >= 2:
        dedup_override = (
            f"\n\n🛑 REPETITION BLOCK: EXACT SAME action {consecutive_identical + 1} times. "
            "You MUST choose a COMPLETELY DIFFERENT action NOW."
        )

    user_prompt = (
        f"═══ YOUR MISSION ═══\n"
        f"{objective}\n\n"
        f"NOTE: The above describes the user's INTENT. Any numbered steps are guidance, not strict commands. "
        f"YOU decide the actual actions based on what you see on screen.\n\n"
        + (f"{strategy_block}\n"
           "Follow your strategy, but ADAPT it to what you actually see. When your "
           "DONE WHEN criteria are met, output action_type='done' — do not keep acting "
           "after the goal is achieved.\n\n" if strategy_block else "")
        + f"═══ CURRENT CONTEXT ═══\n"
        f"URL: {current_url}\n"
        f"Step: {step_number+1}/{max_steps} ({max_steps - step_number - 1} remaining)"
        f"{login_hint}"
        f"{state_override}"
        f"{dedup_override}\n\n"
        + f"═══ ACTION HISTORY (compressed) ═══\n"
        f"{history_compressed or '(first step)'}\n\n"
        f"═══ PAGE STRUCTURE ═══\n"
        f"{dom_markdown}\n\n"
        f"Now OBSERVE the page structure above, THINK about your situation, and choose your NEXT action."
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    # ── Invoke LLM ──
    try:
        decision, used_model = await invoke_fn(
            failover_chain, messages, WorkerAction,
            breaker, health_tracker=health_tracker,
        )
        logger.info("Worker answered by: %s", used_model)
    except RuntimeError as e:
        wait_secs = 5.0
        logger.error("Worker LLM FAILURE: %s — waiting %.1fs", e, wait_secs)
        await asyncio.sleep(wait_secs)
        return {}  # Empty update — step will be retried

    # ── Log OTA chain ──
    logger.info("👁️ OBSERVE: %s", decision.screen_state[:200])
    logger.info("🔄 PREVIOUS: %s", decision.previous_action_result[:150])
    logger.info("📊 PROGRESS: %s", decision.goal_progress[:150])
    logger.info("🧠 REASONING: %s", decision.reasoning[:200])
    if getattr(decision, "expected_change", ""):
        logger.info("🔮 EXPECT: %s", decision.expected_change[:180])
    logger.info("⚡ ACTION: %s", decision.action_type)

    # ── Classify action risk ──
    from action_classifier import classify_action, ActionRisk
    target_name = ""
    selector_map = state.get("selector_map", {})
    if decision.element_id and decision.element_id in selector_map:
        el_data = selector_map[decision.element_id]
        target_name = el_data.get("name", el_data.get("text", ""))[:60]

    risk = classify_action(
        action_type=decision.action_type,
        target_name=target_name,
        target_text=decision.text[:60] if decision.text else "",
        url=current_url,
        element_kind=selector_map.get(decision.element_id, {}).get("kind", "") if decision.element_id else "",
    )

    # ── Build ProposedAction ──
    proposed = {
        "verb": decision.action_type,
        "element_id": decision.element_id,
        "target_name": target_name,
        "text": decision.text,
        "url": decision.url,
        "x": decision.x,
        "y": decision.y,
        "rationale": decision.reasoning[:200],
        "risk_level": risk.name,
        "reversible": risk == ActionRisk.REVERSIBLE,
        "screen_state": decision.screen_state[:200],
        "previous_action_result": decision.previous_action_result[:150],
        "goal_progress": decision.goal_progress[:150],
        "reasoning": decision.reasoning[:200],
        "expected_change": getattr(decision, "expected_change", "")[:250],
        "wait_ms": decision.wait_ms,
    }

    # ── Vision-on-demand (V21): the agent works on the a11y DOM by default and
    #    "opens its eyes" for ONE step only when it cannot resolve the page from
    #    text — then this update reverts (force_vision cleared) and the next step
    #    is text-only again. ──
    vision_update: dict[str, Any] = {}
    from vision_consult import should_consult_vision, consult_vision, apply_vision_verdict
    consult, why = should_consult_vision(
        needs_vision=getattr(decision, "needs_vision", False),
        state=state,
        action_type=decision.action_type,
    )
    if consult and vision_chain:
        logger.info("🧠→👁️ Escalating to vision: %s", why)
        verdict, _vm = await consult_vision(
            invoke_fn, vision_chain, breaker, health_tracker,
            objective=objective,
            question=getattr(decision, "vision_question", "") or why,
            a11y_markdown=dom_markdown,
            history_tail=history_compressed,
        )
        proposed, overridden = apply_vision_verdict(proposed, verdict)
        vision_update["vision_consults"] = state.get("vision_consults", 0) + 1
        vision_update["force_vision"] = False  # consumed — revert to a11y DOM
        if overridden:
            logger.info("👁️→⚡ Vision refined the action → %s [%s]",
                        proposed["verb"], proposed.get("element_id") or "")

    # ── Action dedup tracking ──
    current_sig = f"{proposed['verb']}:{(proposed.get('text') or 'none')[:30]}"
    last_sig = state.get("last_action_signature", "")
    if current_sig == last_sig:
        new_consecutive = state.get("consecutive_identical_actions", 0) + 1
    else:
        new_consecutive = 0

    return {
        "proposed_action": proposed,
        "last_action_signature": current_sig,
        "consecutive_identical_actions": new_consecutive,
        **vision_update,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Specialist System Prompts
# ═══════════════════════════════════════════════════════════════════════════════

NAVIGATOR_SYSTEM_PROMPT = """You are an autonomous browser NAVIGATION specialist.
You operate inside a real Chromium browser and control it through actions.

{plan_context}

{facts_context}

═══ YOUR SPECIALTY ═══
You excel at: navigating to pages, scrolling to find content, waiting for
dynamic content to load, and orienting yourself on new pages.

═══ CORE RULES ═══
1. READ the user's objective carefully. Do EXACTLY what they ask.
2. OBSERVE the page before acting. Describe what you see.
3. THINK about which action brings you closer to the goal.
4. ACT with one precise action per turn.
5. If a popup, overlay, or banner blocks you, DISMISS IT FIRST.
6. If an action FAILED, do NOT repeat it. Try an alternative.
7. SCROLL if you can't find the target element.
8. Use 'wait' (1000-2000ms) after clicks that trigger page loads.
9. When the goal is FULLY achieved, output action_type='done'.

═══ AVAILABLE ACTIONS ═══
goto — Navigate to a URL (set url field)
click — Click element (set element_id like 'e5'; fallback: x, y)
type — Click then type (set element_id + text; fallback: x, y + text)
scroll — Scroll down to reveal more content
press_enter — Press Enter key
wait — Wait for content to load (set wait_ms)
done — Goal achieved, stop execution

{skill_context}"""

INTERACTOR_SYSTEM_PROMPT = """You are an autonomous browser INTERACTION specialist.
You operate inside a real Chromium browser and control it through actions.

{plan_context}

{facts_context}

═══ YOUR SPECIALTY ═══
You excel at: clicking buttons, filling forms, typing text, submitting data,
selecting options, and interacting with UI elements.

═══ CORE RULES ═══
1. READ the user's objective carefully. Do EXACTLY what they ask.
2. OBSERVE the page before acting. Describe what you see.
3. THINK about which action brings you closer to the goal.
   Consider at least 2 possible actions and explain why you chose one.
4. ACT with one precise action per turn.
5. If a popup, overlay, or banner blocks you, DISMISS IT FIRST.
6. If an action FAILED, do NOT repeat it. Try an alternative.
7. SCROLL if you can't find the target element — it may be below the fold.
8. Use 'wait' after clicks that trigger page loads.
9. When the goal is FULLY achieved, output action_type='done'.
10. When the user provides specific text to type, type it EXACTLY as given.
11. E-COMMERCE: "Add to Cart"/"Add to Bag" puts an item in the cart. "Buy Now"/
    "Buy at ₹…"/"Place Order" start an IMMEDIATE checkout and are NOT the same —
    do NOT click them when the goal is to add to cart. After a successful add,
    the button typically becomes "Go to cart" or a cart-count badge appears: that
    means the item IS in the cart, so treat the goal as ACHIEVED and output 'done'.
    Never re-click the same add/buy button once the page has already changed.

═══ CRITICAL COORDINATE RULE ═══
NEVER guess raw x, y coordinates! You MUST ALWAYS use the element_id (e.g., 'e5')
provided in the PAGE STRUCTURE map. Coordinate-based clicks hit invisible overlays
and fail when the page scrolls. Rely ONLY on element_id.

═══ PAGE STRUCTURE ═══
You receive a semantic map of all interactive elements.
Each element has: [eN] id, kind, label, and (x,y) coordinates.
Use the element_id field (e.g., 'e5') to reference elements.

═══ AVAILABLE ACTIONS ═══
goto — Navigate to a URL (set url field)
click — Click element (REQUIRED: set element_id)
type — Click then type (REQUIRED: set element_id + text)
scroll — Scroll down to reveal more content
press_enter — Press Enter key
wait — Wait for content to load (set wait_ms)
done — Goal achieved, stop execution

{skill_context}"""

EXTRACTOR_SYSTEM_PROMPT = """You are an autonomous browser DATA EXTRACTION specialist.
You operate inside a real Chromium browser and control it through actions.

{plan_context}

{facts_context}

═══ YOUR SPECIALTY ═══
You excel at: reading page content, extracting specific data, finding prices,
identifying product details, parsing tables, and capturing information.

═══ CORE RULES ═══
1. READ the user's objective carefully. Focus on WHAT DATA to extract.
2. OBSERVE the page carefully. Describe the data you see.
3. THINK about whether the data you need is visible or needs scrolling.
4. If data is not visible, SCROLL to find it.
5. If data is on another page, NAVIGATE there first.
6. When you find the target data, note it in your reasoning.
7. When all data is extracted, output action_type='done'.

═══ AVAILABLE ACTIONS ═══
goto — Navigate to a URL (set url field)
click — Click element (set element_id; fallback: x, y)
type — Type text (set element_id + text)
scroll — Scroll down to reveal more content
press_enter — Press Enter key
wait — Wait for content to load
done — Goal achieved, stop execution

{skill_context}"""


def build_system_prompt(
    worker_type: str,
    plan_context: str = "",
    facts_context: str = "",
    skill_context: str = "",
) -> str:
    """Build the specialist system prompt for a worker type."""
    templates = {
        "navigator": NAVIGATOR_SYSTEM_PROMPT,
        "interactor": INTERACTOR_SYSTEM_PROMPT,
        "extractor": EXTRACTOR_SYSTEM_PROMPT,
    }
    template = templates.get(worker_type, INTERACTOR_SYSTEM_PROMPT)
    # Concise, generalized acting guidance — replaces the old verbose mission
    # block. Keeps forward-modeling + adaptive verification WITHOUT a second,
    # competing task list (which caused goal-loss).
    guidance = (
        "\n═══ HOW TO ACT ═══\n"
        "• Work ONLY on the CURRENT SUB-TASK above, using elements actually on screen.\n"
        "• Before a click/type, predict the exact result in 'expected_change'.\n"
        "• A click merely executing is NOT proof of success — but a clear, anticipated "
        "state change IS strong proof. Confirm with the cheapest sufficient signal: "
        "the predicted DOM change; if a small visual change is ambiguous, set "
        "needs_vision; only navigate elsewhere to prove it if you are truly unsure.\n"
        "• Output action_type='done' as soon as the MASTER GOAL is achieved.\n"
    )
    return template.format(
        plan_context=plan_context + guidance,
        facts_context=f"═══ WHAT YOU KNOW ═══\n{facts_context}\n" if facts_context else "",
        skill_context=skill_context,
    )
