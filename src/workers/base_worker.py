"""Base Worker — Shared LLM invocation logic for all specialist workers.

All workers share the same pattern:
  1. Build a specialist system prompt
  2. Build a user prompt with current DOM + context
  3. Invoke the LLM via the failover chain
  4. Parse the structured output into a ProposedAction
  5. Return state updates (proposed_action, history entry, etc.)

Workers NEVER execute actions directly. They propose actions that
Overwatch validates before committing.

V30: Added "Look-Before-You-Leap" coordinate validation for vision-returned
pixel coordinates (shadow DOM / custom web component targets not in a11y tree).
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
#  V30: Look-Before-You-Leap Coordinate Validation
# ═══════════════════════════════════════════════════════════════════════════════

async def _validate_coord_click(
    invoke_fn, vision_chain, breaker, health_tracker,
    *, x: float, y: float, intended_target: str, objective: str,
) -> tuple[bool, str]:
    """Ask Vision to verify what is at (x, y) before clicking.

    This is the "Look-Before-You-Leap" safety net for V30 coordinate fallback.
    When Vision returns pixel coordinates instead of an element_id (because the
    target is absent from the a11y tree), we take a SECOND screenshot and ask
    Vision to confirm that the coordinate actually matches the intended target.

    Returns (valid, reason). Does NOT modify Overwatch — if valid, the action
    flows to Overwatch with coordinates that pass its grounding check naturally.
    """
    if invoke_fn is None or not vision_chain:
        return False, "no vision chain available"

    from mcp_tools import mcp_screenshot
    shot = await mcp_screenshot(full_page=False)
    if not shot.get("ok"):
        return False, "screenshot failed for coord validation"

    from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM

    class _CoordValidation(BaseModel):
        element_at_point: str = Field(
            description="Describe EXACTLY what visual element/button/link is at the marked coordinate."
        )
        matches_target: bool = Field(
            description="Does the element at the coordinate match the intended target?"
        )
        confidence: float = Field(description="0.0-1.0 confidence in the match.")
        reasoning: str = Field(description="Why this is or is not the right target.")

    vp_w = shot.get("width", 1440)
    vp_h = shot.get("height", 1080)
    messages = [
        _SM(content=(
            "You are a precision visual validator for a browser automation agent. "
            "You will be given a screenshot, a coordinate point [x,y], and an "
            "intended target description. Your ONLY job: look at EXACTLY what is "
            "at that coordinate on the screenshot and determine if it matches the "
            "intended target. Be precise — a 'Star' button is NOT the same as a "
            "'5 stars' text link. A 'Fork' button is NOT the same as a fork count."
        )),
        _HM(content=(
            f"═══ COORDINATE TO VALIDATE ═══\n"
            f"Point: ({x:.0f}, {y:.0f}) on a {vp_w}×{vp_h} viewport\n\n"
            f"═══ INTENDED TARGET ═══\n{intended_target}\n\n"
            f"═══ OBJECTIVE ═══\n{objective}\n\n"
            f"Look at the screenshot. What EXACTLY is at coordinate "
            f"({x:.0f}, {y:.0f})? Is it '{intended_target}'?"
        )),
    ]

    try:
        result, _model = await invoke_fn(
            vision_chain, messages, _CoordValidation,
            breaker, base64_image=shot["base64"], health_tracker=health_tracker,
        )
        if result is None:
            return False, "vision validation returned None"
        if result.matches_target and result.confidence >= 0.65:
            logger.info(
                "✅ COORD VALIDATED: (%.0f,%.0f) confirmed as '%s' [%.0f%%]",
                x, y, result.element_at_point[:50], result.confidence * 100,
            )
            return True, f"confirmed: {result.element_at_point[:60]}"
        else:
            logger.warning(
                "❌ COORD REJECTED: (%.0f,%.0f) is '%s', not '%s' [%.0f%%]",
                x, y, result.element_at_point[:50], intended_target[:40],
                result.confidence * 100,
            )
            return False, f"mismatch: found '{result.element_at_point[:60]}'"
    except Exception as e:
        logger.warning("coord validation error: %s", str(e)[:120])
        return False, f"validation error: {str(e)[:80]}"


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
        description=("One of: 'goto', 'click', 'type', 'scroll', 'press_enter', "
                     "'wait', 'done', 'ask_user', 'hover', 'select_option', 'press_key'")
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
    confidence: float = Field(
        default=0.7,
        description=(
            "Your confidence (0.0-1.0) that THIS action is the correct next step. "
            "Be honest and calibrated: high (>0.8) only when the target and intent "
            "are unambiguous; low (<0.5) when several elements look plausible or "
            "you are unsure the action will work. Used to weight multi-model "
            "consensus on critical irreversible actions."
        ),
    )
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
    # V31: Worker Veto — proof of completion for done actions
    proof_of_completion: str = Field(
        default="",
        description=(
            "MANDATORY when action_type='done'. Provide concrete, observed state-change "
            "evidence that proves the goal is achieved. Cite EXACTLY what you witnessed: "
            "e.g. 'Cart icon count changed from 0 to 1 after clicking Add to Cart', "
            "'Button text changed from Star to Unstar', 'Confirmation toast appeared "
            "saying Thank you for your order', 'Page redirected to order confirmation "
            "URL'. This is YOUR testimony as the execution witness — the verification "
            "system will evaluate it. Be specific and factual. Empty for non-done actions."
        ),
    )
    # V17: Interactive Data Guardrails — stop and ask when personal data is needed
    missing_data: str = Field(
        default="",
        description=(
            "If action_type='ask_user': describe WHAT information is missing that "
            "prevents you from continuing. E.g., 'Shipping address required for "
            "checkout' or 'Phone number needed for account verification'."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Worker Invocation (shared by all workers)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_target_lock_section(state: dict[str, Any], objective: str) -> tuple[Any, str]:
    try:
        from feature_flags import target_lock_enabled
        if not target_lock_enabled():
            return None, ""
        from target_lock import extract_target, render_target_lock_block
        active_sub = next(
            (s.get("desc", "") for s in (state.get("plan_steps") or [])
             if s.get("status") in ("active", "in_progress")),
            "",
        )
        bound_target = extract_target(objective, active_sub)
        return bound_target, render_target_lock_block(bound_target, objective)
    except Exception as e:  # noqa: BLE001
        logger.debug("Target Lock skipped (non-fatal): %s", e)
        return None, ""


def _build_hesitation_section(pending_intent: Any) -> str:
    try:
        from feature_flags import intent_journal_enabled
        if not (intent_journal_enabled() and pending_intent):
            return ""
        from intent_journal import render_hesitation
        block = render_hesitation(pending_intent)
        if block:
            logger.info(
                "⚠️ HESITATION: predecessor %s on '%s' unconfirmed — "
                "warning worker not to blindly repeat",
                pending_intent.get("verb"),
                pending_intent.get("target_name") or pending_intent.get("element_id") or "",
            )
        return block
    except Exception as e:  # noqa: BLE001
        logger.debug("Hesitation block skipped (non-fatal): %s", e)
        return ""


def _build_lock_list_section(state: dict[str, Any]) -> str:
    try:
        from feature_flags import subgoal_lock_enabled
        if subgoal_lock_enabled() and state.get("prm_checklist"):
            from subgoal_lock import render_lock_list
            return render_lock_list(state.get("prm_checklist"))
    except Exception as e:  # noqa: BLE001
        logger.debug("Lock-list block skipped (non-fatal): %s", e)
    return ""


def _assemble_worker_prompt(state: dict[str, Any]) -> tuple[str, Any, Any]:
    """Assemble the structured user prompt + return `(prompt, bound_target, pending_intent)`."""
    from cognition import build_guidance, render_strategy_block

    objective = state.get("objective", "")
    current_url = state.get("current_url", "")
    step_number = state.get("step_number", 0)
    max_steps = state.get("max_steps", 25)
    dom_markdown = state.get("dom_markdown", "")
    history_compressed = state.get("history_compressed", "")
    pending_intent = state.get("last_attempted_action")

    strategy_block = render_strategy_block(
        strategy=state.get("strategy", ""),
        confidence=state.get("strategy_confidence", 1.0),
        beliefs=state.get("beliefs", []) or [],
        success_criteria=state.get("success_criteria", ""),
    )
    bound_target, target_lock_block = _build_target_lock_section(state, objective)
    hesitation_block = _build_hesitation_section(pending_intent)
    lock_list_block = _build_lock_list_section(state)

    login_hint = (
        "\n\n🔑 LOGIN STATUS: You appear to be LOGGED IN. "
        "Profile/account indicators detected. DO NOT attempt to log in again."
        if state.get("login_detected", False)
        else ""
    )
    guidance_block = build_guidance(state)
    if guidance_block:
        logger.info("🧭 GUIDANCE: %s", guidance_block.split("\n", 1)[-1][:120])

    user_prompt = (
        f"═══ YOUR MISSION ═══\n"
        f"{objective}\n\n"
        f"NOTE: The above describes the user's INTENT. Any numbered steps are guidance, not strict commands. "
        f"YOU decide the actual actions based on what you see on screen.\n\n"
        + (f"{strategy_block}\n"
           "Follow your strategy, but ADAPT it to what you actually see. When your "
           "DONE WHEN criteria are met, output action_type='done' — do not keep acting "
           "after the goal is achieved.\n\n" if strategy_block else "")
        + (f"{target_lock_block}\n\n" if target_lock_block else "")
        + f"═══ CURRENT CONTEXT ═══\n"
        f"URL: {current_url}\n"
        f"Step: {step_number + 1}/{max_steps} ({max_steps - step_number - 1} remaining)"
        f"{login_hint}\n\n"
        + (f"{guidance_block}\n\n" if guidance_block else "")
        + (f"{hesitation_block}\n\n" if hesitation_block else "")
        + (f"{lock_list_block}\n\n" if lock_list_block else "")
        + f"═══ ACTION HISTORY (compressed) ═══\n"
        f"{history_compressed or '(first step)'}\n\n"
        f"═══ PAGE STRUCTURE ═══\n"
        f"{dom_markdown}\n\n"
        f"═══ REMEMBER YOUR MISSION ═══\n"
        f"{objective[:300]}\n"
        f"Now OBSERVE the page structure above, THINK about your situation, and choose your NEXT action."
    )
    return user_prompt, bound_target, pending_intent


def _classify_and_build_proposed(
    decision: WorkerAction,
    selector_map: dict[str, Any],
    current_url: str,
) -> tuple[dict[str, Any], Any]:
    """Log the worker's OTA reasoning chain, classify risk, and build ProposedAction dict."""
    from action_classifier import ActionRisk, classify_action

    logger.info("👁️ OBSERVE: %s", decision.screen_state[:200])
    logger.info("🔄 PREVIOUS: %s", decision.previous_action_result[:150])
    logger.info("📊 PROGRESS: %s", decision.goal_progress[:150])
    logger.info("🧠 REASONING: %s", decision.reasoning[:200])
    if getattr(decision, "expected_change", ""):
        logger.info("🔮 EXPECT: %s", decision.expected_change[:180])
    logger.info("⚡ ACTION: %s", decision.action_type)

    el_data = selector_map.get(decision.element_id, {}) if decision.element_id else {}
    target_name = (el_data.get("name", el_data.get("text", "")) or "")[:60]
    risk = classify_action(
        action_type=decision.action_type,
        target_name=target_name,
        target_text=decision.text[:60] if decision.text else "",
        url=current_url,
        element_kind=el_data.get("kind", "") if decision.element_id else "",
    )
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
    return proposed, risk


def _compute_clarity_signal(
    decision: WorkerAction,
    proposed: dict[str, Any],
    state: dict[str, Any],
    bound_target: Any,
    selector_map: dict[str, Any],
) -> Any:
    try:
        from clarity import compute_clarity
        clarity_input = {
            **proposed,
            "confidence": float(getattr(decision, "confidence", 0.7) or 0.7),
            "needs_vision": bool(getattr(decision, "needs_vision", False)),
        }
        clarity_sig = compute_clarity(
            clarity_input, state, target=bound_target, selector_map=selector_map
        )
        if clarity_sig.uncertain:
            logger.info("🔎 CLARITY: uncertain (%s)", "; ".join(clarity_sig.reasons[:3]))
        return clarity_sig
    except Exception as e:  # noqa: BLE001
        logger.debug("Clarity signal skipped (non-fatal): %s", e)
        return None


def _apply_cascade_outcome(
    cascade: Any,
    decision: WorkerAction,
    proposed: dict[str, Any],
    is_irrev: bool,
    vision_chain: list | None,
    selector_map: dict[str, Any],
    current_url: str,
    state: dict[str, Any],
    consensus_update: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    from action_classifier import ActionRisk, classify_action

    if cascade.abstain:
        consensus_update["abstentions"] = state.get("abstentions", 0) + 1
        if vision_chain:
            logger.warning(
                "🛑 ABSTAIN → consulting vision to disambiguate the ambiguous action before acting."
            )
            return proposed, True
        if is_irrev:
            logger.warning(
                "🛑 ABSTAIN (irreversible) → no vision; holding (wait) and re-examining before committing."
            )
            proposed = {
                **proposed,
                "verb": "wait",
                "wait_ms": 800,
                "risk_level": "REVERSIBLE",
                "reversible": True,
            }
            consensus_update["correction_context"] = (
                "\n\n⚖️ The critical irreversible action is AMBIGUOUS — the "
                "models disagreed on the target. Re-examine the page and "
                "identify the SINGLE correct element before committing."
            )
        else:
            logger.info(
                "⚖️ ABSTAIN (reversible) → proceeding on primary (a retry can recover if wrong)."
            )
        return proposed, False

    cd = cascade.decision
    if cd is not decision and (
        getattr(cd, "action_type", None) != decision.action_type
        or getattr(cd, "element_id", None) != decision.element_id
    ):
        win_el = selector_map.get(cd.element_id, {}) if cd.element_id else {}
        win_target = (win_el.get("name", win_el.get("text", "")) or "")[:60]
        win_risk = classify_action(
            action_type=cd.action_type,
            target_name=win_target,
            target_text=(cd.text or "")[:60],
            url=current_url,
            element_kind=win_el.get("kind", ""),
        )
        logger.info("🗳️ CASCADE override → %s [%s]", cd.action_type, cd.element_id or "")
        proposed = {
            **proposed,
            "verb": cd.action_type,
            "element_id": cd.element_id,
            "text": cd.text,
            "target_name": win_target,
            "risk_level": win_risk.name,
            "reversible": win_risk == ActionRisk.REVERSIBLE,
        }
    return proposed, False


async def _run_consensus_stage(
    *,
    decision: WorkerAction,
    used_model: str,
    proposed: dict[str, Any],
    risk: Any,
    clarity_sig: Any,
    pending_intent: Any,
    messages: list,
    state: dict[str, Any],
    failover_chain: list,
    breaker: Any,
    health_tracker: Any,
    invoke_fn: Any,
    vision_chain: list | None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    consensus_update: dict[str, Any] = {}
    try:
        from action_classifier import ActionRisk
        from clarity import needs_consensus
        from consensus import cascade_consensus, consensus_enabled, count_distinct_base_models
        from feature_flags import clarity_consensus_enabled

        is_irrev = risk == ActionRisk.IRREVERSIBLE
        if clarity_sig is not None:
            do_vote, vote_reason = needs_consensus(
                clarity_sig, is_irreversible=is_irrev, broaden=clarity_consensus_enabled()
            )
        else:
            do_vote, vote_reason = is_irrev, "irreversible action"

        repeating_uncertain = False
        try:
            from intent_journal import same_action
            if pending_intent and same_action(pending_intent, proposed):
                repeating_uncertain = True
                do_vote = True
                vote_reason = ((vote_reason + "; ") if vote_reason else "") + (
                    "repeating an UNCONFIRMED prior action (possible double-apply)"
                )
        except Exception:  # noqa: BLE001
            pass

        if not (do_vote and consensus_enabled() and count_distinct_base_models(failover_chain) >= 2):
            return proposed, consensus_update, False

        logger.info("🗳️ PRE-ACTION CONSENSUS — %s", vote_reason)
        force_escalate = (
            state.get("consecutive_identical_actions", 0) >= 2
            or bool(state.get("correction_context"))
            or repeating_uncertain
            or (clarity_sig is not None and clarity_sig.uncertain)
        )
        selector_map = state.get("selector_map", {})
        cascade = await cascade_consensus(
            primary_decision=decision,
            primary_model=used_model,
            messages=messages,
            schema=WorkerAction,
            invoke_fn=invoke_fn,
            chain=failover_chain,
            breaker=breaker,
            health_tracker=health_tracker,
            selector_map=selector_map,
            force_escalate=force_escalate,
        )
        consensus_update["consensus_votes"] = state.get("consensus_votes", 0) + 1
        logger.info(
            "🗳️ CASCADE [%s] +%d calls, agreement=%.0f%% — %s",
            cascade.path,
            cascade.extra_calls,
            cascade.agreement * 100,
            cascade.detail,
        )
        proposed, force_consult = _apply_cascade_outcome(
            cascade, decision, proposed, is_irrev, vision_chain,
            selector_map, state.get("current_url", ""), state, consensus_update,
        )
        return proposed, consensus_update, force_consult
    except Exception as e:  # noqa: BLE001 — consensus never breaks the step
        logger.debug("Cascade consensus skipped (non-fatal): %s", e)
        return proposed, consensus_update, False


async def _validate_vision_coords_if_needed(
    proposed: dict[str, Any],
    objective: str,
    vision_chain: list,
    breaker: Any,
    health_tracker: Any,
    invoke_fn: Any,
) -> None:
    if not (
        proposed.get("vision_coords")
        and proposed.get("x") is not None
        and proposed.get("y") is not None
    ):
        return
    intended = (
        proposed.get("reasoning", "").replace("[vision] ", "")[:120]
        or proposed.get("target_name", "unknown target")
    )
    try:
        valid, reason = await _validate_coord_click(
            invoke_fn, vision_chain, breaker, health_tracker,
            x=proposed["x"], y=proposed["y"],
            intended_target=intended, objective=objective,
        )
        if valid:
            logger.info("👁️✅ Coordinate click validated: %s", reason[:80])
        else:
            logger.warning("👁️❌ Coordinate click REJECTED: %s — dropping coords", reason[:80])
            proposed["x"] = None
            proposed["y"] = None
            proposed["vision_coords"] = False
    except Exception as val_err:  # noqa: BLE001
        logger.warning("coord validation failed (non-fatal): %s", val_err)


async def _run_vision_stage(
    *,
    decision: WorkerAction,
    proposed: dict[str, Any],
    clarity_sig: Any,
    force_consult: bool,
    state: dict[str, Any],
    vision_chain: list | None,
    breaker: Any,
    health_tracker: Any,
    invoke_fn: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from vision_consult import apply_vision_verdict, consult_vision, should_consult_vision

    vision_update: dict[str, Any] = {}
    consult, why = should_consult_vision(
        needs_vision=getattr(decision, "needs_vision", False),
        state=state,
        action_type=decision.action_type,
    )
    if not consult and clarity_sig is not None:
        try:
            from clarity import needs_vision_for_clarity
            from vision_consult import MAX_VISION_CONSULTS
            cl_v, cl_why = needs_vision_for_clarity(clarity_sig)
            if cl_v and int(state.get("vision_consults", 0) or 0) < MAX_VISION_CONSULTS:
                consult, why = True, cl_why
        except Exception:  # noqa: BLE001
            pass
    if force_consult:
        consult, why = True, "consensus abstention — disambiguate critical action"
    if not (consult and vision_chain):
        return proposed, vision_update

    objective = state.get("objective", "")
    logger.info("🧠→👁️ Escalating to vision: %s", why)
    verdict, _vm = await consult_vision(
        invoke_fn, vision_chain, breaker, health_tracker,
        objective=objective,
        question=getattr(decision, "vision_question", "") or why,
        a11y_markdown=state.get("dom_markdown", ""),
        history_tail=state.get("history_compressed", ""),
    )
    proposed, overridden = apply_vision_verdict(proposed, verdict)
    vision_update["vision_consults"] = state.get("vision_consults", 0) + 1
    vision_update["force_vision"] = False
    if overridden:
        logger.info("👁️→⚡ Vision refined the action → %s [%s]", proposed["verb"], proposed.get("element_id") or "")
        await _validate_vision_coords_if_needed(
            proposed, objective, vision_chain, breaker, health_tracker, invoke_fn
        )
    return proposed, vision_update


async def _run_webdreamer_stage(
    *,
    proposed: dict[str, Any],
    clarity_sig: Any,
    state: dict[str, Any],
    dreamer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    webdreamer_update: dict[str, Any] = {}
    try:
        from action_classifier import ActionRisk, classify_action
        from feature_flags import webdreamer_enabled
        if not (webdreamer_enabled() and dreamer is not None and clarity_sig is not None and clarity_sig.uncertain):
            return proposed, webdreamer_update
        from web_dreamer import CandidateAction, should_invoke_dreamer, should_override_with_dreamer

        selector_map = state.get("selector_map", {})
        if not should_invoke_dreamer(
            element_count=len(selector_map or {}),
            action_risk_level=proposed.get("risk_level", "REVERSIBLE"),
            consecutive_no_progress=int(state.get("ineffective_streak", 0) or 0),
            step_number=int(state.get("step_number", 0) or 0),
            same_url_streak=int(state.get("same_url_streak", 0) or 0),
        ):
            return proposed, webdreamer_update

        proposed_ca = CandidateAction(
            action_type=proposed["verb"], element_id=proposed.get("element_id"),
            text=proposed.get("text"), url=proposed.get("url"),
            x=proposed.get("x"), y=proposed.get("y"),
            reasoning=proposed.get("reasoning", ""),
        )
        logger.info(
            "🌙 WebDreamer: simulating candidates before a high-stakes ambiguous action (%s)…",
            proposed.get("verb"),
        )
        webdreamer_update["webdreamer_runs"] = state.get("webdreamer_runs", 0) + 1
        dr = await asyncio.wait_for(
            dreamer.plan_and_select(
                dom_markdown=state.get("dom_markdown", ""),
                objective=state.get("objective", ""),
                plan_context=state.get("plan_render", ""),
                action_history=state.get("history_compressed", ""),
                current_url=state.get("current_url", ""),
                proposed_action=proposed_ca,
                situation=state,
            ),
            timeout=45.0,
        )
        best = dr.best_action
        logger.info("🌙 WebDreamer: best=%s score=%.2f (k=%d)", best.describe(), dr.best_score, dr.candidates_generated)
        if should_override_with_dreamer(
            dr.best_score, best.action_type, best.element_id, proposed["verb"], proposed.get("element_id")
        ):
            win_el = selector_map.get(best.element_id, {}) if best.element_id else {}
            win_target = (win_el.get("name", win_el.get("text", "")) or "")[:60]
            win_risk = classify_action(
                action_type=best.action_type, target_name=win_target,
                target_text=(best.text or "")[:60], url=state.get("current_url", ""),
                element_kind=win_el.get("kind", ""),
            )
            proposed = {
                **proposed,
                "verb": best.action_type,
                "element_id": best.element_id,
                "text": best.text,
                "url": best.url,
                "target_name": win_target,
                "risk_level": win_risk.name,
                "reversible": win_risk == ActionRisk.REVERSIBLE,
                "reasoning": f"[dreamer] {(best.reasoning or '')[:160]}",
            }
            if not best.element_id and best.x is None and best.y is None and proposed.get("vision_coords"):
                logger.info(
                    "🌙→👁️ WebDreamer inherited validated vision coords (%.0f,%.0f)",
                    proposed.get("x", 0), proposed.get("y", 0),
                )
            webdreamer_update["webdreamer_overrides"] = state.get("webdreamer_overrides", 0) + 1
            logger.info("🌙 WebDreamer OVERRIDE → %s", best.describe())
        else:
            logger.info("🌙 WebDreamer confirms the original (best_score=%.2f)", dr.best_score)
    except asyncio.TimeoutError:
        logger.warning("🌙 WebDreamer timed out — keeping the original action")
    except Exception as e:  # noqa: BLE001 — simulation never breaks the step
        logger.debug("WebDreamer skipped (non-fatal): %s", e)
    return proposed, webdreamer_update


def _apply_subgoal_lock_backstop(
    proposed: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    subgoal_lock_update: dict[str, Any] = {}
    try:
        from feature_flags import subgoal_lock_enabled
        if not (subgoal_lock_enabled() and state.get("done_blocked", 0) > 0 and state.get("prm_checklist")):
            return proposed, subgoal_lock_update
        from subgoal_lock import remaining_subgoals, targets_locked_subgoal

        hit = targets_locked_subgoal(proposed, state.get("prm_checklist"))
        if hit is not None:
            rem = remaining_subgoals(state.get("prm_checklist"))
            rem_str = "; ".join((d.get("desc") or "")[:60] for d in rem[:2]) or "the remaining work"
            logger.warning(
                "🔒 Sub-Goal Lock: blocked a RE-DO of completed '%s' — redirecting to remaining work.",
                (hit.get("desc") or "")[:50],
            )
            proposed = {
                **proposed,
                "verb": "wait",
                "wait_ms": 600,
                "risk_level": "REVERSIBLE",
                "reversible": True,
            }
            subgoal_lock_update["correction_context"] = (
                f"\n\n🔒 STOP — '{(hit.get('desc') or 'that sub-goal')[:60]}' is ALREADY "
                f"DONE and LOCKED. Do NOT repeat it (its control may still be visible). "
                f"Work ONLY on what remains: {rem_str}."
            )
    except Exception as e:  # noqa: BLE001 — backstop never breaks the step
        logger.debug("Sub-Goal Lock backstop skipped (non-fatal): %s", e)
    return proposed, subgoal_lock_update


async def invoke_worker(
    state: dict[str, Any],
    system_prompt: str,
    failover_chain: list,
    breaker,
    health_tracker,
    invoke_fn,
    vision_chain: list | None = None,
    dreamer=None,
) -> dict[str, Any]:
    """Core worker invocation logic shared by all specialists."""
    user_prompt, bound_target, pending_intent = _assemble_worker_prompt(state)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

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
        return {}

    selector_map = state.get("selector_map", {})
    proposed, risk = _classify_and_build_proposed(
        decision, selector_map, state.get("current_url", "")
    )
    clarity_sig = _compute_clarity_signal(decision, proposed, state, bound_target, selector_map)

    proposed, consensus_update, force_consult = await _run_consensus_stage(
        decision=decision, used_model=used_model, proposed=proposed, risk=risk,
        clarity_sig=clarity_sig, pending_intent=pending_intent, messages=messages,
        state=state, failover_chain=failover_chain, breaker=breaker,
        health_tracker=health_tracker, invoke_fn=invoke_fn, vision_chain=vision_chain,
    )
    proposed, vision_update = await _run_vision_stage(
        decision=decision, proposed=proposed, clarity_sig=clarity_sig,
        force_consult=force_consult, state=state, vision_chain=vision_chain,
        breaker=breaker, health_tracker=health_tracker, invoke_fn=invoke_fn,
    )
    proposed, webdreamer_update = await _run_webdreamer_stage(
        proposed=proposed, clarity_sig=clarity_sig, state=state, dreamer=dreamer,
    )
    proposed, subgoal_lock_update = _apply_subgoal_lock_backstop(proposed, state)

    from intent_journal import text_fingerprint
    current_sig = f"{proposed['verb']}:{text_fingerprint(proposed.get('text')) or 'none'}"
    new_consecutive = (
        state.get("consecutive_identical_actions", 0) + 1
        if current_sig == state.get("last_action_signature", "")
        else 0
    )

    return {
        "proposed_action": proposed,
        "last_action_signature": current_sig,
        "consecutive_identical_actions": new_consecutive,
        "bound_target": (bound_target.phrase if bound_target else ""),
        **consensus_update,
        **vision_update,
        **webdreamer_update,
        **subgoal_lock_update,
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
9. When the goal is FULLY achieved, output action_type='done' with proof_of_completion.

═══ PROOF OF COMPLETION (CRITICAL) ═══
When you output action_type='done', you MUST fill proof_of_completion with the
EXACT state-changes you observed that prove success. Examples:
  • 'After clicking Add to Cart, the cart badge count changed from 0 to 1'
  • 'The Star button text changed to Unstar after clicking'
  • 'Form submitted, page redirected to /thank-you confirmation page'
This proof is YOUR testimony — be specific and factual about what changed.

═══ HIERARCHY OF TRUTH (V32 — CRITICAL) ═══
If an action log reports an error (e.g., 'Click Failed', 'Click Ineffective'), but your
observation of the CURRENT PAGE STATE shows the intended result DID happen (e.g., the
radio button is now selected, the checkbox is checked, the text is typed, the dropdown
value changed), YOU MUST TRUST THE VISUAL REALITY. The action succeeded — the error
log is a false negative from the mechanical click engine.
DO NOT retry an action that visually succeeded. Ignore the false error, consider the
step complete, and proceed to the next step in your plan.
Priority order: Visual Page State > Overwatch Confirmation > Action Engine Logs.

═══ AVAILABLE ACTIONS ═══
goto — Navigate to a URL (set url field)
click — Click element (set element_id like 'e5'; fallback: x, y)
type — Click then type (set element_id + text; fallback: x, y + text)
scroll — Scroll down to reveal more content
press_enter — Press Enter key
wait — Wait for content to load (set wait_ms)
select_option — Select dropdown option (set element_id + text for the option value/label)
hover — Hover over element to reveal menus/tooltips (set element_id; fallback: x, y)
press_combo — Press key or shortcut (set key_combo like 'Escape', 'Control+A', 'Tab', 'ArrowDown')
drag_and_drop — Drag from source to target (set x,y for source + target_x,target_y for dest)
upload_file — Upload file to file input (set element_id + file_path)
scroll_to — Scroll in direction (set direction: 'up'/'down'/'left'/'right' + scroll_amount in px)
done — Goal achieved, stop execution (MUST include proof_of_completion)

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
9. When the goal is FULLY achieved, output action_type='done' with proof_of_completion.
10. When the user provides specific text to type, type it EXACTLY as given.
11. E-COMMERCE: "Add to Cart"/"Add to Bag" puts an item in the cart. "Buy Now"/
    "Buy at ₹…"/"Place Order" start an IMMEDIATE checkout and are NOT the same —
    do NOT click them when the goal is to add to cart. After a successful add,
    the button typically becomes "Go to cart" or a cart-count badge appears: that
    means the item IS in the cart, so treat the goal as ACHIEVED and output 'done'.
    Never re-click the same add/buy button once the page has already changed.
12. MISSING PERSONAL DATA: If a form, checkout, or registration requires information
    you DO NOT have (shipping address, payment info, phone number, custom specs),
    output action_type='ask_user' and describe the missing data in 'missing_data'.
    NEVER guess, hallucinate, or fabricate personal information.

═══ PROOF OF COMPLETION (CRITICAL) ═══
When you output action_type='done', you MUST fill proof_of_completion with the
EXACT state-changes you observed that prove success. Examples:
  • 'After clicking Add to Cart, the cart badge count changed from 0 to 1'
  • 'The Star button text changed to Unstar after clicking'
  • 'Confirmation toast appeared: Your order has been placed'
This proof is YOUR testimony — be specific and factual about what changed.

═══ HIERARCHY OF TRUTH (V32 — CRITICAL) ═══
If an action log reports an error (e.g., 'Click Failed', 'Click Ineffective'), but your
observation of the CURRENT PAGE STATE shows the intended result DID happen (e.g., the
radio button is now selected, the checkbox is checked, the text is typed, the dropdown
value changed), YOU MUST TRUST THE VISUAL REALITY. The action succeeded — the error
log is a false negative from the mechanical click engine.
DO NOT retry an action that visually succeeded. Ignore the false error, consider the
step complete, and proceed to the next step in your plan.
Priority order: Visual Page State > Overwatch Confirmation > Action Engine Logs.

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
select_option — Select dropdown option (set element_id + text for the option value/label)
hover — Hover over element to reveal menus/tooltips (set element_id)
press_combo — Press key or shortcut (set key_combo like 'Escape', 'Control+A', 'Tab', 'ArrowDown')
drag_and_drop — Drag from source to target (set x,y for source + target_x,target_y for dest)
upload_file — Upload file to file input (set element_id + file_path)
scroll_to — Scroll in direction (set direction: 'up'/'down'/'left'/'right' + scroll_amount in px)
done — Goal achieved, stop execution (MUST include proof_of_completion)

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
7. When all data is extracted, output action_type='done' with proof_of_completion.

═══ PROOF OF COMPLETION (CRITICAL) ═══
When you output action_type='done', you MUST fill proof_of_completion with the
EXACT evidence of what you found/extracted. Be specific and factual.

═══ HIERARCHY OF TRUTH (V32 — CRITICAL) ═══
If an action log reports an error (e.g., 'Click Failed', 'Click Ineffective'), but your
observation of the CURRENT PAGE STATE shows the intended result DID happen (e.g., the
radio button is now selected, the checkbox is checked, the text is typed, the dropdown
value changed), YOU MUST TRUST THE VISUAL REALITY. The action succeeded — the error
log is a false negative from the mechanical click engine.
DO NOT retry an action that visually succeeded. Ignore the false error, consider the
step complete, and proceed to the next step in your plan.
Priority order: Visual Page State > Overwatch Confirmation > Action Engine Logs.

═══ AVAILABLE ACTIONS ═══
goto — Navigate to a URL (set url field)
click — Click element (set element_id; fallback: x, y)
type — Type text (set element_id + text)
scroll — Scroll down to reveal more content
press_enter — Press Enter key
wait — Wait for content to load
select_option — Select dropdown option (set element_id + text for the option value/label)
hover — Hover over element to reveal menus/tooltips (set element_id; fallback: x, y)
press_combo — Press key or shortcut (set key_combo like 'Escape', 'Control+A', 'Tab')
drag_and_drop — Drag from source to target (set x,y + target_x,target_y)
upload_file — Upload file to file input (set element_id + file_path)
scroll_to — Scroll in direction (set direction + scroll_amount in px)
done — Goal achieved, stop execution (MUST include proof_of_completion)

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
        "• Output action_type='done' as soon as the MASTER GOAL is achieved. "
        "ALWAYS fill proof_of_completion with the exact state-changes you observed.\n"
    )
    try:
        from feature_flags import hybrid_primitives_enabled
        if hybrid_primitives_enabled():
            guidance += (
                "\n═══ EXTRA ACTIONS (use only when a plain click won't do) ═══\n"
                "• 'hover' (set element_id): reveal a hover menu / tooltip / submenu.\n"
                "• 'select_option' (set element_id + put the option's visible text in "
                "'text'): choose a value from a NATIVE dropdown <select>. For a custom/"
                "styled dropdown, click it open and click the option instead.\n"
                "• 'press_key' (put the key in 'text', e.g. 'Escape', 'Tab', "
                "'ArrowDown', 'Enter'): press a single key or chord.\n"
            )
    except Exception:
        pass
    return template.format(
        plan_context=plan_context + guidance,
        facts_context=f"═══ WHAT YOU KNOW ═══\n{facts_context}\n" if facts_context else "",
        skill_context=skill_context,
    )
