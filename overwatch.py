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
        # V21: the 'vision' rung now triggers a REAL screenshot consult on the
        # next worker step (was a no-op text directive before).
        if lad["tactic"] == "vision":
            updates["force_vision"] = True
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

    # ═══════════════════════════════════════════════════════════════════
    #  V29 Atomic Intent Journal — WRITE-AHEAD record BEFORE the side-effect.
    #  If the action times out / crashes mid-execution, this record survives so the
    #  NEXT decision knows the action was attempted (and may have partially applied)
    #  and must NOT blindly repeat it. The in-state write rides the node's atomic
    #  super-step commit; the durable file is written immediately (pre-execution).
    # ═══════════════════════════════════════════════════════════════════
    intent_entry = None
    try:
        from feature_flags import intent_journal_enabled
        if intent_journal_enabled():
            from intent_journal import make_intent, should_journal, persist_intent
            if should_journal(verb):
                el_kind = ""
                if element_id and isinstance(selector_map, dict):
                    el_kind = (selector_map.get(element_id, {}) or {}).get("kind", "") or ""
                intent_entry = make_intent(proposed, step_number, element_kind=el_kind)
                updates["last_attempted_action"] = intent_entry  # survives node return
                persist_intent(intent_entry)                      # durable, atomic, pre-exec
                logger.info("📝 Intent journaled (pre-exec): %s [%s] hazard=%s",
                            intent_entry["verb"], intent_entry.get("element_id") or "",
                            intent_entry.get("hazard"))
    except Exception as e:  # noqa: BLE001 — journaling never blocks the action
        logger.debug("Intent journal write-ahead skipped (non-fatal): %s", e)

    # Execute the action via MCP tools
    action_outcome = await _execute_action(proposed, page)
    updates["action_outcome"] = action_outcome
    updates["page_fsm"] = "ACTION_PENDING"
    updates["total_actions"] = state.get("total_actions", 0) + 1

    # Stamp the journal with the post-execution status so every downstream return
    # path (contradiction / ineffective / retry) carries an accurate ledger. The
    # verified-success path below clears it entirely.
    if intent_entry is not None:
        try:
            from intent_journal import classify_status
            intent_entry = {**intent_entry, "status": classify_status(action_outcome)}
            updates["last_attempted_action"] = intent_entry
        except Exception:  # noqa: BLE001
            pass

    # ═══════════════════════════════════════════════════════════════════
    #  V29 Smart-Scroll guard — stop scrolling into a wall (Mandate 3)
    #  A scroll that doesn't move the page, or that has reached the bottom, is not
    #  going to reveal the target. Track the streak and escalate a DIFFERENT tactic
    #  instead of looping the scroll. Flag-gated; off ⇒ identical to V28.
    # ═══════════════════════════════════════════════════════════════════
    if verb == "scroll":
        try:
            from feature_flags import smart_scroll_enabled
            if smart_scroll_enabled():
                ol = action_outcome.lower()
                unproductive = ("no-op" in ol or "reached page bottom" in ol
                                or "did not move" in ol)
                if unproductive:
                    streak = state.get("scroll_stuck_streak", 0) + 1
                    updates["scroll_stuck_streak"] = streak
                    if streak >= 2:
                        logger.warning("🧭 Smart-scroll: %d unproductive scrolls — the "
                                       "target isn't reachable by scrolling here; "
                                       "escalating a different tactic.", streak)
                        _escalate(state, "Scrolling is not revealing the target "
                                  "(page at bottom / won't move).", updates)
                        if updates.get("overwatch_verdict") != "rollback":
                            updates["overwatch_verdict"] = "retry"
                        updates["page_fsm"] = "READY"
                        return updates
                else:
                    updates["scroll_stuck_streak"] = 0
        except Exception as e:  # noqa: BLE001
            logger.debug("Smart-scroll guard skipped (non-fatal): %s", e)

    # Post-action DOM check
    post_markdown = ""
    post_url = state.get("current_url", "")
    try:
        import dom_parser
        from cognitive_core import dom_data_to_a11y_format
        post_dom = await dom_parser.extract(page, timeout=3.0)
        post_a11y = dom_data_to_a11y_format(post_dom)
        # V29 Reality Monitor inputs (free — we already extracted the post DOM).
        post_markdown = post_dom.get("markdown", "") or ""
        try:
            post_url = page.url or post_url
        except Exception:
            pass
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
    # V29 Phase A: surface the unified state-change score (CriticV12 diffing).
    updates["state_change_score"] = float(getattr(verdict, "state_change_score", 0.0) or 0.0)

    # ═══════════════════════════════════════════════════════════════════
    #  V29 Reality Monitor — did the page do what the worker PREDICTED?
    #  CriticV12 above only answers "did *anything* change?". This compares the
    #  worker's pre-committed `expected_change` against the LIVE screen. A
    #  CONTRADICTED screen (error / rejection / wrong redirect / out-of-stock)
    #  must NOT be committed as progress just because the DOM mutated — that is
    #  exactly the "blind execution" failure. We block it, note the discrepancy,
    #  and feed it back to the worker (via the guidance bus) to re-evaluate.
    #  Fully additive + flag-gated (V29_REALITY) — off ⇒ identical to V28.
    # ═══════════════════════════════════════════════════════════════════
    try:
        from feature_flags import reality_enabled, reality_llm_enabled
        if reality_enabled() and verb in ("click", "type", "press_enter"):
            from reality import (classify_reality, reconcile_with_llm,
                                 CONTRADICTED, CONFIRMED, UNCLEAR)
            rv = classify_reality(
                expected_change=proposed.get("expected_change", "") or "",
                verb=verb,
                action_outcome=action_outcome,
                pre_text=state.get("dom_markdown", "") or "",
                post_text=post_markdown,
                pre_url=state.get("current_url", ""),
                post_url=post_url,
                critic_success=verdict.success,
            )
            # Deterministic-first: spend a cheap LLM call ONLY on ambiguous deltas.
            if (rv.status == UNCLEAR and reality_llm_enabled()
                    and _JUDGE_CHAIN and _JUDGE_INVOKE_FN):
                rv = await reconcile_with_llm(
                    _JUDGE_INVOKE_FN, _JUDGE_CHAIN, _JUDGE_BREAKER, _JUDGE_HEALTH,
                    objective=state.get("objective", ""),
                    expected_change=proposed.get("expected_change", "") or "",
                    action_outcome=action_outcome,
                    post_text=post_markdown,
                    fallback=rv,
                )
            updates["reality_status"] = rv.status

            if rv.status == CONTRADICTED:
                logger.warning("🚨 Overwatch REALITY: screen contradicts the "
                               "prediction — %s", rv.note[:160])
                # Temporal self-questioning + target lock: a contradiction on the
                # bound target must NOT cause a drift to a neighboring look-alike.
                note = rv.note[:300]
                bt = (state.get("bound_target") or "").strip()
                if bt:
                    note = (note + f" Your bound target is: {bt[:90]}. Do NOT now act "
                            "on a similar control for a DIFFERENT or neighboring item "
                            "— that abandons the goal. Re-examine YOUR target, or halt "
                            "and let the ensemble vote.")
                updates["reality_note"] = note[:500]
                updates["critic_no_progress"] = state.get("critic_no_progress", 0) + 1
                if verb in ("click", "type"):
                    updates["ineffective_streak"] = state.get("ineffective_streak", 0) + 1
                updates["reflexion_triggers"] = state.get("reflexion_triggers", 0) + 1
                # Escalate a DISTINCT tactic and re-evaluate the real screen.
                _escalate(state,
                          "The screen did NOT do what the action predicted.",
                          updates)
                if updates.get("overwatch_verdict") != "rollback":
                    updates["overwatch_verdict"] = "retry"
                return updates

            if rv.status == CONFIRMED:
                # Positive reconciliation — clear any stale discrepancy note.
                updates["reality_note"] = ""
                # V32: Reality confirmed the action worked → clear the intent
                # journal so HESITATION doesn't fire on the next step. Without
                # this, the Worker gets told "predecessor click unconfirmed"
                # even though Overwatch visually confirmed the click succeeded.
                # This was the root cause of the "false hesitation" loop on
                # radio buttons and checkboxes.
                updates["last_attempted_action"] = None
                try:
                    from intent_journal import resolve_intent
                    resolve_intent()
                except Exception:  # noqa: BLE001
                    pass
                if rv.note:
                    logger.info("✅ Overwatch REALITY: prediction confirmed — %s",
                                rv.note[:120])
    except Exception as e:  # noqa: BLE001 — reality monitor never breaks the step
        logger.debug("Reality monitor skipped (non-fatal): %s", e)

    if verdict.success:
        updates["correction_failures"] = 0
        updates["correction_context"] = ""
        updates["critic_progress"] = state.get("critic_progress", 0) + 1
        updates["ineffective_streak"] = 0  # V21: progress clears the visual-trigger
        logger.info("✓ Overwatch L3: %s [%.0f%%]", verdict.reason[:100], verdict.confidence * 100)
    else:
        cf = state.get("correction_failures", 0) + 1
        updates["correction_failures"] = cf
        updates["critic_no_progress"] = state.get("critic_no_progress", 0) + 1
        # V21: a click/type that produced no visible effect — count it so that a
        # short streak escalates to a vision look (the DOM says act, page doesn't).
        if verb in ("click", "type"):
            updates["ineffective_streak"] = state.get("ineffective_streak", 0) + 1

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

    # V29: the action is VERIFIED-effective → resolve the intent journal. There is
    # no ambiguity to carry forward, so the next decision starts with a clean slate.
    if intent_entry is not None:
        updates["last_attempted_action"] = None
        try:
            from intent_journal import resolve_intent
            resolve_intent()
        except Exception:  # noqa: BLE001
            pass

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
#  Layer 4: Outcome verification (for 'done' actions)
# ═══════════════════════════════════════════════════════════════════════════════
#  V20: the gate is now an EVIDENCE-GROUNDED judge over the FRESH page state
#  (objective + planner success-criteria + live DOM + action trail), not the old
#  process-trail heuristics. The heuristics remain ONLY as a fallback when no
#  LLM verdict can be obtained. See outcome_judge.py for the full rationale.

# Judge dependencies — injected once at startup by brain_graph (same pattern as
# mcp_tools.set_page). Without configuration the gate falls back to heuristics.
_JUDGE_INVOKE_FN = None
_JUDGE_CHAIN: list = []
_JUDGE_BREAKER = None
_JUDGE_HEALTH = None


def configure_outcome_judge(invoke_fn, failover_chain, breaker, health_tracker) -> None:
    """Wire the Plan-1 model layer into the done-gate (called at brain startup)."""
    global _JUDGE_INVOKE_FN, _JUDGE_CHAIN, _JUDGE_BREAKER, _JUDGE_HEALTH
    _JUDGE_INVOKE_FN = invoke_fn
    _JUDGE_CHAIN = failover_chain or []
    _JUDGE_BREAKER = breaker
    _JUDGE_HEALTH = health_tracker


def _block_done(state: dict, updates: dict, reason: str,
                correction: str = "") -> dict:
    """Reject a 'done', with a TERMINATION GUARANTEE: after MAX_DONE_BLOCKS
    rejections we stop looping and finalize honestly (mission_success=False).
    The old counter was incremented but never read — block-loops could spin
    forever (observed: done ×5 blocked on the HN run).

    V31 WORKER VETO: If the worker has provided substantive proof_of_completion
    and been rejected 2+ times, force-accept. The Worker has temporal context
    that the static Judge lacks (e.g. saw cart count change, button state flip).
    """
    from outcome_judge import MAX_DONE_BLOCKS
    blocked = state.get("done_blocked", 0) + 1
    updates["done_blocked"] = blocked
    updates["history"] = [{
        "step": state.get("step_number", 0) + 1,
        "action": "done",
        "outcome": f"BLOCKED: {reason[:80]}",
    }]

    # ── V31 WORKER VETO: Confidence Override ──
    # If the worker has fired done 2+ times with substantive proof, the worker's
    # temporal context overrides the judge's static snapshot. This prevents
    # infinite stagnation on successfully completed tasks (e.g. Amazon smart-wagon
    # redirect stripping the product name from DOM).
    proof = (state.get("proposed_action") or {}).get("proof_of_completion", "").strip()
    if blocked >= 2 and len(proof) > 30:
        logger.warning(
            "🏆 Overwatch L4: WORKER VETO — done rejected %d× but worker provided "
            "substantive proof (%d chars). Accepting worker's temporal evidence. "
            "Proof: %s", blocked, len(proof), proof[:200],
        )
        updates["overwatch_verdict"] = "pass"
        updates["mission_success"] = True
        updates["done_evidence"] = f"WORKER VETO (proof accepted after {blocked} rejections): {proof[:300]}"
        return updates

    if blocked >= MAX_DONE_BLOCKS:
        logger.warning(
            "🛑 Overwatch L4: done blocked %d× — finalizing UNVERIFIED (honest failure). "
            "Last gap: %s", blocked, reason[:120],
        )
        updates["overwatch_verdict"] = "escalate"   # routes straight to finalize
        updates["mission_success"] = False
        updates["done_evidence"] = f"UNVERIFIED after {blocked} attempts: {reason[:160]}"
        return updates
    logger.warning("🛡️ Overwatch L4: done BLOCKED (%d/%d) — %s",
                   blocked, MAX_DONE_BLOCKS, reason[:120])
    updates["overwatch_verdict"] = "retry"
    if correction:
        updates["correction_context"] = correction
    return updates


def _render_verified_subgoals(state: dict) -> str:
    """Sub-goals already verified complete during execution, with their captured
    proof — trusted prior evidence handed to the done-judge so it confirms rather
    than re-deriving subtle UI changes from a cold final page. Generalized: read
    straight from the task's own checklist, no website/action rules."""
    # ONLY genuinely-locked sub-goals — never the background audit's unverified
    # "done" guesses (those would bias the judge toward false completion). The
    # done-judge otherwise stays purely evidence-grounded on the live page.
    lines = []
    for d in (state.get("prm_checklist", []) or []):
        if d.get("verified"):
            ev = (d.get("evidence") or "").strip()
            lines.append(f"  ✓ {(d.get('desc') or '').strip()}" + (f" — {ev}" if ev else ""))
    return "\n".join(lines)


async def final_outcome_audit(state: dict, page) -> tuple[bool, str]:
    """Last-chance outcome verification when the agent shuts down WITHOUT a
    verified 'done' (budget exhausted, restrategize cap, …).

    The agent failing to *declare* done does not mean the task failed — on
    sites where the confirmation click misfires, the goal evidence may already
    be on screen (e.g. Flipkart's button switched to 'Go to cart'). One judge
    call here converts those false-failures into verified successes, and makes
    every shutdown report honest: verified-success / verified-failure /
    unverifiable. Returns (achieved, evidence_or_gap)."""
    from outcome_judge import judge_done, should_accept

    page_text = ""
    dom_markdown = state.get("dom_markdown", "")
    url = state.get("current_url", "")
    try:
        url = page.url or url
        import dom_parser
        fresh = await dom_parser.extract(page, timeout=4.0)
        if fresh.get("elements"):
            dom_markdown = fresh.get("markdown", dom_markdown)
        page_text = await page.evaluate(
            "() => (document.body && document.body.innerText || '')"
            ".replace(/\\s+/g, ' ').slice(0, 4500)"
        )
    except Exception as e:
        logger.debug("final audit perception failed (%s) — using last state", e)

    verdict = await judge_done(
        _JUDGE_INVOKE_FN, _JUDGE_CHAIN, _JUDGE_BREAKER, _JUDGE_HEALTH,
        objective=state.get("objective", ""),
        success_criteria=state.get("success_criteria", ""),
        url=url,
        dom_markdown=dom_markdown,
        history_tail=state.get("history_compressed", ""),
        claim="(none — the agent shut down without declaring done)",
        page_text=page_text,
        verified_subgoals=_render_verified_subgoals(state),
        bound_target=state.get("bound_target", ""),
    )
    if verdict is None:
        return False, "unverifiable (judge offline)"
    if should_accept(verdict):
        return True, verdict.evidence[:300]
    return False, (verdict.missing or verdict.evidence)[:300]


async def _layer_4_cove_check(state: dict, page, updates: dict) -> dict:
    """Outcome verification before accepting 'done'."""
    from outcome_judge import judge_done, should_accept, rejection_feedback

    objective = state.get("objective", "")

    # ── Cheap, high-precision pre-block: an auth wall is never success ──
    try:
        url = page.url if page else state.get("current_url", "")
    except Exception:
        url = state.get("current_url", "")
    if any(kw in (url or "").lower() for kw in ("login", "signin", "auth", "signup")):
        return _block_done(
            state, updates, f"Still on auth page ({url[:60]})",
            "\n\n🛡️ DONE REJECTED: you are on a login/auth page — the task cannot "
            "be complete here. Deal with the auth wall or navigate back to the task.",
        )

    # ── Fresh perception: judge the page AS IT IS NOW (a confirmation toast /
    #    cart badge may have appeared after the last snapshot) ──
    dom_markdown = state.get("dom_markdown", "")
    try:
        import dom_parser
        fresh = await dom_parser.extract(page, timeout=4.0)
        if fresh.get("elements"):
            dom_markdown = fresh.get("markdown", dom_markdown)
    except Exception as e:
        logger.debug("L4 fresh snapshot failed (%s) — using last snapshot", e)

    # Visible page text — proof routinely lives in NON-interactive content
    # (confirmation messages, a fact the task asked to find, cart line-items).
    # The element map alone made the judge blind to all of it.
    page_text = ""
    try:
        page_text = await page.evaluate(
            "() => (document.body && document.body.innerText || '')"
            ".replace(/\\s+/g, ' ').slice(0, 4500)"
        )
    except Exception as e:
        logger.debug("L4 page-text read failed (%s)", e)

    proposed = state.get("proposed_action") or {}
    claim = " | ".join(
        s for s in (
            (proposed.get("screen_state", "") or "").strip(),
            (proposed.get("goal_progress", "") or "").strip(),
            (proposed.get("reasoning", "") or "").strip(),
        ) if s
    )

    # V31: Extract worker's proof_of_completion for the judge
    proof_of_completion = (proposed.get("proof_of_completion", "") or "").strip()
    if proof_of_completion:
        logger.info("📋 Worker proof_of_completion: %s", proof_of_completion[:200])

    verdict = await judge_done(
        _JUDGE_INVOKE_FN, _JUDGE_CHAIN, _JUDGE_BREAKER, _JUDGE_HEALTH,
        objective=objective,
        success_criteria=state.get("success_criteria", ""),
        url=url,
        dom_markdown=dom_markdown,
        history_tail=state.get("history_compressed", ""),
        claim=claim,
        page_text=page_text,
        verified_subgoals=_render_verified_subgoals(state),
        bound_target=state.get("bound_target", ""),
        proof_of_completion=proof_of_completion,
    )

    if verdict is not None:
        if should_accept(verdict):
            logger.info("✅ Overwatch L4: outcome VERIFIED — %s", verdict.evidence[:140])
            updates["overwatch_verdict"] = "pass"
            updates["mission_success"] = True
            updates["done_evidence"] = verdict.evidence[:300]
            return updates
        # V29 Sub-Goal Lock: a global rejection must NOT erase verified sub-goals.
        # Re-affirm the locked-done work + name only what REMAINS (Partial Success),
        # and align the plan to the ledger — instead of a bare "missing X" that makes
        # the agent re-do an already-finished sub-goal (the amnesia loop).
        correction = rejection_feedback(verdict)
        try:
            from feature_flags import subgoal_lock_enabled
            if subgoal_lock_enabled() and state.get("prm_checklist"):
                from subgoal_lock import compose_rejection, reconcile_plan_with_ledger
                correction = compose_rejection(verdict.missing, verdict.next_hint,
                                               state["prm_checklist"])
                reconciled = reconcile_plan_with_ledger(state.get("plan_steps"),
                                                        state["prm_checklist"])
                if reconciled is not None:
                    updates["plan_steps"] = reconciled
                    logger.info("🔒 Sub-Goal Lock: plan realigned to the verified ledger "
                                "(locked sub-goals marked done).")
        except Exception as e:  # noqa: BLE001 — lock logic never breaks the gate
            logger.debug("Sub-Goal Lock rejection compose skipped: %s", e)

        return _block_done(
            state, updates,
            verdict.missing or "no on-page proof of completion",
            correction,
        )

    # ── Fallback (judge unavailable): legacy heuristic trail-check ──
    from execution_safety import cove_pre_done_check
    from cognitive_core import PlanState, WorkingMemory

    plan = PlanState(
        mission=objective,
        steps=state.get("plan_steps", []),
        current_step_id=state.get("plan_cursor", 0),
    )
    working_mem = WorkingMemory()
    for h in state.get("history", []):
        working_mem.episodic.append(h)

    cove_ok, cove_reason = await cove_pre_done_check(page, plan, working_mem, objective)

    if not cove_ok:
        return _block_done(state, updates, f"(heuristic) {cove_reason}")

    logger.info("✅ Overwatch L4: heuristic-verified done (judge offline) — %s",
                cove_reason[:60])
    updates["overwatch_verdict"] = "pass"
    updates["mission_success"] = True
    updates["done_evidence"] = f"(heuristic only) {cove_reason[:200]}"
    return updates


# ═══════════════════════════════════════════════════════════════════════════════
#  Action Executor — delegates to MCP tools
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_fail(clean: bool, verb: str, legacy_prefix: str, error: str) -> str:
    """Failure outcome: clean semantic FailureClass when hybrid feedback is on, else
    the legacy string. Both preserve the raw error so downstream detectors work."""
    if not clean:
        return f"→ {legacy_prefix}: {error}"
    from action_feedback import classify_failure, render_failure
    return render_failure(classify_failure(verb, error), error)


async def _execute_action(proposed: dict, page) -> str:
    """Execute a proposed action via MCP tools. Returns an outcome string.

    V29 Phase A: asymmetric verbosity — terse on success (observable effect only,
    no internal strategy name), semantic FailureClass on failure — gated by
    `hybrid_primitives_enabled`. Plus expanded primitives (hover/select_option/
    press_key). Success keeps the "→ OK" prefix and failures keep the raw error so
    the win-state / Reality / Intent-Journal detectors are unaffected.
    """
    verb = proposed.get("verb", "wait")
    try:
        from feature_flags import hybrid_primitives_enabled
        clean = hybrid_primitives_enabled()
    except Exception:
        clean = False

    try:
        if verb == "goto":
            from mcp_tools import mcp_navigate
            result = await mcp_navigate(proposed.get("url", ""))
            if result["success"]:
                return f"→ OK (navigated to {result['url'][:60]})"
            return _fmt_fail(clean, verb, "FAILED", result["error"])

        elif verb == "click":
            from mcp_tools import mcp_click
            result = await mcp_click(
                x=proposed.get("x", 0),
                y=proposed.get("y", 0),
                element_id=proposed.get("element_id"),
            )
            if result["success"]:
                bits = []
                if result.get("navigated"):
                    bits.append("navigated")
                if result.get("dom_changed"):
                    bits.append("DOM changed")
                if clean:  # terse, NO internal strategy name
                    return "→ OK" + (f" ({', '.join(bits)})" if bits else "")
                return (f"→ OK (click via {result['strategy']}"
                        f"{',' + ' navigated' if result['navigated'] else ''}"
                        f"{',' + ' DOM changed' if result['dom_changed'] else ''})")
            return _fmt_fail(clean, verb, "CLICK INEFFECTIVE", result["error"])

        elif verb == "type":
            from mcp_tools import mcp_type
            result = await mcp_type(
                text=proposed.get("text", ""),
                x=proposed.get("x", 0),
                y=proposed.get("y", 0),
                element_id=proposed.get("element_id"),
            )
            if result["success"]:
                if clean:
                    return f"→ OK (typed {result['actual_length']} chars)"
                return f"→ OK (typed {result['actual_length']} chars via {result['strategy']})"
            return _fmt_fail(clean, verb, "TYPE FAILED", result["error"])

        elif verb == "scroll":
            from mcp_tools import mcp_scroll
            result = await mcp_scroll(600)
            if not result["success"]:
                return _fmt_fail(clean, verb, "FAILED", result["error"])
            moved = int(result.get("scrolled_px", 0) or 0)
            if abs(moved) < 5:
                return "→ NO-OP (page did not move — already at bottom or unscrollable)"
            if result.get("at_bottom"):
                return f"→ OK (scrolled {moved}px, reached page bottom)"
            return f"→ OK (scrolled {moved}px)"

        elif verb == "press_enter":
            from mcp_tools import mcp_press_enter
            result = await mcp_press_enter()
            return "→ OK" if result["success"] else _fmt_fail(clean, verb, "FAILED", result["error"])

        elif verb == "hover":
            from mcp_tools import mcp_hover
            result = await mcp_hover(element_id=proposed.get("element_id"),
                                     x=proposed.get("x", 0), y=proposed.get("y", 0))
            return "→ OK (hovered)" if result["success"] else _fmt_fail(clean, verb, "HOVER FAILED", result["error"])

        elif verb == "select_option":
            from mcp_tools import mcp_select_option
            result = await mcp_select_option(proposed.get("element_id"), proposed.get("text", "") or "")
            if result["success"]:
                return f"→ OK (selected '{result.get('selected', '')}')"
            return _fmt_fail(clean, verb, "SELECT FAILED", result["error"])

        elif verb == "press_key":
            from mcp_tools import mcp_press_key
            result = await mcp_press_key(proposed.get("text", "") or proposed.get("key", ""))
            return "→ OK" if result["success"] else _fmt_fail(clean, verb, "FAILED", result["error"])

        elif verb == "wait":
            from mcp_tools import mcp_wait
            result = await mcp_wait(proposed.get("wait_ms", 800))
            return f"→ OK (waited {result.get('waited_ms', 800)}ms)"

        else:
            return f"→ UNKNOWN verb: {verb}"

    except Exception as e:
        logger.warning("Action execution failed: %s", e)
        return f"→ CRASHED: {str(e)[:100]}"
