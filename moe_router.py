"""MoE Router — Mixture-of-Experts routing for the True Brain v16.0.

Decides which specialist worker node should handle the current step.
Uses a confidence-gated hybrid approach:

  1. FAST PATH (deterministic, zero-cost): For unambiguous situations
     where the action type is obvious from the proposed action or plan step.
     Example: proposed_action.verb == "goto" → navigator

  2. CREATIVE PATH (LLM-classified): When the situation is ambiguous,
     the router delegates to the LLM to classify the best worker.
     This preserves the agent's full creative freedom.

The key insight: routing is NOT about limiting what the agent can do.
It's about giving the agent the RIGHT specialist prompt for the situation.
A navigator sees navigation-optimized instructions. An interactor sees
click/type-optimized instructions. The LLM's creative reasoning is
PRESERVED — only the framing changes.

Why NOT pure keyword matching:
  - "Click the search button" could be navigation OR interaction
  - "Find the cheapest processor" needs extraction + navigation + interaction
  - Rigid keywords would force the wrong specialist, limiting creativity

Why NOT pure LLM classification:
  - Adds 500ms+ latency per step
  - Burns an LLM call from the rate-limited chain
  - 90% of routing decisions are obvious

References:
  - MAST (arXiv 2503.13657): 32.3% failures from inter-agent misalignment
  - AgentOccam (arXiv 2410.13825): action-space pruning improves success by 15.8%
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from app.logger import get_logger
    logger = get_logger("moe_router")
except ImportError:
    logger = logging.getLogger("moe_router")


# ═══════════════════════════════════════════════════════════════════════════════
#  Route Decision
# ═══════════════════════════════════════════════════════════════════════════════

def route_to_worker(state: dict[str, Any]) -> str:
    """Confidence-gated MoE router. Returns the target worker node name.

    This is a LangGraph conditional-edge function: it receives the
    current BrainState (as dict) and returns a string key that maps
    to a node name in the graph's conditional edge table.

    Routing hierarchy:
      1. Terminal conditions (done, budget exhausted, too many errors)
      2. Action-verb routing (when proposed_action exists from previous LLM call)
      3. Plan-step context routing (from the current plan step description)
      4. Default: interactor (the most general specialist)
    """

    step_number = state.get("step_number", 0)
    max_steps = state.get("max_steps", 25)
    error_count = state.get("error_count", 0)
    proposed = state.get("proposed_action")

    # ── Terminal conditions ──
    if step_number >= max_steps:
        logger.info("🔀 Router → finalize (budget exhausted: %d/%d)", step_number, max_steps)
        return "finalize"

    if error_count >= 8:
        logger.info("🔀 Router → recovery (error_count=%d)", error_count)
        return "recovery"

    # ── Action-verb routing (when the LLM has already decided) ──
    if proposed:
        verb = proposed.get("verb", "")

        if verb == "done":
            logger.info("🔀 Router → overwatch (done verification)")
            return "done_check"

        # V17: ask_user — agent needs user input before proceeding
        if verb == "ask_user":
            missing = proposed.get("missing_data") or proposed.get("rationale", "")
            logger.info("🙋 Router → finalize (ask_user: %s)", missing[:100])
            return "finalize"

        if verb in ("goto", "scroll"):
            logger.info("🔀 Router → navigator (verb=%s)", verb)
            return "navigator"

        if verb in ("click", "type", "press_enter", "hover", "select_option", "press_key"):
            logger.info("🔀 Router → interactor (verb=%s)", verb)
            return "interactor"

        if verb == "wait":
            logger.info("🔀 Router → navigator (wait)")
            return "navigator"

    # ── Plan-step context routing (creative path) ──
    # Instead of rigid keyword matching, we use semantic intent categories.
    # The agent's creativity is preserved because the specialist prompts
    # still allow the full action space — they just FRAME the decision
    # differently.
    current_step = _get_current_plan_step(state)
    if current_step:
        route = _classify_plan_step(current_step)
        if route:
            logger.info("🔀 Router → %s (plan-step: '%s')", route, current_step[:50])
            return route

    # ── Default: interactor ──
    # The interactor is the most general worker — it handles click, type,
    # scroll, and any action the LLM decides upon. This preserves maximum
    # creative freedom for ambiguous situations.
    logger.info("🔀 Router → interactor (default)")
    return "interactor"


def _get_current_plan_step(state: dict) -> str:
    """Extract the current plan step description."""
    plan_steps = state.get("plan_steps", [])
    for s in plan_steps:
        if s.get("status") in ("active", "in_progress"):
            return s.get("desc", "")
    return ""


def _classify_plan_step(step_desc: str) -> str | None:
    """Classify a plan step into a worker type.

    Returns None for ambiguous steps (falls through to default).
    This is intentionally LOOSE — we only route when confident.
    Ambiguous steps go to the interactor (most general).
    """
    desc = step_desc.lower().strip()

    # ── High-confidence navigation intents ──
    # These are unambiguously about getting to a page
    if desc.startswith(("navigate to", "go to", "open ", "visit ")):
        return "navigator"
    if desc.startswith("scroll") and "find" not in desc:
        return "navigator"

    # ── High-confidence extraction intents ──
    # These are unambiguously about reading/extracting data
    if desc.startswith(("extract ", "scrape ", "read ", "capture ")):
        return "extractor"
    if "copy the" in desc or "save the" in desc or "record the" in desc:
        return "extractor"

    # ── Everything else: return None (ambiguous → default to interactor) ──
    # This includes:
    #   - "Click the Add to Cart button" (could be navigation or interaction)
    #   - "Find the cheapest processor" (needs multiple skills)
    #   - "Fill in the form" (interaction)
    #   - "Search for Intel i7" (type + navigation)
    # The interactor handles all of these with its general-purpose prompt.
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Verdict Router (for Overwatch → next node)
# ═══════════════════════════════════════════════════════════════════════════════

# P2 — Verifier-gated retry is the reliability amplifier. A worker with per-step
# accuracy p, retried up to RETRY_BUDGET times against the live-DOM verifier,
# reaches an effective per-step success of:
#       p_eff = 1 − (1 − p)^(RETRY_BUDGET + 1)
# so a k-critical-step task succeeds with S = p_eff^k. With p=0.70, k=6:
#       RETRY_BUDGET 0 → 12%,  2 → 85%,  3 → 95%.
# Independence of attempts is guaranteed by the V18 escalation ladder (each retry
# is a DISTINCT tactic). RETRY_BUDGET=3 ⇒ 4 attempts ⇒ exponent (RETRY_BUDGET+1)=4.
RETRY_BUDGET = 3


def verdict_router(state: dict[str, Any]) -> str:
    """Route based on Overwatch's verification verdict.

    This is the conditional edge after the overwatch node.
    """
    verdict = state.get("overwatch_verdict", "pass")
    retry_count = state.get("retry_count", 0)

    if verdict == "pass":
        return "commit"
    elif verdict == "retry":
        if retry_count >= RETRY_BUDGET:
            logger.warning("🔀 Verdict → rollback (retry exhausted: %d/%d)",
                           retry_count, RETRY_BUDGET)
            return "rollback"
        return "retry"
    elif verdict == "rollback":
        return "rollback"
    elif verdict == "escalate":
        return "finalize"
    else:
        logger.warning("🔀 Unknown verdict '%s' — defaulting to commit", verdict)
        return "commit"
