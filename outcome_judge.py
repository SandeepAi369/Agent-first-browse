"""Outcome Judge — evidence-grounded verification of task completion (V20).

THE problem this solves
═══════════════════════
The old done-gate (`cove_pre_done_check`) verified the PROCESS TRAIL, never the
OUTCOME: plan-step counters (advanced by a trivial '"OK" in outcome' heuristic),
a progress percentage, a URL-not-login check, and keyword presence in history.
It never once looked at what the page actually shows, and the explicit
`success_criteria` ("Done when: …") computed by the strategic planner was never
given to it. Consequences observed in live runs:
  • FALSE BLOCK — the task was truly complete but the counters said otherwise →
    'done' rejected repeatedly with non-actionable reasons → done-spam / wandering
    (HN run: done ×5 blocked; Flipkart: Buy-Now loop after the cart add succeeded).
  • FALSE PASS — heuristics trivially satisfied mid-task → mission "success"
    without ever checking the outcome → the agent shuts down unverified.

What this module does instead (browser-use judge + "Are We Done Yet?" pattern):
a single SKEPTICAL LLM verdict over the FRESH page state — objective + planner
success-criteria + fresh DOM evidence + action trail + the agent's claim — with
a BINARY achieved/not-achieved answer (binary beats rubric scores in practice),
concrete cited evidence, and, on rejection, actionable feedback the worker can
use on its very next step.

Design constraints:
  • Typed, strict-safe schema (Plan-1 model layer) via the existing failover chain.
  • Judge unavailable (no LLM / call fails) → caller falls back to the legacy
    heuristic gate — strict superset, no regression.
  • Pure decision helpers, unit-testable without a browser or LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

try:
    from app.logger import get_logger
    logger = get_logger("outcome_judge")
except ImportError:
    logger = logging.getLogger("outcome_judge")


# ═══════════════════════════════════════════════════════════════════════════════
#  Verdict schema (flat + strict-safe)
# ═══════════════════════════════════════════════════════════════════════════════

class DoneVerdict(BaseModel):
    """Binary, evidence-cited verdict on whether the objective is achieved."""

    achieved: bool = Field(
        description="True ONLY if the page evidence proves the objective is met."
    )
    confidence: float = Field(
        description="0.0-1.0 — how strong the cited evidence is."
    )
    evidence: str = Field(
        description=(
            "The CONCRETE on-page proof for your verdict: quote the exact element "
            "label / page text / URL fragment that shows success (or its absence)."
        )
    )
    missing: str = Field(
        default="",
        description=(
            "If not achieved: exactly what proof is absent from the page "
            "(e.g. 'cart page does not list the product'). Empty when achieved."
        )
    )
    next_hint: str = Field(
        default="",
        description=(
            "If not achieved: ONE concrete browser action that would surface the "
            "missing proof (e.g. 'open the cart page and check the item is listed'). "
            "Empty when achieved."
        )
    )


# How many times a 'done' may be rejected before we stop looping and finalize
# honestly (mission_success=False). Termination guarantee — the old counter was
# incremented but never read, so block-loops could spin forever.
MAX_DONE_BLOCKS = 4

# Minimum judge confidence to accept an 'achieved' verdict.
MIN_ACCEPT_CONFIDENCE = 0.6


# ═══════════════════════════════════════════════════════════════════════════════
#  Judge prompt
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """You are a task-completion verifier for a browser agent.
The agent claims its task is finished. Your job is to decide whether the objective
is truly achieved, using ALL available evidence.

CRITICAL WORKER-DEFERENCE RULE (V31)
═══════════════════════════════════════
The Worker agent has TEMPORAL CONTEXT that you do not have. It observed the page
BEFORE, DURING, and AFTER each action — you only see a frozen snapshot AFTER.

If the Worker provides a PROOF OF COMPLETION with concrete, specific state-change
evidence (e.g. 'cart count increased from 0 to 1', 'button changed from Star to
Unstar', 'confirmation toast appeared', 'form submitted and redirected to
thank-you page'), you MUST EVALUATE THE WORKER'S PROOF FIRST:

  • If the proof describes a PLAUSIBLE, SPECIFIC state-change that is consistent
    with the task objective AND there is no CONTRADICTING evidence on the current
    page → RETURN ACHIEVED=TRUE. You do not need to re-find the exact product
    name or item text in the current DOM. The Worker witnessed it happen live.
  • Only REJECT the worker's proof if it CLEARLY contradicts hard evidence on
    the current page (e.g. worker says 'added to cart' but cart badge explicitly
    shows 0, or the page shows an error message, or the URL is a login/error page).
  • Post-action redirect pages (cart popups, confirmation screens, thank-you
    pages, smart-wagon overlays) routinely strip the original item text. This is
    NORMAL behavior and NOT grounds for rejection.

VERIFICATION RULES
1. THINK SITUATIONALLY FIRST: before looking at the page, reason about what
   proof THIS PARTICULAR task would leave behind when truly complete. Every kind
   of task has its own natural evidence — derive it from the objective, do not
   apply a fixed checklist. Examples of the reasoning (not an exhaustive list):
   • "add X to cart"      → cart badge/count increased, "Go to cart" state, or
                            the cart page listing X
   • "post/publish X"     → the published post/comment visible on the page or feed
   • "find/report a fact" → the fact itself visible in the page text
   • "fill/submit a form" → a submission confirmation, NOT the filled form
   • "download/upload"    → the completed-state indicator the site shows
   Then check whether THAT proof is present in the evidence below.
2. Evaluate evidence in this priority order:
   a) WORKER'S PROOF OF COMPLETION (temporal, first-hand witness testimony)
   b) VERIFIED SUB-GOALS (already confirmed during execution)
   c) FRESH PAGE STATE (URL + DOM + visible text — static snapshot)
   The worker's proof and verified sub-goals have higher weight than a cold
   static snapshot, because the snapshot may be a post-redirect page that no
   longer shows the action's immediate result.
3. Cite concrete proof: quote the worker's testimony, element labels, page text,
   or URL fragments that demonstrate success.
4. Know the classic traps — these are NOT success:
   • a filled form is not a SUBMITTED form
   • a product page is not an item IN THE CART
   • a compose box with text is not a PUBLISHED post/comment
   • a search results page is not an OPENED result
   But be careful: these traps apply to the ACTION, not the page state AFTER.
   If the worker says 'I submitted the form and saw a confirmation toast', and
   the current page is a thank-you page, that IS success — don't reject because
   the form is no longer visible.
5. Distinguish the goal action from look-alikes (e.g. "Buy Now" / checkout flows
   are NOT "add to cart"; a login wall means the goal was NOT reached).
6. Be decisive and binary. No partial credit. When proof is present, say achieved
   with the citation; when it is not, say not achieved with what is missing.
"""


def build_judge_messages(
    *,
    objective: str,
    success_criteria: str,
    url: str,
    dom_markdown: str,
    history_tail: str,
    claim: str,
    page_text: str = "",
    verified_subgoals: str = "",
    bound_target: str = "",
    proof_of_completion: str = "",
) -> list[dict]:
    """Build the judge conversation (pure — unit-testable).

    page_text is the rendered body text — essential evidence the interactive
    element map cannot carry (confirmation toasts, facts the task asked to find,
    cart line-items). Without it the judge is blind to every non-clickable proof.

    verified_subgoals lists sub-goals that were ALREADY verified complete DURING
    execution (with their captured proof). These are trusted prior evidence — the
    judge should confirm the objective using them, not re-derive every sub-goal
    from a cold final page (which causes false 'not done' on subtle UI changes).

    proof_of_completion (V31) is the worker's first-hand testimony of state-changes
    it observed during execution. This is TRUSTED temporal evidence — the judge
    evaluates it with deference, rejecting only on clear contradiction.
    """
    user = (
        f"═══ OBJECTIVE ═══\n{objective}\n\n"
        + (f"═══ 🎯 TARGET FOCUS (the specific item/entity this task is about) ═══\n"
           f"{bound_target}\nVerify the objective is achieved for THIS specific "
           f"target — never accept proof that belongs to a similar but DIFFERENT "
           f"item/element.\n\n" if bound_target else "")
        + (f"═══ PLANNED SUCCESS CRITERIA (done when) ═══\n{success_criteria}\n\n"
           if success_criteria else "")
        + (f"═══ WORKER'S PROOF OF COMPLETION (trusted temporal evidence) ═══\n"
           f"{proof_of_completion}\n"
           f"The Worker observed these state-changes LIVE during execution. Evaluate "
           f"this proof against the objective. Accept it unless it CLEARLY contradicts "
           f"the current page evidence below.\n\n" if proof_of_completion else "")
        + (f"═══ SUB-GOALS ALREADY VERIFIED DURING EXECUTION (trusted proof) ═══\n"
           f"{verified_subgoals}\nTreat these as established — confirm the objective "
           f"in light of them; do not mark the task incomplete merely because this "
           f"final page no longer re-shows their proof.\n\n" if verified_subgoals else "")
        + f"═══ AGENT'S CLAIM (NOT evidence — use only for context) ═══\n{claim or '(none)'}\n\n"
        f"═══ RECENT ACTION TRAIL ═══\n{history_tail or '(none)'}\n\n"
        f"═══ FRESH PAGE STATE (static snapshot — may not show post-redirect proof) ═══\n"
        f"URL: {url}\n\n"
        f"── Interactive elements ──\n{dom_markdown[:5000]}\n\n"
        + (f"── Visible page text ──\n{page_text[:4500]}\n\n" if page_text else "")
        + "Verdict: is the objective ACHIEVED, considering ALL evidence above "
        "(worker proof + sub-goals + page state)?"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Judge invocation (uses the Plan-1 failover layer; never raises)
# ═══════════════════════════════════════════════════════════════════════════════

async def judge_done(
    invoke_fn,
    failover_chain: list,
    breaker,
    health_tracker,
    *,
    objective: str,
    success_criteria: str = "",
    url: str = "",
    dom_markdown: str = "",
    history_tail: str = "",
    claim: str = "",
    page_text: str = "",
    verified_subgoals: str = "",
    bound_target: str = "",
    proof_of_completion: str = "",
) -> DoneVerdict | None:
    """Run the outcome judge. Returns None when no verdict could be obtained
    (caller falls back to the legacy heuristic gate)."""
    if invoke_fn is None or not failover_chain:
        return None

    from langchain_core.messages import HumanMessage, SystemMessage
    raw = build_judge_messages(
        objective=objective, success_criteria=success_criteria, url=url,
        dom_markdown=dom_markdown, history_tail=history_tail, claim=claim,
        page_text=page_text, verified_subgoals=verified_subgoals,
        bound_target=bound_target, proof_of_completion=proof_of_completion,
    )
    messages = [SystemMessage(content=raw[0]["content"]),
                HumanMessage(content=raw[1]["content"])]

    try:
        verdict, used_model = await invoke_fn(
            failover_chain, messages, DoneVerdict,
            breaker, health_tracker=health_tracker,
        )
        if verdict is None:
            return None
        logger.info(
            "⚖️ Outcome judge (%s): achieved=%s conf=%.2f — %s",
            used_model, verdict.achieved, verdict.confidence,
            (verdict.evidence or verdict.missing)[:100],
        )
        return verdict
    except Exception as e:
        logger.warning("Outcome judge unavailable (%s) — falling back to heuristics",
                       str(e)[:120])
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure decision helpers
# ═══════════════════════════════════════════════════════════════════════════════

def should_accept(verdict: DoneVerdict, min_confidence: float = MIN_ACCEPT_CONFIDENCE) -> bool:
    """Accept 'done' only on a confident, evidence-backed achieved verdict."""
    return bool(verdict.achieved) and float(verdict.confidence) >= min_confidence


def rejection_feedback(verdict: DoneVerdict) -> str:
    """Actionable correction injected into the worker's next prompt.

    The old gate rejected 'done' with process-trail reasons the worker could not
    act on ("Plan incomplete: 3 steps remaining") — so it just re-emitted 'done'.
    This gives it the judge's page-grounded gap + one concrete next move.
    """
    missing = (verdict.missing or "the page does not show proof of completion").strip()
    hint = (verdict.next_hint or "surface the proof on screen, then declare done").strip()
    return (
        "\n\n🛡️ DONE REJECTED — outcome verification failed.\n"
        f"MISSING PROOF: {missing}\n"
        f"DO THIS NOW: {hint}\n"
        "Only output 'done' again once the missing proof is VISIBLE on the page."
    )
