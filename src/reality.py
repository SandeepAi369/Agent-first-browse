"""Reality Monitor — Screen-Reality Reconciliation (V29 / Phase 1, Mandate 1).

THE BLIND-EXECUTION CURE
════════════════════════
Before every action the worker emits `expected_change` — a forward model of the
exact observable effect ("the button flips to 'Starred' and the count
increments"). Until now that prediction was generated, logged, and then THROWN
AWAY: the only post-action check (CriticV12) answers "did *anything* change?"
(binary progress), never "did the change we PREDICTED happen — or did the page do
something *else*?"

That gap is the root of "blind execution": a click that pops an error toast, hits
an out-of-stock state, or redirects to a login wall still makes the DOM change, so
the agent reads "progress", commits, and marches on — acting mechanically against
a screen that is actually telling it "no".

The Reality Monitor classifies the post-action page into three states:

  • CONFIRMED    — the predicted change is visibly present  → strong, trustworthy
                   progress (the success path proceeds).
  • CONTRADICTED — a *different*, adverse change happened (error / invalid /
                   rejected / unexpected redirect / captcha / out-of-stock)
                   → DO NOT commit as progress. Halt, note the discrepancy, and
                   feed it back to the worker so it re-evaluates the REAL screen.
  • NULL/UNCLEAR — nothing changed, or changed-but-ambiguous → existing behavior
                   (ineffective-streak path / optional cheap LLM reconcile).

DESIGN
  • Pure, deterministic, side-effect-free → unit-testable offline (no network).
  • Compares against the worker's OWN pre-committed prediction, never a free
    re-judgement — this sidesteps MLLM "agreement bias" (Self-Grounded
    Verification, arXiv 2507.11662), where a verifier rationalizes whatever it is
    shown.
  • CONTRADICTION only ESCALATES (a cheap re-evaluation), never terminates — so a
    rare false-positive costs one extra think, not a failed task. Conservative by
    construction.

References: InferAct / preemptive misaligned-action detection (arXiv 2407.11843);
Grounded Test-Time Adaptation (arXiv 2511.04847); CRITIC (external grounding for
self-verification).
"""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass, field

logger = logging.getLogger("reality")

# ── Status constants ──
CONFIRMED = "CONFIRMED"
CONTRADICTED = "CONTRADICTED"
UNCLEAR = "UNCLEAR"
NULL = "NULL"

# Reconciliation is meaningful only for actions with a predicted UI effect.
_RECONCILABLE_VERBS = ("click", "type", "press_enter")

# Distinctive ADVERSE indicators — their NEW appearance (absent before the action,
# present after) is strong evidence the page rejected/failed the action rather than
# doing what was predicted. Kept literal + task-agnostic; only counted when NEW, so
# a word already on the page never trips a false contradiction.
CONTRADICTION_PATTERNS: tuple[str, ...] = (
    "error", "invalid", "incorrect", "failed", "failure", "went wrong",
    "try again", "couldn't", "could not", "cannot ", "unable to", "not allowed",
    "not available", "unavailable", "out of stock", "sold out", "no results",
    "not found", "page not found", "404", "access denied", "denied", "rejected",
    "expired", "session expired", "captcha", "are you a robot", "verify you are",
    "verify that you", "unusual traffic", "too many requests", "forbidden",
    "permission denied", "something went wrong", "please try", "is required",
    "required field", "must be filled", "enter a valid",
)

# Affirmative markers — their NEW appearance corroborates a successful, predicted
# outcome (used alongside predicted-token overlap, not alone).
AFFIRM_PATTERNS: tuple[str, ...] = (
    "added to", "in your cart", "in your bag", "added to cart", "added to bag",
    "successfully", "success", "confirmed", "confirmation", "thank you",
    "starred", "unstar", "following", "posted", "your post", "submitted",
    "published", "order placed", "saved", "subscribed", "go to cart",
    "view cart", "proceed to", "added to wishlist",
)

# Auth-wall hints in a URL — an UNEXPECTED redirect to one of these contradicts a
# normal task action (you clicked 'Add to cart' and landed on a login page).
_AUTH_URL_HINTS = ("login", "signin", "sign-in", "auth", "signup", "sign-up",
                   "account/login")

_STOPWORDS = set(
    "the a an of to and or in on at is are be was were this that with for from "
    "your you will into onto then now once when after before page will should "
    "shows show appear appears change changes become becomes click clicks button "
    "element state new it its their there here also more most".split()
)


@dataclass
class RealityVerdict:
    """Outcome of reconciling the predicted change against the live screen."""
    status: str = NULL
    note: str = ""               # human-readable discrepancy (CONTRADICTED only)
    confidence: float = 0.0
    matched: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def _content_tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", (s or "").lower())
            if w not in _STOPWORDS}


def _new_hits(patterns: tuple[str, ...], pre: str, post: str) -> list[str]:
    """Patterns present in `post` but NOT in `pre` (i.e., the action introduced them)."""
    return [p for p in patterns if p in post and p not in pre]


def classify_reality(
    *,
    expected_change: str,
    verb: str,
    action_outcome: str = "",
    pre_text: str = "",
    post_text: str = "",
    pre_url: str = "",
    post_url: str = "",
    url_changed: bool | None = None,
    critic_success: bool = True,
) -> RealityVerdict:
    """Deterministically reconcile the worker's prediction with the live screen.

    Inputs are all already available in Overwatch after execution (the predicted
    change, the pre/post DOM text + URL, and CriticV12's binary verdict). Returns
    a RealityVerdict; only CONTRADICTED changes control flow upstream.
    """
    verb = (verb or "").strip().lower()
    if verb not in _RECONCILABLE_VERBS:
        return RealityVerdict(NULL, "", 0.0)

    pre = _norm(pre_text)
    post = _norm(post_text)
    pre_u = _norm(pre_url)
    post_u = _norm(post_url)
    exp = _norm(expected_change)
    if url_changed is None:
        url_changed = bool(post_u) and post_u != pre_u

    # ── 1. CONTRADICTION — adverse signal the action did NOT predict ──
    neg = _new_hits(CONTRADICTION_PATTERNS, pre, post)
    # An unexpected redirect to an auth wall (unless the task/prediction is auth).
    expects_auth = any(h in exp for h in
                       ("log in", "login", "sign in", "sign up", "account", "credential"))
    auth_redirect = (
        any(h in post_u for h in _AUTH_URL_HINTS)
        and not any(h in pre_u for h in _AUTH_URL_HINTS)
        and not expects_auth
    )
    if neg or auth_redirect:
        bits = list(neg)
        if auth_redirect:
            bits.append(f"redirected to an auth page ({post_url[:50]})")
        note = (
            f"Predicted: '{(expected_change or 'a successful effect').strip()[:90]}'. "
            f"Instead the page now shows: {', '.join(bits[:3])[:120]}."
        )
        return RealityVerdict(CONTRADICTED, note, 0.8, matched=bits[:5])

    # No prediction to verify against → nothing to confirm/contradict.
    if not exp:
        return RealityVerdict(UNCLEAR if critic_success else NULL, "", 0.0)

    # ── 2. CONFIRMATION — predicted effect visibly present ──
    pred_tokens = _content_tokens(expected_change)
    newly_present = pred_tokens & (_content_tokens(post_text) - _content_tokens(pre_text))
    predicts_nav = any(k in exp for k in
                       ("redirect", "navigat", "detail page", "product page",
                        "open the", "go to the", "load", "the url"))
    affirm = _new_hits(AFFIRM_PATTERNS, pre, post)

    if (predicts_nav and url_changed) or len(newly_present) >= 2 or affirm:
        why = []
        if predicts_nav and url_changed:
            why.append("predicted navigation occurred")
        if len(newly_present) >= 2:
            why.append(f"predicted terms appeared ({', '.join(sorted(newly_present)[:3])})")
        if affirm:
            why.append(f"affirmative state ({', '.join(affirm[:2])})")
        return RealityVerdict(CONFIRMED, "; ".join(why), 0.75,
                              matched=sorted(newly_present)[:5] + affirm[:2])

    # ── 3. Changed but cannot tell deterministically → ambiguous ──
    return RealityVerdict(UNCLEAR if critic_success else NULL, "", 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Optional cheap LLM reconcile — ONLY for genuinely ambiguous deltas (UNCLEAR)
# ═══════════════════════════════════════════════════════════════════════════════
#  Per the approved design: deterministic-first, with a single cheap auxiliary
#  call reserved for the cases the deterministic pass cannot resolve. Reuses the
#  already-wired judge chain; never raises; falls back to the deterministic verdict.

async def reconcile_with_llm(
    invoke_fn,
    chain: list,
    breaker,
    health_tracker,
    *,
    objective: str,
    expected_change: str,
    action_outcome: str,
    post_text: str,
    fallback: RealityVerdict,
) -> RealityVerdict:
    """Resolve an UNCLEAR delta with one cheap structured call. Returns `fallback`
    on any failure or if no chain is configured (no regression)."""
    if invoke_fn is None or not chain:
        return fallback

    from pydantic import BaseModel, ConfigDict, Field
    from langchain_core.messages import SystemMessage, HumanMessage

    class RealityCheck(BaseModel):
        model_config = ConfigDict(extra="forbid")
        matches_prediction: bool = Field(
            description="True if the page AFTER the action shows the predicted change.")
        contradiction: bool = Field(
            description=("True if something ADVERSE/unexpected happened instead "
                         "(error, rejection, validation failure, captcha, wrong "
                         "page, out-of-stock, unexpected redirect)."))
        note: str = Field(default="",
                          description="One short phrase describing what the page actually did.")

    system = SystemMessage(content=(
        "You are a strict screen-reality auditor for a browser agent. The agent "
        "PREDICTED a specific change before acting. Judge ONLY from the page text "
        "AFTER the action: did the predicted change actually happen, or did "
        "something contradictory/adverse happen instead? Be literal and skeptical "
        "— a page merely changing is NOT proof the prediction came true."
    ))
    user = HumanMessage(content=(
        f"OBJECTIVE: {objective[:300]}\n"
        f"PREDICTED CHANGE: {expected_change[:300]}\n"
        f"ACTION RESULT: {action_outcome[:200]}\n"
        f"PAGE AFTER (text): {post_text[:2500]}\n\n"
        "Did the predicted change happen, or did something contradictory occur?"
    ))
    try:
        res, _model = await invoke_fn(chain, [system, user], RealityCheck,
                                      breaker, health_tracker=health_tracker)
        if getattr(res, "contradiction", False):
            return RealityVerdict(
                CONTRADICTED,
                f"Predicted: '{expected_change.strip()[:90]}'. "
                f"Audit: {(getattr(res, 'note', '') or 'the page did not do that').strip()[:120]}.",
                0.7)
        if getattr(res, "matches_prediction", False):
            return RealityVerdict(CONFIRMED, getattr(res, "note", "")[:120], 0.7)
        return fallback
    except Exception as e:  # noqa: BLE001 — reconcile must never break the step
        logger.debug("reality LLM reconcile failed (%s) — keeping deterministic verdict", e)
        return fallback
