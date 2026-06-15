"""Consensus — multi-model voting + abstention for IRREVERSIBLE actions (V27 / P2).

The reliability amplifier. Verifier-gated retry already converts a weak per-step
model into a near-perfect executor on REVERSIBLE setup steps:

    S = [1 − (1−p)^(r+1)]^k          (p=0.70, k=6: r=0→12%, r=2→85%, r=3→95%)

But an IRREVERSIBLE action (place order, submit, star, post) CANNOT be retried —
one wrong commit is unrecoverable. So for exactly those actions we replace
"trust one model" with a CONSENSUS of independent voters and ABSTAIN when they
disagree:

    Condorcet:  P_maj(n,p) = Σ_{i>n/2} C(n,i) p^i (1−p)^(n−i) → 1   (p>½)
                P_maj(3, 0.70) = 0.784
    CISC:       confidence-weighted vote ⇒ same accuracy as plain self-consistency
                with ~46% fewer samples; abstain when agreement < θ.

Diversity note: the worker chain runs at temperature 0, so re-sampling ONE model
is useless. Voters here are DISTINCT base-models already in the chain
(gpt-oss-120b, gemma-4-31b-it, gpt-oss-20b, …) — a true ensemble that catches
model-specific errors, not just sampling noise. Single-model (premium) mode has
<2 distinct models → consensus auto-skips (that model is trusted/strong).

Pure logic + a thin async sampler that REUSES the existing `invoke_fn` with a
1-element chain — no new model plumbing, no change to the §6-protected layers.

References: Six Sigma Agent (consensus-driven critical steps, arXiv 2601.22290);
CISC (arXiv/ACL 2025); voting ensembles with abstention (arXiv 2510.04048).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("consensus")

# ── Tunables (env-overridable) ──
CONSENSUS_N = 3        # max independent voters (distinct base models) in the cascade
THETA_AGREE = 0.6      # min winning weight-share to ACT on an irreversible action
THETA_CONF = 0.5       # min mean self-confidence of the winning voters
DEFAULT_CONF = 0.7     # assumed confidence when a model omits the field
HIGH_CONF = 0.85       # primary executes IMMEDIATELY at/above this (if structurally sound)


def consensus_enabled() -> bool:
    """CONSENSUS_ENABLED env (default ON). 0/false/no/off disables."""
    return os.getenv("CONSENSUS_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def canonical_action_key(verb: str | None, element_id: str | None,
                         text: str | None) -> str:
    """Identity of an action for vote-bucketing (verb + target + typed text)."""
    return (
        f"{(verb or '').strip().lower()}|"
        f"{(element_id or '').strip()}|"
        f"{(text or '').strip()[:40]}"
    )


@dataclass
class VoteResult:
    """Outcome of a confidence-weighted vote over candidate actions."""
    verb: str = ""
    element_id: str | None = None
    text: str | None = None
    agreement: float = 0.0      # weight(winner) / total weight  ∈ [0,1]
    mean_conf: float = 0.0      # mean self-confidence of the winning voters
    n_votes: int = 0
    n_unique: int = 0
    tally: dict[str, float] = field(default_factory=dict)
    winner_sample: dict[str, Any] = field(default_factory=dict)


def weighted_vote(samples: list[dict[str, Any]]) -> VoteResult | None:
    """Confidence-weighted (CISC) majority vote over action samples.

    Each sample is a dict with at least {verb, element_id, text, confidence}.
    Returns the winning action plus agreement/mean-confidence diagnostics.
    """
    if not samples:
        return None

    weights: dict[str, float] = {}
    confs: dict[str, list[float]] = {}
    rep: dict[str, dict] = {}
    total = 0.0

    for s in samples:
        key = canonical_action_key(s.get("verb"), s.get("element_id"), s.get("text"))
        c = float(s.get("confidence", DEFAULT_CONF) or DEFAULT_CONF)
        c = max(0.0, min(1.0, c))
        weights[key] = weights.get(key, 0.0) + c
        confs.setdefault(key, []).append(c)
        rep.setdefault(key, s)
        total += c

    if total <= 0:
        total = 1.0

    winner = max(weights, key=lambda k: weights[k])
    w = rep[winner]
    return VoteResult(
        verb=w.get("verb", ""),
        element_id=w.get("element_id"),
        text=w.get("text"),
        agreement=weights[winner] / total,
        mean_conf=sum(confs[winner]) / len(confs[winner]),
        n_votes=len(samples),
        n_unique=len(weights),
        tally={k: round(v, 2) for k, v in weights.items()},
        winner_sample=w,
    )


def should_abstain(vote: VoteResult | None,
                   theta_agree: float = THETA_AGREE,
                   theta_conf: float = THETA_CONF) -> bool:
    """Abstain (do NOT fire the irreversible action) when the voters disagree
    or are jointly unconfident."""
    if vote is None:
        return True
    return vote.agreement < theta_agree or vote.mean_conf < theta_conf


def _logical_model(name: str) -> str:
    """Normalize an instance name to its LOGICAL model id, so the same model on
    different providers counts ONCE (e.g. 'openai/gpt-oss-120b' and the Cerebras
    'gpt-oss-120b' → 'gpt-oss-120b'; 'google/gemma-4-31b-it' and the Gemini
    'gemma-4-31b-it' → 'gemma-4-31b-it'). This keeps the cascade voters genuinely
    independent rather than the same model polled twice."""
    from model_registry import ProviderHealthTracker, normalize_model_id
    return normalize_model_id(ProviderHealthTracker._base_model_name(name))


def distinct_base_model_clients(chain: list, n: int = CONSENSUS_N,
                                exclude_base: str = "") -> list:
    """Up to `n` chain entries with DISTINCT logical models (independent voters)."""
    seen: set[str] = set()
    if exclude_base:
        seen.add(normalize_model_id_safe(exclude_base))
    out: list = []
    for mc in chain:
        base = _logical_model(getattr(mc, "name", str(mc)))
        if base in seen:
            continue
        seen.add(base)
        out.append(mc)
        if len(out) >= n:
            break
    return out


def normalize_model_id_safe(s: str) -> str:
    """normalize_model_id but tolerant of full instance names."""
    return _logical_model(s) if (":" in s) else _logical_model("x:" + s + ":0")


def count_distinct_base_models(chain: list) -> int:
    return len({_logical_model(getattr(mc, "name", str(mc))) for mc in chain})


async def sample_ensemble(models: list, messages: list, schema,
                          invoke_fn, breaker, health_tracker) -> list[tuple]:
    """Poll each distinct model ONCE, in parallel. Returns [(decision, model_name)].

    Reuses the existing failover function with a 1-element chain per model, so a
    transient failure on one voter simply drops that vote (graceful)."""
    async def _one(mc):
        try:
            return await invoke_fn(
                [mc], messages, schema, breaker, health_tracker=health_tracker,
            )
        except Exception as e:  # noqa: BLE001 — a failed voter just abstains
            logger.debug("ensemble voter failed: %s", e)
            return None

    results = await asyncio.gather(*[_one(m) for m in models])
    return [r for r in results if r is not None]


# ═══════════════════════════════════════════════════════════════════════════════
#  Dynamic cascade consensus — need-based, latency-aware (the time/accuracy lever)
# ═══════════════════════════════════════════════════════════════════════════════
#  Instead of always polling N voters, escalate ONLY when needed:
#    Tier 1  primary confident + structurally sound + not stuck → execute (0 extra calls)
#    Tier 2  else poll the SECONDARY; if it AGREES → execute              (1 extra call)
#    Tier 3  else poll the TERTIARY → confidence-weighted (CISC) vote     (2 extra calls)
#    Abstain vote still split/unconfident → caller defers (vision / safe-wait)
#  High-confidence steps pay ZERO extra latency; only genuine ambiguity escalates.
#  Self-contained: the caller passes the model primitives and the already-built
#  `messages` — this layer NEVER builds prompts or touches brain state.


def _base_name(model_name: str) -> str:
    """The primary's LOGICAL model id (for excluding it from the voter pool)."""
    return _logical_model(model_name)


def _decision_sample(d) -> dict:
    """Extract the vote fields from a schema decision (generic attribute access)."""
    return {
        "verb": getattr(d, "action_type", "") or "",
        "element_id": getattr(d, "element_id", None),
        "text": getattr(d, "text", None),
        "confidence": float(getattr(d, "confidence", DEFAULT_CONF) or DEFAULT_CONF),
    }


def _decision_key(d) -> str:
    return canonical_action_key(getattr(d, "action_type", ""),
                                getattr(d, "element_id", None),
                                getattr(d, "text", None))


def structural_ok(decision, selector_map: dict | None = None) -> bool:
    """Is the primary's action WELL-FORMED? (the 'structural integrity' gate)

    A confident-but-malformed action (click with no/unknown element, goto with no
    URL) must NOT short-circuit the cascade — it escalates for a second opinion.
    """
    verb = (getattr(decision, "action_type", "") or "").lower()
    if verb in ("click", "type"):
        eid = getattr(decision, "element_id", None)
        if eid:
            return (not selector_map) or (eid in selector_map)
        # no element_id → only well-formed if explicit coordinates are present
        return (getattr(decision, "x", None) is not None
                and getattr(decision, "y", None) is not None)
    if verb == "goto":
        return bool(getattr(decision, "url", None))
    if verb == "ask_user":
        return bool(getattr(decision, "missing_data", "")
                    or getattr(decision, "rationale", ""))
    # done / wait / scroll / press_enter need no target → structurally fine
    return bool(verb)


@dataclass
class CascadeResult:
    """Outcome of the dynamic cascade."""
    decision: Any                      # the chosen schema decision to act on
    path: str = ""                     # primary_confident|primary_solo|secondary_agree|voted|abstain
    extra_calls: int = 0               # LLM calls beyond the primary (0,1,2)
    agreement: float = 1.0
    abstain: bool = False
    detail: str = ""


async def _poll_one(mc, messages, schema, invoke_fn, breaker, health_tracker):
    try:
        decision, _model = await invoke_fn(
            [mc], messages, schema, breaker, health_tracker=health_tracker)
        return decision
    except Exception as e:  # noqa: BLE001
        logger.debug("cascade voter failed: %s", e)
        return None


async def cascade_consensus(*, primary_decision, primary_model, messages, schema,
                            invoke_fn, chain, breaker, health_tracker,
                            selector_map: dict | None = None,
                            force_escalate: bool = False,
                            high_conf: float = HIGH_CONF) -> CascadeResult:
    """Resolve an IRREVERSIBLE action with the minimum voters necessary.

    `force_escalate` lets the caller demand a second opinion even on a confident
    primary (e.g. the agent is stuck / repeating). Returns a CascadeResult; the
    caller acts on `.decision` unless `.abstain` is set.
    """
    p_conf = float(getattr(primary_decision, "confidence", DEFAULT_CONF) or DEFAULT_CONF)
    p_struct = structural_ok(primary_decision, selector_map)

    # ── Tier 1: primary confident + sound + not stuck → execute immediately. ──
    if p_conf >= high_conf and p_struct and not force_escalate:
        return CascadeResult(primary_decision, "primary_confident", 0, 1.0, False,
                             f"conf={p_conf:.2f}≥{high_conf}, struct=ok")

    why = ("stuck/hesitation" if force_escalate
           else ("malformed" if not p_struct else f"low_conf({p_conf:.2f})"))

    voters = distinct_base_model_clients(chain, n=2, exclude_base=_base_name(primary_model))
    if not voters:
        # Single-model (e.g. premium) — no diverse second opinion available.
        return CascadeResult(primary_decision, "primary_solo", 0, 1.0, False,
                             f"{why}; no distinct voter")

    samples = [_decision_sample(primary_decision)]
    by_key = {_decision_key(primary_decision): primary_decision}

    # ── Tier 2: poll the SECONDARY only. If it agrees with primary → done. ──
    sec = await _poll_one(voters[0], messages, schema, invoke_fn, breaker, health_tracker)
    extra = 1
    if sec is not None:
        samples.append(_decision_sample(sec))
        by_key.setdefault(_decision_key(sec), sec)
        if _decision_key(sec) == _decision_key(primary_decision):
            return CascadeResult(primary_decision, "secondary_agree", extra, 1.0, False,
                                 f"{why}; secondary confirmed")

    # ── Tier 3: still ambiguous → poll the TERTIARY, then CISC-vote all. ──
    if len(voters) > 1:
        ter = await _poll_one(voters[1], messages, schema, invoke_fn, breaker, health_tracker)
        extra = 2
        if ter is not None:
            samples.append(_decision_sample(ter))
            by_key.setdefault(_decision_key(ter), ter)

    vote = weighted_vote(samples)
    if should_abstain(vote):
        return CascadeResult(primary_decision, "abstain", extra,
                             vote.agreement if vote else 0.0, True,
                             f"{why}; split {vote.tally if vote else {}}")

    win = by_key.get(canonical_action_key(vote.verb, vote.element_id, vote.text),
                     primary_decision)
    return CascadeResult(win, "voted", extra, vote.agreement, False,
                         f"{why}; winner agreement={vote.agreement:.2f}")
