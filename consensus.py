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
CONSENSUS_N = 3        # number of independent voters (distinct base models)
THETA_AGREE = 0.6      # min winning weight-share to ACT on an irreversible action
THETA_CONF = 0.5       # min mean self-confidence of the winning voters
DEFAULT_CONF = 0.7     # assumed confidence when a model omits the field


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


def distinct_base_model_clients(chain: list, n: int = CONSENSUS_N,
                                exclude_base: str = "") -> list:
    """Up to `n` chain entries with DISTINCT base models (the independent voters)."""
    from model_registry import ProviderHealthTracker
    seen: set[str] = set()
    if exclude_base:
        seen.add(exclude_base)
    out: list = []
    for mc in chain:
        base = ProviderHealthTracker._base_model_name(getattr(mc, "name", str(mc)))
        if base in seen:
            continue
        seen.add(base)
        out.append(mc)
        if len(out) >= n:
            break
    return out


def count_distinct_base_models(chain: list) -> int:
    from model_registry import ProviderHealthTracker
    return len({
        ProviderHealthTracker._base_model_name(getattr(mc, "name", str(mc)))
        for mc in chain
    })


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
