"""Feature Flags — the V29 Cognitive Overhaul master switchboard.

Every V29 organ is ADDITIVE and behind a switch, so the new cognitive layer can
be toggled without touching (or risking) the proven V16–V28 pipeline. This is the
anti-regression contract (Mandate 6) made literal:

  • `V29_ENABLED=0`  → master kill-switch: the entire overhaul is inert and the
    agent behaves EXACTLY like V28. One env var reverts everything.
  • Per-feature flags let a single organ be disabled in isolation if a live test
    surfaces a problem, without losing the others.

Defaults are ON (the features are meant to be exercised in live testing), but a
flag is honored the instant it is set — `V29_REALITY=0` disables just the Reality
Monitor, etc. The env is read live (not cached) so a flag can be flipped between
runs without code changes.

Pattern mirrors `consensus.consensus_enabled()` (env-var truthiness) for
consistency with the existing codebase.
"""

from __future__ import annotations

import os

_OFF = ("0", "false", "no", "off")


def _flag(name: str, default: bool = True) -> bool:
    """Read a boolean env flag. Unset → `default`; any of 0/false/no/off → False."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _OFF


# ── Master switch ──────────────────────────────────────────────────────────────
def v29_enabled() -> bool:
    """Master kill-switch. `V29_ENABLED=0` reverts the agent to pure V28 behavior."""
    return _flag("V29_ENABLED", True)


# ── Phase 1 — Screen-Reality Reconciliation (Mandate 1) ─────────────────────────
def reality_enabled() -> bool:
    """The Reality Monitor: block 'blind execution' when the live screen
    contradicts the worker's predicted change."""
    return v29_enabled() and _flag("V29_REALITY", True)


def reality_llm_enabled() -> bool:
    """Deterministic-first: a cheap LLM reconcile call is made ONLY for genuinely
    ambiguous deltas (status==UNCLEAR). `V29_REALITY_LLM=0` keeps it purely
    deterministic."""
    return reality_enabled() and _flag("V29_REALITY_LLM", True)


# ── Phase 2 — Clarity-Triggered Consensus & Vision (Mandates 4, 5) ──────────────
def clarity_consensus_enabled() -> bool:
    """Broaden PRE-action consensus from IRREVERSIBLE-only to ANY low-clarity step
    (zero risk: if unsure, poll the ensemble before acting)."""
    return v29_enabled() and _flag("V29_CLARITY_CONSENSUS", True)


def target_lock_enabled() -> bool:
    """Strict goal-binding: bind the action to the target item's identity and
    resist identical-looking distractor controls (anti context-drift)."""
    return v29_enabled() and _flag("V29_TARGET_LOCK", True)


def intent_journal_enabled() -> bool:
    """Atomic Intent Journaling: write-ahead record of a side-effecting action
    BEFORE it runs, fed to the next decision so a timed-out/crashed action is never
    blindly repeated (handoff-amnesia / double-toggle fix)."""
    return v29_enabled() and _flag("V29_INTENT_JOURNAL", True)


def subgoal_lock_enabled() -> bool:
    """Sub-Goal Lock: a verified-complete sub-goal stays LOCKED even when the
    Outcome Judge globally rejects a premature 'done' — so the agent never re-does
    a finished sub-goal (the multi-part 'amnesia loop' fix)."""
    return v29_enabled() and _flag("V29_SUBGOAL_LOCK", True)


# ── Phase 3 — Progress-Aware Loops & Smart Scrolling (Mandate 3) ────────────────
def stagnation_enabled() -> bool:
    """Revive the dead `same_url_streak` into a generalized stagnation detector."""
    return v29_enabled() and _flag("V29_STAGNATION", True)


def smart_scroll_enabled() -> bool:
    """Scroll with feedback (delta / at-bottom / new-content) instead of blind 600px."""
    return v29_enabled() and _flag("V29_SMART_SCROLL", True)


# ── Phase 4 — Page-Subject Understanding (Mandate 2) ────────────────────────────
def page_context_enabled() -> bool:
    """Page archetype + instruction-aware DOM re-rank."""
    return v29_enabled() and _flag("V29_PAGE_CONTEXT", True)


# ── Adaptive Perception Engine (universal; P0 = Tier-1 passthrough router) ───────
def adaptive_perception_enabled() -> bool:
    """Route page perception through the Adaptive Perception Engine. P0 is a
    behavior-identical Tier-1 passthrough; later phases add deep-sweep / vision
    tiers behind the same flag. Off ⇒ the direct single-pass snapshot (V28 path)."""
    return v29_enabled() and _flag("V29_ADAPTIVE_PERCEPTION", True)


def strict_viewport_enabled() -> bool:
    """AP-P1: strict viewport filter on Tier-1 output — drop off-screen noise;
    preserve + tag off-screen actionables (universal recall). Gated under the
    perception engine; ON by default."""
    return adaptive_perception_enabled() and _flag("V29_STRICT_VIEWPORT", True)


# ── Phase A — Stabilizers (DOM diffing + hybrid primitives) ─────────────────────
def diffing_enabled() -> bool:
    """CriticV12 page-signal-vector diff: catch subtle overlay/modal/panel changes
    a node-count diff misses, and emit a unified state_change_score."""
    return v29_enabled() and _flag("V29_DIFFING", True)


def hybrid_primitives_enabled() -> bool:
    """Clean action feedback (asymmetric verbosity + FailureClass, no strategy-name
    leak) + expanded primitives (hover / select_option / press_key)."""
    return v29_enabled() and _flag("V29_HYBRID_PRIMITIVES", True)


# ── Phase B / C — Simulator + Autonomy (flags reserved; wired in later phases) ───
def webdreamer_enabled() -> bool:
    """Phase B: predictive top-K action simulation (LLM-imagined, no real actions),
    gated behind the Clarity Gate + a cost gate so it only fires on high-stakes
    ambiguous/irreversible steps."""
    return v29_enabled() and _flag("V29_WEBDREAMER", True)


def webdreamer_situational_enabled() -> bool:
    """Situational scoring for WebDreamer — reward engaging a just-revealed toggle,
    decay dead-end scroll, elevate goto when stuck. `=0` ⇒ instant fallback to the
    vacuum-scoring baseline. Gated under WebDreamer."""
    return webdreamer_enabled() and _flag("V29_WEBDREAMER_SITUATIONAL", True)


def lats_enabled() -> bool:
    """Phase C: tree-search/backtracking over checkpoints (default OFF until built)."""
    return v29_enabled() and _flag("V29_LATS", False)


def skill_memory_v2_enabled() -> bool:
    """Phase C: deepened retrieval-augmented procedural memory (default OFF until built)."""
    return v29_enabled() and _flag("V29_SKILL_MEMORY_V2", False)


def active_flags() -> dict[str, bool]:
    """Snapshot of every flag — logged once at startup for run-log auditability."""
    return {
        "V29_ENABLED": v29_enabled(),
        "V29_REALITY": reality_enabled(),
        "V29_REALITY_LLM": reality_llm_enabled(),
        "V29_CLARITY_CONSENSUS": clarity_consensus_enabled(),
        "V29_TARGET_LOCK": target_lock_enabled(),
        "V29_INTENT_JOURNAL": intent_journal_enabled(),
        "V29_SUBGOAL_LOCK": subgoal_lock_enabled(),
        "V29_STAGNATION": stagnation_enabled(),
        "V29_SMART_SCROLL": smart_scroll_enabled(),
        "V29_PAGE_CONTEXT": page_context_enabled(),
        "V29_ADAPTIVE_PERCEPTION": adaptive_perception_enabled(),
        "V29_STRICT_VIEWPORT": strict_viewport_enabled(),
        "V29_DIFFING": diffing_enabled(),
        "V29_HYBRID_PRIMITIVES": hybrid_primitives_enabled(),
        "V29_WEBDREAMER": webdreamer_enabled(),
        "V29_WEBDREAMER_SITUATIONAL": webdreamer_situational_enabled(),
        "V29_LATS": lats_enabled(),
        "V29_SKILL_MEMORY_V2": skill_memory_v2_enabled(),
    }
