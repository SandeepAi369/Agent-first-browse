"""Clarity Gate — uncertainty-triggered PRE-action consensus & vision (V29 / Phase 2).

THE ZERO-RISK RULE
══════════════════
Until now a second opinion (consensus) was reserved for IRREVERSIBLE actions only.
The mandate: if the agent has ANY doubt, hesitation, or low clarity about the
screen — or is tempted by an ambiguous/neighboring look-alike control — it must
poll the ensemble BEFORE executing, never act-then-check. "If unsure, vote first."

This module computes ONE deterministic `ClaritySignal` from signals already on the
state, and exposes two gates that share it:
  • `needs_consensus()` — broadens the cascade trigger to low-clarity OR irreversible.
  • `needs_vision()`     — opens the eyes on the SAME ambiguity (esp. look-alikes).

Cost stays bounded: the consensus cascade already short-circuits on a confident,
structurally-sound primary (Tier-1 = 0 extra calls). Broadening only changes WHEN
the cascade is entered — and, per the approved decision, we accept 1–2 extra calls
when genuinely uncertain (zero-risk over speed).

References: Reliability-Aware Adaptive Self-Consistency (arXiv 2601.02970) and
CoRefine (arXiv 2602.08948) — confidence as the control signal for spending
compute; "Are GUI Agents Focused Enough?" (arXiv 2604.07831) — distractor focus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Below this self-reported confidence the action is "not clear enough to fire blind".
THETA_LOW_CONF = 0.6


@dataclass
class ClaritySignal:
    low_clarity: bool = False
    target_ambiguity: bool = False     # ≥2 identical primary-action controls
    off_target: bool = False           # about to act on a non-target look-alike
    score: float = 1.0                 # rough 0..1 clarity estimate (higher = clearer)
    reasons: list[str] = field(default_factory=list)

    @property
    def uncertain(self) -> bool:
        return self.low_clarity or self.target_ambiguity or self.off_target


def compute_clarity(decision: dict, state: dict, *,
                    target=None, selector_map: dict | None = None) -> ClaritySignal:
    """Build the clarity signal for the proposed action.

    `decision` is the worker's structured output as a dict (verb/confidence/
    needs_vision/element_id/target_name/text). `target` is the TargetDescriptor
    from target_lock.extract_target (optional). Pure + deterministic.
    """
    selector_map = selector_map or state.get("selector_map", {}) or {}
    reasons: list[str] = []
    low = False

    conf = float(decision.get("confidence", decision.get("conf", 0.7)) or 0.7)
    if conf < THETA_LOW_CONF:
        low = True
        reasons.append(f"low confidence ({conf:.2f})")

    if decision.get("needs_vision"):
        low = True
        reasons.append("worker flagged needs_vision")

    # The Reality Monitor said the screen contradicted the last action → high doubt.
    if state.get("reality_status") == "CONTRADICTED" or (state.get("reality_note") or "").strip():
        low = True
        reasons.append("last action was contradicted by the screen")

    # Hesitation / churn.
    if int(state.get("consecutive_identical_actions", 0) or 0) >= 2:
        low = True
        reasons.append("repeating the same action")
    if (state.get("correction_context") or "").strip():
        low = True
        reasons.append("under an active escalation directive")

    # Chronic no-progress (Phase 3 stagnation) also lowers clarity.
    if (state.get("stagnation_note") or "").strip():
        low = True
        reasons.append("stagnation detected (no real progress)")

    # ── Target ambiguity & off-target urge (the Context-Drift guard) ──
    target_ambiguity = False
    off_target = False
    verb = (decision.get("verb") or decision.get("action_type") or "").lower()
    label = f"{decision.get('target_name','')} {decision.get('text','') or ''}".strip()
    try:
        from target_lock import (is_primary_action_click, count_lookalikes,
                                 off_target_risk)
        if is_primary_action_click(verb, decision.get("target_name", ""),
                                   decision.get("text", "") or ""):
            n = count_lookalikes(selector_map, label)
            if n >= 2:
                target_ambiguity = True
                reasons.append(f"{n} identical primary-action controls on page")
            if target is not None:
                chosen_ctx = label
                eid = decision.get("element_id")
                if eid and eid in selector_map and isinstance(selector_map[eid], dict):
                    el = selector_map[eid]
                    chosen_ctx = f"{el.get('name','')} {el.get('text','')} {label}"
                if off_target_risk(target, chosen_ctx, selector_map, label):
                    off_target = True
                    low = True
                    reasons.append("about to act on a NON-target look-alike item")
    except Exception:
        pass

    score = 1.0
    score -= 0.4 * (1 if low else 0)
    score -= 0.3 * (1 if target_ambiguity else 0)
    score -= 0.3 * (1 if off_target else 0)
    score = max(0.0, min(1.0, score if (low or target_ambiguity or off_target) else conf))

    return ClaritySignal(low_clarity=low, target_ambiguity=target_ambiguity,
                         off_target=off_target, score=score, reasons=reasons)


def needs_consensus(signal: ClaritySignal, *, is_irreversible: bool,
                    broaden: bool) -> tuple[bool, str]:
    """Should this action be put to a PRE-action ensemble vote?

    IRREVERSIBLE always votes (unchanged contract). When `broaden` (the Clarity
    Gate flag) is on, ANY uncertainty also votes — including target ambiguity and
    the urge to click a non-target look-alike, exactly as mandated.
    """
    if is_irreversible:
        return True, "irreversible action"
    if not broaden:
        return False, ""
    if signal.uncertain:
        return True, "; ".join(signal.reasons[:3]) or "low clarity"
    return False, ""


def needs_vision_for_clarity(signal: ClaritySignal) -> tuple[bool, str]:
    """Open the eyes on visual ambiguity — most importantly to disambiguate WHICH
    of several identical controls belongs to the bound target."""
    if signal.off_target or signal.target_ambiguity:
        return True, "visually disambiguate the target among look-alikes"
    return False, ""
