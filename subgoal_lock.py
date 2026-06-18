"""Sub-Goal Lock — anti-amnesia for multi-part objectives (V29).

THE AMNESIA LOOP
════════════════
On "Do X and Do Y", the agent finishes Y, prematurely says 'done', and the Outcome
Judge (Overwatch L4) rejects globally ("missing X"). The OLD rejection was a binary,
"missing-only" message — it never re-affirmed that Y was DONE. So the agent, seeing
Y's still-visible button + a "you failed" correction, re-executed Y. Forever.

THE FIX — make the EXISTING sticky PRM ledger the active authority
═════════════════════════════════════════════════════════════════
A verified PRM checklist item is already LOCKED/immutable (V26). This module turns
that latent memory into behavior:
  • compose_rejection(): the rejection now says "✅ Y done & LOCKED — do NOT repeat;
    ❗ remaining: X" instead of a global "False" (Partial-Success, from the ONE
    existing checklist — no competing decomposition, so no V26 regression).
  • render_lock_list(): a FORBID list ("already done — never repeat") for the worker
    prompt. It forbids, it does not add pending focus → complementary to plan_steps.
  • reconcile_plan_with_ledger(): aligns plan_steps to the ledger (locked→done).
  • targets_locked_subgoal(): deterministic backstop — detect a re-do of a locked
    sub-goal so it can be stopped.

Pure logic; reuses the Target-Lock tokenizer (no duplication). Unit-testable.
"""

from __future__ import annotations

# Verbs that actually (re-)execute a sub-goal — the ones worth guarding.
_STATE_CHANGING = ("click", "type", "select_option", "press_enter")


def locked_subgoals(prm_checklist) -> list[dict]:
    """Verified (sticky-locked) sub-goals — completed and immutable."""
    return [d for d in (prm_checklist or []) if d.get("verified")]


def remaining_subgoals(prm_checklist) -> list[dict]:
    """Sub-goals not yet done/verified — the real remaining work."""
    out = []
    for d in (prm_checklist or []):
        if d.get("verified") or (d.get("status") or "") == "done":
            continue
        out.append(d)
    return out


def render_lock_list(prm_checklist, maxn: int = 6) -> str:
    """A compact 'already done — never repeat' block for the worker prompt.

    This FORBIDS (it never adds a pending sub-goal), so it is complementary to
    plan_steps and cannot re-introduce the V26 competing-checklist regression."""
    locked = locked_subgoals(prm_checklist)
    if not locked:
        return ""
    lines = ["═══ ✅ ALREADY DONE — LOCKED (never repeat these) ═══"]
    for d in locked[:maxn]:
        desc = (d.get("desc") or "").strip()[:80]
        ev = (d.get("evidence") or "").strip()[:60]
        lines.append(f"• {desc}" + (f" — {ev}" if ev else ""))
    lines.append("These sub-goals are COMPLETE. Do NOT click/redo their controls even if "
                 "the button is still visible — focus ONLY on what remains.")
    return "\n".join(lines)


def compose_rejection(missing: str, next_hint: str, prm_checklist) -> str:
    """Sub-goal-aware 'done' rejection: re-affirm locked-done work + name the
    remaining work, instead of a global binary 'False' that erases progress."""
    locked = locked_subgoals(prm_checklist)
    remaining = remaining_subgoals(prm_checklist)
    parts = ["\n\n🛡️ NOT DONE YET — but your completed work is SAFE and LOCKED."]
    if locked:
        done_str = "; ".join((d.get("desc") or "")[:50] for d in locked[:5])
        parts.append(f"✅ DONE & LOCKED (do NOT repeat): {done_str}")
    if remaining:
        rem_str = "; ".join((d.get("desc") or "")[:60] for d in remaining[:3])
        parts.append(f"❗ STILL REMAINING — do ONLY this now: {rem_str}")
    elif missing:
        parts.append(f"❗ STILL MISSING: {missing[:140]}")
    if next_hint:
        parts.append(f"DO THIS NOW: {next_hint[:140]}")
    parts.append("Only output 'done' once the REMAINING work shows proof on the page.")
    return "\n".join(parts)


def _tok(s: str) -> set[str]:
    from target_lock import _tokens
    return _tokens(s or "")


def reconcile_plan_with_ledger(plan_steps, prm_checklist):
    """Align plan_steps to the ledger: mark a not-done step whose description
    majority-overlaps a LOCKED sub-goal as 'done', then ensure one active step.
    Returns the updated list, or None if nothing changed (conservative)."""
    locked = locked_subgoals(prm_checklist)
    if not locked or not plan_steps:
        return None
    locked_tok: set[str] = set()
    for d in locked:
        locked_tok |= _tok(d.get("desc", ""))
    if not locked_tok:
        return None
    steps = [dict(s) for s in plan_steps]
    changed = False
    for s in steps:
        if s.get("status") == "done":
            continue
        st = _tok(s.get("desc", ""))
        if st and len(st & locked_tok) >= max(1, len(st) // 2):  # majority overlap
            s["status"] = "done"
            changed = True
    if not changed:
        return None
    if not any(s.get("status") in ("active", "in_progress") for s in steps):
        for s in steps:
            if s.get("status") == "pending":
                s["status"] = "active"
                break
    return steps


def targets_locked_subgoal(proposed: dict, prm_checklist):
    """Deterministic backstop: is the proposed action RE-DOING a control that
    belongs to an already-LOCKED sub-goal? Returns the locked item, or None.

    Conservative: requires a state-changing verb AND ≥2 shared identity tokens
    between the action's target and a locked sub-goal's description — so a distinct
    REMAINING action (which matches a different sub-goal) is never falsely blocked."""
    locked = locked_subgoals(prm_checklist)
    if not locked:
        return None
    if (proposed.get("verb") or "").lower() not in _STATE_CHANGING:
        return None
    ctoks = _tok(f"{proposed.get('target_name', '')} {proposed.get('text', '') or ''}")
    if len(ctoks) < 2:
        return None
    for d in locked:
        if len(ctoks & _tok(d.get("desc", ""))) >= 2:
            return d
    return None
