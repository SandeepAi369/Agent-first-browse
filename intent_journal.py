"""Atomic Intent Journal — write-ahead logging for side-effecting actions (V29).

THE HANDOFF-AMNESIA / DOUBLE-TOGGLE CURE
════════════════════════════════════════
The ONLY place a browser side-effect happens is Overwatch's `_execute_action`
(CDP click / type / press_enter). `mcp_click` wraps the click in a 30s timeout; on
timeout it returns success=False ("Click timed out"). But the CDP click may have
ALREADY dispatched and taken effect — the verification just didn't complete in
time. Recorded as "ineffective", the next worker decision re-issues the same click
→ a double-toggle (un-stars, double add-to-cart, double vote): unrecoverable for an
IRREVERSIBLE action.

(NOTE: the LLM `_invoke_with_failover` is the DECISION layer — Model A→B failover
happens while CHOOSING an action; no page action runs there. So the journal lives
at the real execute point, and is fed to the whole chain via the prompt so any
failover model sees it.)

THE FIX — write-ahead intent journaling (a classic WAL / idempotency pattern):
  1. BEFORE executing, record an Intent Payload {ts, verb, element_id, target,
     risk, signature, status} — both into BrainState (survives the node's atomic
     super-step commit) AND into a durable, atomically-written file (survives a
     hard mid-execution crash).
  2. The action runs. On a CONFIRMED success the journal is resolved/cleared. On
     any non-confirmed outcome (timeout/crash/ineffective) the journal PERSISTS.
  3. The NEXT decision is fed the pending journal with a strict HESITATION rule:
     "your predecessor attempted X and disconnected — do NOT blindly repeat it;
     assume it may have partially succeeded; re-read the DOM first." Every model in
     the failover chain sees this (it is part of the prompt).
  4. If the worker proposes the SAME action again while the journal is pending, the
     Clarity Gate forces a PRE-action consensus instead of firing blind.

Pure helpers + a tiny durable ledger. All durable I/O is best-effort (never raises
into the step). Unit-testable without a browser or LLM.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("intent_journal")

# Verbs that mutate page/server state and can double-apply if repeated.
JOURNALED_VERBS = ("click", "type", "press_enter", "select_option")

# Outcome substrings that mean "we genuinely don't know if it took effect".
_UNCERTAIN_MARKERS = ("timed out", "timeout", "crashed", "disconnect")

# State-FLIPPING controls. Repeating one of these blind is the WORST double-apply
# case — it doesn't duplicate, it inverts (checkbox flips back off, star un-stars,
# follow→unfollow). Crucially these are usually classified REVERSIBLE, so the
# hazard is invisible to the risk label and must be detected on its own.
_TOGGLE_KINDS = ("checkbox", "switch", "radio", "menuitemcheckbox", "menuitemradio")
_TOGGLE_HINTS = (
    "toggle", "check box", "checkbox", "switch", "on/off", "turn on", "turn off",
    "enable", "disable", "check", "uncheck", "tick", "untick", "select all",
    "opt in", "opt out", "subscribe", "unsubscribe", "follow", "unfollow",
    "like", "unlike", "star", "unstar", "mute", "unmute", "save", "unsave",
    "remember me", "show password", "hide password",
)


def action_signature(verb: str | None, element_id: str | None, text: str | None) -> str:
    """Stable identity of an action (matches the consensus key shape)."""
    return f"{(verb or '').strip().lower()}|{(element_id or '').strip()}|{(text or '').strip()[:40]}"


def should_journal(verb: str | None) -> bool:
    return (verb or "").strip().lower() in JOURNALED_VERBS


def is_toggle_like(target_name: str = "", text: str = "", element_kind: str = "") -> bool:
    """NON-EXHAUSTIVE HINT only — true if this LOOKS like a binary state-flip control.

    Universality note: the real universal signal is structural (role=switch/checkbox/
    radio or an aria-checked/pressed/selected state) — language-agnostic. Those are
    used when available; the keyword list is just an extra hint for the common cases.
    This is NEVER the source of truth: the journal protects EVERY side-effecting
    action regardless of this, and the worker is told to reason from the live DOM.
    """
    if (element_kind or "").strip().lower() in _TOGGLE_KINDS:
        return True
    blob = f"{target_name} {text}".lower()
    return any(h in blob for h in _TOGGLE_HINTS)


def hazard_class(verb: str, target_name: str = "", text: str = "",
                 risk_level: str = "REVERSIBLE", element_kind: str = "") -> str:
    """A SOFT HINT about the double-apply hazard — 'toggle' | 'irreversible' |
    'state-change'. It only tunes the *flavour* of the hesitation note; the actual
    do-not-repeat protection is universal and does not depend on this label. Returns
    'state-change' (the safe generic) whenever nothing more specific is detected, so
    a novel/foreign-language control is still covered by the universal rule."""
    if is_toggle_like(target_name, text, element_kind):
        return "toggle"
    if (risk_level or "").upper() == "IRREVERSIBLE":
        return "irreversible"
    return "state-change"


def make_intent(proposed: dict, step_number: int, element_kind: str = "") -> dict:
    """Build the Intent Payload written BEFORE execution."""
    now = time.time()
    verb = proposed.get("verb", "")
    target_name = (proposed.get("target_name", "") or "")[:60]
    text = (proposed.get("text", "") or "")[:40]
    risk_level = proposed.get("risk_level", "REVERSIBLE")
    return {
        "ts": round(now, 3),
        "iso": time.strftime("%H:%M:%S", time.localtime(now)),
        "step": int(step_number) + 1,
        "verb": verb,
        "element_id": proposed.get("element_id"),
        "target_name": target_name,
        "text": text,
        "risk_level": risk_level,
        "hazard": hazard_class(verb, target_name, text, risk_level, element_kind),
        "signature": action_signature(verb, proposed.get("element_id"),
                                      proposed.get("text")),
        "status": "executing",
    }


def is_uncertain_outcome(outcome: str) -> bool:
    """True when the execution outcome leaves it genuinely unknown whether the
    action took effect (timeout / crash) — the dangerous double-apply case."""
    o = (outcome or "").lower()
    return any(m in o for m in _UNCERTAIN_MARKERS)


def classify_status(outcome: str) -> str:
    """confirmed (clean OK) | uncertain (timeout/crash) | executed (ran, no clear OK)."""
    if "ok" in (outcome or "").lower():
        return "confirmed"
    if is_uncertain_outcome(outcome):
        return "uncertain"
    return "executed"


def same_action(entry: dict | None, proposed: dict | None) -> bool:
    """Is the proposed action the SAME one already journaled (and unconfirmed)?"""
    if not entry or not proposed:
        return False
    return entry.get("signature") == action_signature(
        proposed.get("verb"), proposed.get("element_id"), proposed.get("text"))


def render_hesitation(entry: dict | None) -> str:
    """The strict handoff-awareness block injected into the worker prompt (and thus
    seen by every model in the failover chain).

    UNIVERSAL BY DESIGN. It does not branch on a fixed taxonomy of control types —
    it gives ONE situational rule and tells the model to reason about whatever THIS
    specific control actually is, from the live DOM. The toggle/checkbox 'flip-back'
    case is mentioned only as ONE example of that reasoning, never as the scope. So
    it adapts to steppers, multi-state controls, expand/collapse, quantity inputs,
    foreign-language UIs, and anything not yet seen. A detected `hazard` (itself only
    a heuristic) is appended as a soft, explicitly-non-authoritative hint.
    """
    if not entry:
        return ""
    verb = entry.get("verb", "?")
    tgt = entry.get("target_name") or entry.get("element_id") or "an element"
    hazard = entry.get("hazard")

    # Soft, non-authoritative hint — the model must still verify for itself.
    hint = ""
    if hazard == "toggle":
        hint = (" (Heuristic hint — verify yourself: this may be a state-FLIPPING "
                "control, where repeating would INVERT the state rather than retry.)")
    elif hazard == "irreversible":
        hint = (" (Heuristic hint — verify yourself: this may be a COMMITTING action, "
                "where repeating would DUPLICATE the effect.)")

    return (
        "═══ ⚠️ PENDING-ACTION LEDGER (handoff awareness) ═══\n"
        f"Your previous attempt — {verb} on '{tgt}' (id {entry.get('element_id') or 'n/a'}) "
        f"at {entry.get('iso', '?')} — did NOT return a confirmed success; it may have "
        f"TIMED OUT or only PARTIALLY applied.{hint}\n"
        "UNIVERSAL RULE — reason about THIS exact situation; do NOT assume and do NOT "
        "blindly repeat:\n"
        "1) From the CURRENT live DOM, determine whether the intended effect is ALREADY "
        "present (value already set, state already changed, item already there, page "
        "already navigated, text already entered, post already made).\n"
        "2) Reason about what repeating THIS specific control would do in THIS context — "
        "would it duplicate the effect, invert/undo a state, advance a counter, or "
        "re-submit? Adapt to whatever this control actually is.\n"
        "3) Repeat the action ONLY if the live DOM proves it did NOT take effect. If it "
        "already did, or you cannot tell, do NOT repeat — verify the outcome or move on."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Durable atomic ledger (write-ahead file) — best-effort, never raises
# ═══════════════════════════════════════════════════════════════════════════════

def default_ledger_path() -> str:
    return str(Path(__file__).parent / "persistence" / "intent_journal.json")


def persist_intent(entry: dict, path: str | None = None) -> None:
    """Atomically write the intent BEFORE the action runs (tmp + os.replace)."""
    path = path or default_ledger_path()
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entry, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    except Exception as e:  # noqa: BLE001 — ledger I/O must never break the step
        logger.debug("persist_intent non-fatal: %s", e)


def resolve_intent(path: str | None = None) -> None:
    """Clear the durable ledger once the action is confirmed/handled."""
    path = path or default_ledger_path()
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:  # noqa: BLE001
        logger.debug("resolve_intent non-fatal: %s", e)


def read_pending(path: str | None = None) -> dict | None:
    """Read a dangling write-ahead entry (present ⇒ a prior execution never resolved)."""
    path = path or default_ledger_path()
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.debug("read_pending non-fatal: %s", e)
    return None
