"""Action Feedback — clean, semantic execution feedback (V29 Phase A, Stagehand-style).

The 4-tier click waterfall is ALREADY internal (the LLM issues `click`, never
`cdp_click`/`js_click`). This module finishes the abstraction by cleaning up what
the LLM is TOLD about an action:

  • SUCCESS  → terse: only the observable EFFECT (navigated / DOM changed), never
               the internal strategy name. Keeps the reasoning context clean.
  • FAILURE  → rich but SEMANTIC: a small, universal `FailureClass` + one recourse
               hint the agent can act on — never a raw tier/stack trace.

This is asymmetric verbosity: quiet when it works, specific when it doesn't.
Universal — failure classes are execution semantics (timeout/blocked/no-effect),
not site rules. The raw error is preserved in-line so downstream detectors
(Reality Monitor / Intent Journal: "timed out", "crashed") keep working unchanged.
"""

from __future__ import annotations

from enum import Enum


class FailureClass(str, Enum):
    NOT_FOUND = "not_found"      # element gone / detached / unresolved
    OBSCURED = "obscured"        # covered by an overlay / intercepted
    NO_EFFECT = "no_effect"      # acted, but nothing changed (wrong/disabled target)
    BLOCKED = "blocked"          # navigation / domain / permission blocked
    TIMEOUT = "timeout"          # timed out — may have PARTIALLY applied
    INPUT_FAILED = "input_failed"  # text could not be entered
    UNKNOWN = "unknown"


# Universal recourse — what the agent should do next, by class (no site logic).
_RECOURSE: dict[FailureClass, str] = {
    FailureClass.NOT_FOUND: "the element is gone or changed — re-read the page and pick a fresh element.",
    FailureClass.OBSCURED: "a popup/overlay is likely covering the target — dismiss it first, then retry.",
    FailureClass.NO_EFFECT: "the element did not respond — it may be disabled or the wrong target; choose a different element.",
    FailureClass.BLOCKED: "this route is blocked — try a different path or URL.",
    FailureClass.TIMEOUT: "the action timed out and may have only PARTIALLY applied — re-read the live DOM before repeating.",
    FailureClass.INPUT_FAILED: "the text could not be entered — make sure the field is focused/editable, or pick the right input.",
    FailureClass.UNKNOWN: "the action did not succeed — re-evaluate the page and try a different approach.",
}


def classify_failure(verb: str, error: str) -> FailureClass:
    """Map a low-level execution error to a semantic failure class (universal)."""
    e = (error or "").lower()
    if any(t in e for t in ("timed out", "timeout", "crashed", "disconnect")):
        return FailureClass.TIMEOUT
    if any(t in e for t in ("blocked", "not allowed", "forbidden", "permission")):
        return FailureClass.BLOCKED
    if any(t in e for t in ("no node", "not found", "detached", "no live", "unresolved", "no element")):
        return FailureClass.NOT_FOUND
    if any(t in e for t in ("overlay", "covered", "intercept", "obscured")):
        return FailureClass.OBSCURED
    if (verb or "") == "type" and any(t in e for t in ("input", "editable", "focus", "field")):
        return FailureClass.INPUT_FAILED
    if any(t in e for t in ("ineffective", "no effect", "did not", "no-op", "not respond")):
        return FailureClass.NO_EFFECT
    return FailureClass.UNKNOWN


def render_failure(fc: FailureClass, error: str = "") -> str:
    """Semantic failure line for the LLM. Preserves the raw error in-line so
    substring-based detectors elsewhere keep working."""
    raw = f" ({error[:80]})" if error else ""
    return f"→ FAILED [{fc.value}]: {_RECOURSE[fc]}{raw}"


def render_success(effect_bits: list[str] | None = None) -> str:
    """Terse success line — ONLY the observable effect, never the internal strategy."""
    bits = [b for b in (effect_bits or []) if b]
    return "→ OK" + (f" ({', '.join(bits)})" if bits else "")
