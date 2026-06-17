"""Target Lock — strict goal-binding + distraction resistance (V29 / Mandate: Contextual Focus).

THE CONTEXT-DRIFT CURE
══════════════════════
Dense pages repeat identical-looking controls — a row of 'Add to cart' / 'Buy' /
'Select' buttons, one per item. When the agent's intended action fails and it gets
confused, its focus drifts and it clicks the SAME-looking control for a NEIGHBORING
item. That silently abandons the goal (you asked for Item A; it added Item B).

Research calls this "UI element misclick" from layout crowding / visual ambiguity,
and the robust defense is **semantic target binding** — bind the action to the
item's identity, not the (identical) button label (arXiv 2604.07831 "Are GUI Agents
Focused Enough?"; OSCAR dual-grounding). This module provides:

  • `extract_target()`    — the semantic identity of the CURRENT sub-task's target.
  • `count_lookalikes()`  — how many identical primary-action controls exist (the
                            ambiguity that should trigger the Clarity Gate).
  • `off_target_risk()`   — best-effort: is the chosen control for a DIFFERENT item
                            than the bound target? (fires when item context is visible).
  • `render_target_lock_block()` — the persistent prompt block encoding the explicit
                            TEMPORAL self-question the user mandated.

Pure logic, universal (no hardcoded sites/items), side-effect-free, unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Primary, item-scoped action controls. Repeated instances of these are the
# distractors. Broader than the irreversible set on purpose (select/choose/add
# are item-scoped even when reversible).
PRIMARY_ACTION_PATTERNS: tuple[str, ...] = (
    "add to cart", "add to bag", "add to basket", "add to wishlist", "add to list",
    "buy now", "buy at", "buy", "place order", "checkout", "check out", "order now",
    "select", "choose", "pick", "reserve", "book now", "subscribe", "get it",
    "add", "remove",
)

# Tokens that carry no target identity (verbs / chrome / filler) — stripped so the
# residual tokens are the item's distinguishing words.
_NOISE = set(
    "the a an of to and or in on at is are be for from your you this that with into "
    "onto then now click type add cart bag basket buy select choose press enter open "
    "go goto navigate visit find search scroll wait done page button link item items "
    "product products result results please click on its the first relevant verify "
    "mark complete task immediately do not after using what see screen actual real "
    "once it will should i.e e.g etc".split()
)


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", (s or "").lower()) if w not in _NOISE}


@dataclass
class TargetDescriptor:
    """The semantic identity the current step must act on."""
    phrase: str = ""           # short human-readable focus (the sub-task)
    tokens: set[str] = field(default_factory=set)  # distinguishing identity tokens

    @property
    def specified(self) -> bool:
        return bool(self.tokens)


def extract_target(objective: str, active_subtask: str = "") -> TargetDescriptor:
    """Derive the bound target from the current sub-task + objective.

    The sub-task is the primary source of WHAT to act on now; the objective
    supplies the distinguishing item identity (often a quoted product name). We
    union them so a generic sub-task ('Add to cart') still inherits the specific
    item identity from the objective.
    """
    quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", objective or "")
    toks = _tokens(active_subtask)
    for q in quoted:
        toks |= _tokens(q)
    toks |= _tokens(objective)
    phrase = (active_subtask or objective or "").strip()[:140]
    return TargetDescriptor(phrase=phrase, tokens=toks)


def _matched_primary_pattern(text: str) -> str | None:
    t = (text or "").lower()
    for p in PRIMARY_ACTION_PATTERNS:
        if p in t:
            return p
    return None


def is_primary_action_click(verb: str, target_name: str = "", target_text: str = "") -> bool:
    """True if this is a click on an item-scoped primary action (the distractor class)."""
    if (verb or "").lower() != "click":
        return False
    return bool(_matched_primary_pattern(f"{target_name} {target_text}"))


def _label_of(el: dict) -> str:
    return f"{el.get('name','')} {el.get('text','')}".strip()


def count_lookalikes(selector_map: dict, chosen_label: str) -> int:
    """How many interactive elements carry the SAME primary-action label as the
    chosen one (i.e., the count of identical-looking controls on the page)."""
    pat = _matched_primary_pattern(chosen_label)
    if not pat:
        return 0
    n = 0
    for el in (selector_map or {}).values():
        if not isinstance(el, dict):
            continue
        if _matched_primary_pattern(_label_of(el)) == pat:
            n += 1
    return n


def target_match(target: TargetDescriptor, context_text: str) -> float:
    """Fraction of the bound-target identity tokens present in a control's context.
    1.0 when no specific target is bound (no constraint to enforce)."""
    if not target.specified:
        return 1.0
    ctoks = _tokens(context_text)
    if not ctoks:
        return 0.0
    return len(target.tokens & ctoks) / max(1, len(target.tokens))


def off_target_risk(target: TargetDescriptor, chosen_context: str,
                    selector_map: dict, chosen_label: str) -> bool:
    """Best-effort: are we about to act on a look-alike control that belongs to a
    DIFFERENT item than the bound target? Fires only when item context is visible
    in the element labels (otherwise the prompt-level Target Lock carries it)."""
    if not target.specified:
        return False
    pat = _matched_primary_pattern(chosen_label)
    if not pat:
        return False
    # The chosen control does NOT match the bound target …
    if target_match(target, chosen_context) > 0:
        return False
    # … but some OTHER identical control DOES → we're about to pick the wrong item.
    for el in (selector_map or {}).values():
        if not isinstance(el, dict):
            continue
        lbl = _label_of(el)
        if _matched_primary_pattern(lbl) == pat and target_match(target, lbl) > 0:
            return True
    return False


def render_target_lock_block(target: TargetDescriptor, objective: str) -> str:
    """Persistent prompt block: strict goal-binding + the explicit TEMPORAL
    self-question. Universal — describes the failure pattern, not any specific site."""
    if not target.phrase:
        return ""
    focus = target.phrase
    ident = ", ".join(sorted(target.tokens)[:8]) if target.specified else "(see the sub-task)"
    return (
        "═══ 🎯 TARGET LOCK ═══\n"
        f"This step must concern ONLY your bound target: {focus}\n"
        f"Identifying words for your target: {ident}\n"
        "Many pages repeat identical-looking controls (e.g., one 'Add to cart' / "
        "'Buy' / 'Select' per item). Every such control that is NOT your target item "
        "is a DISTRACTOR. Before clicking one, confirm it belongs to YOUR target by "
        "its surrounding context (the product name/row/card it sits in), not by the "
        "button label — the labels are identical.\n"
        "TEMPORAL SELF-CHECK (do this every step): \"My goal is the target above. If "
        "my last action on it failed or the screen contradicted my expectation, "
        "clicking a similar control for a NEIGHBORING item would violate my goal. So "
        "I must re-examine MY target, or halt and let the ensemble vote — I must NOT "
        "blindly act on an adjacent look-alike.\""
    )
