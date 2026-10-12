"""Action risk classifier for browser automation safety.

Classifies every candidate browser action as REVERSIBLE, CAUTIOUS, or
IRREVERSIBLE *before* execution so the agent loop can decide whether to
proceed directly, run a WebDreamer simulation first, or ask for human
confirmation.

Usage::

    from action_classifier import classify_action, requires_simulation, ActionRisk

    risk = classify_action(
        action_type="click",
        target_name="submit-btn",
        target_text="Place Order",
        url="https://shop.example.com/checkout",
        element_kind="button",
    )
    if requires_simulation(risk, step_count=5):
        # run WebDreamer before executing
        ...
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Sequence

try:
    from app.logger import get_logger
    logger = get_logger('action_classifier')
except ImportError:
    import logging
    logger = logging.getLogger('action_classifier')

# ---------------------------------------------------------------------------
# Public enum
# ---------------------------------------------------------------------------


class ActionRisk(Enum):
    """Risk level assigned to a browser action before execution.

    * **REVERSIBLE** – safe to execute without simulation (e.g. navigation,
      scrolling, pressing Enter on a search bar).
    * **CAUTIOUS** – modifies page state in a way that *might* be undoable but
      warrants extra scrutiny after several agent steps (e.g. typing into a
      form, clicking an ambiguous button).
    * **IRREVERSIBLE** – triggers a side-effect that cannot easily be undone
      (e.g. submitting an order, deleting a record, signing out).  Should
      always be simulated or confirmed before execution.
    """

    REVERSIBLE = "reversible"
    CAUTIOUS = "cautious"
    IRREVERSIBLE = "irreversible"


# ---------------------------------------------------------------------------
# Pattern sets (compiled once at import time)
# ---------------------------------------------------------------------------

# Patterns whose presence in target_name or target_text signals an
# irreversible action.  Order doesn't matter; we check *all* of them.
IRREVERSIBLE_PATTERNS: Sequence[re.Pattern[str]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsubmit\b",
        r"\bpost\b",
        r"\bpublish\b",
        r"\bbuy\b",
        r"\bpurchase\b",
        r"\bdelete\b",
        r"\bremove\b",
        r"\bplace\s*order\b",
        r"\bconfirm\b",
        r"\bsend\b",
        r"\bcheckout\b",
        r"\bcheck\s*out\b",
        r"\bpay\b",
        r"\badd\s*to\s*cart\b",
        r"\bsign\s*out\b",
        r"\blog\s*out\b",
        r"\bunsubscribe\b",
    )
)

# Action types that are unconditionally safe.
_ALWAYS_REVERSIBLE_ACTIONS: frozenset[str] = frozenset({
    "goto",
    "scroll",
    "wait",
    "done",
    "hover",      # V29: pointer move only — never mutates state
    "scroll_to",  # V33: directional scroll — never mutates state
})

# Element kinds that are generally navigational (and therefore safer).
_NAVIGATIONAL_ELEMENT_KINDS: frozenset[str] = frozenset({
    "link",
    "a",
    "anchor",
    "nav",
    "menuitem",
})

# Element kinds that are associated with form submission / mutations.
_BUTTON_ELEMENT_KINDS: frozenset[str] = frozenset({
    "button",
    "btn",
    "submit",
    "input",
    "reset",
})

# Heuristic: if the user presses Enter on an element whose name/text
# contains any of these tokens, treat it as a safe search action.
_SEARCH_TOKENS: frozenset[str] = frozenset({
    "search",
    "query",
    "find",
    "lookup",
    "filter",
    "q",
})


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------


def _matches_irreversible(text: str) -> re.Pattern[str] | None:
    """Return the first irreversible pattern that matches *text*, or None."""
    for pattern in IRREVERSIBLE_PATTERNS:
        if pattern.search(text):
            return pattern
    return None


def _is_search_context(target_name: str, target_text: str) -> bool:
    """Return True if the target looks like a search bar / search input."""
    combined = f"{target_name} {target_text}".lower()
    return any(token in combined for token in _SEARCH_TOKENS)


def classify_action(
    action_type: str,
    target_name: str = "",
    target_text: str = "",
    url: str = "",
    element_kind: str = "",
) -> ActionRisk:
    """Classify a browser action by its risk level.

    The classification is intentionally conservative — when in doubt the
    function returns the *higher* risk level so the agent loop can apply
    additional safety checks (e.g. WebDreamer simulation).

    Parameters
    ----------
    action_type:
        The primitive action verb, e.g. ``"click"``, ``"type"``,
        ``"goto"``, ``"scroll"``, ``"wait"``, ``"press_enter"``,
        ``"done"``.
    target_name:
        The DOM ``name`` / ``id`` / ``aria-label`` of the target element
        (may be empty).
    target_text:
        The visible inner-text of the target element (may be empty).
    url:
        The current page URL — reserved for future domain-level rules.
    element_kind:
        A normalised tag/role descriptor such as ``"button"``, ``"link"``,
        ``"input"`` (may be empty).

    Returns
    -------
    ActionRisk
        The assessed risk level for the action.
    """
    action = action_type.strip().lower()
    name_lower = target_name.strip().lower()
    text_lower = target_text.strip().lower()
    kind_lower = element_kind.strip().lower()

    # ------------------------------------------------------------------
    # 1. Unconditionally safe actions
    # ------------------------------------------------------------------
    if action in _ALWAYS_REVERSIBLE_ACTIONS:
        logger.debug(
            "Action '%s' is unconditionally REVERSIBLE.",
            action,
        )
        return ActionRisk.REVERSIBLE

    # ------------------------------------------------------------------
    # 2. press_enter — safe when on a search bar, cautious otherwise
    # ------------------------------------------------------------------
    if action == "press_enter":
        if _is_search_context(name_lower, text_lower):
            logger.debug(
                "press_enter on search context ('%s' / '%s') → REVERSIBLE.",
                target_name,
                target_text,
            )
            return ActionRisk.REVERSIBLE
        logger.debug(
            "press_enter on non-search context ('%s' / '%s') → CAUTIOUS.",
            target_name,
            target_text,
        )
        return ActionRisk.CAUTIOUS

    # ------------------------------------------------------------------
    # 3. type — cautious by default (modifies form state)
    # ------------------------------------------------------------------
    if action == "type":
        logger.debug("Action 'type' is CAUTIOUS by default.")
        return ActionRisk.CAUTIOUS

    # ------------------------------------------------------------------
    # 4. click — depends on the target
    # ------------------------------------------------------------------
    if action == "click":
        # 4a. Check for irreversible patterns in name / text.
        match = _matches_irreversible(name_lower) or _matches_irreversible(
            text_lower,
        )
        if match:
            # V15.0 F4: URL context gate — distinguish navigation-to-compose from submission
            # "Create Post" on a feed page (/r/test/) = navigation, not submission
            # "Post" on a compose page (/r/test/submit/) = actual submission
            COMPOSE_URL_PATTERNS = ('/submit', '/compose', '/new', '/create', '/editor',
                                    '/checkout', '/review', '/confirm')
            on_compose_page = any(p in (url or '') for p in COMPOSE_URL_PATTERNS)
            matched_word = match.pattern.replace(r'\b', '').replace('\\s*', ' ').strip()

            if not on_compose_page and matched_word in (
                'submit', 'post', 'publish', 'send',
            ):
                logger.info(
                    "click on '%s' / '%s' matched /%s/ but NOT on compose page "
                    "(url='%s') → CAUTIOUS (navigation, not submission).",
                    target_name,
                    target_text,
                    match.pattern,
                    url[:80],
                )
                return ActionRisk.CAUTIOUS

            logger.info(
                "click on '%s' / '%s' matched irreversible pattern /%s/ "
                "→ IRREVERSIBLE.",
                target_name,
                target_text,
                match.pattern,
            )
            return ActionRisk.IRREVERSIBLE

        # 4b. Navigation links are generally reversible.
        if kind_lower in _NAVIGATIONAL_ELEMENT_KINDS:
            logger.debug(
                "click on navigational element ('%s', kind='%s') → REVERSIBLE.",
                target_text or target_name,
                element_kind,
            )
            return ActionRisk.REVERSIBLE

        # 4c. Buttons are cautious by default (could submit a form).
        if kind_lower in _BUTTON_ELEMENT_KINDS:
            logger.debug(
                "click on button-like element ('%s', kind='%s') → CAUTIOUS.",
                target_text or target_name,
                element_kind,
            )
            return ActionRisk.CAUTIOUS

        # 4d. Unknown element kind — be cautious.
        logger.debug(
            "click on element with unknown kind ('%s', kind='%s') → CAUTIOUS.",
            target_text or target_name,
            element_kind,
        )
        return ActionRisk.CAUTIOUS

    # ------------------------------------------------------------------
    # 5. V33: New action types — CAUTIOUS by default (mutate form state)
    # ------------------------------------------------------------------
    if action in ("select_option", "press_combo", "drag_and_drop", "upload_file"):
        logger.debug("V33 action '%s' is CAUTIOUS by default.", action)
        return ActionRisk.CAUTIOUS

    # ------------------------------------------------------------------
    # 6. Unrecognised action — default to CAUTIOUS
    # ------------------------------------------------------------------
    logger.warning(
        "Unrecognised action type '%s' — defaulting to CAUTIOUS.",
        action_type,
    )
    return ActionRisk.CAUTIOUS


# ---------------------------------------------------------------------------
# Simulation gate
# ---------------------------------------------------------------------------


def requires_simulation(risk: ActionRisk, step_count: int = 0) -> bool:
    """Decide whether an action should go through WebDreamer simulation.

    The goal is to avoid costly simulations for clearly safe actions while
    ensuring that dangerous or ambiguous actions are vetted.

    Parameters
    ----------
    risk:
        The :class:`ActionRisk` classification of the action.
    step_count:
        How many steps the agent has already taken in the current task.
        A higher step count signals that the agent has been "thinking" for
        a while, making it more likely that the next action has accumulated
        compounding uncertainty.

    Returns
    -------
    bool
        ``True`` if the action should be simulated before execution.
    """
    if risk is ActionRisk.IRREVERSIBLE:
        logger.info(
            "IRREVERSIBLE action → simulation required (step_count=%d).",
            step_count,
        )
        return True

    if risk is ActionRisk.CAUTIOUS and step_count > 3:
        logger.info(
            "CAUTIOUS action at step %d (>3) → simulation required.",
            step_count,
        )
        return True

    logger.debug(
        "risk=%s, step_count=%d → simulation NOT required.",
        risk.value,
        step_count,
    )
    return False
