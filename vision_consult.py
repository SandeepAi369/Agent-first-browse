"""Vision-on-demand — the a11y-DOM ⇄ vision "thinker" toggle (V21).

PRINCIPLE (the user's instruction, made literal)
═════════════════════════════════════════════════
The agent operates on the ACCESSIBILITY DOM by default — it is fast (~16ms/
extract), cheap (text tokens), and sufficient for the vast majority of steps.
It "opens its eyes" (one screenshot → vision model) ONLY when it is genuinely
confused or needs visual confirmation, resolves THAT step, and immediately falls
back to the a11y DOM. Vision is never the default and never sticky.

This is a thinker, not a parrot: the worker SELF-ASSESSES whether the DOM is
enough (`needs_vision`), and the system adds objective safety-net triggers
(repeated ineffective clicks; the escalation ladder's vision rung) so it also
gets eyes when it's stuck but hasn't realised it.

The vision model is given the a11y element MAP alongside the screenshot, so its
answer comes back as a stable `element_id` (resolved via the V19 registry) — not
brittle pixel coordinates. If vision is unavailable or unsure, we keep the a11y
decision (strict superset, no regression).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

try:
    from app.logger import get_logger
    logger = get_logger("vision_consult")
except ImportError:
    logger = logging.getLogger("vision_consult")


# Per-task budget — vision is expensive + slow vs the a11y DOM, so it stays rare.
MAX_VISION_CONSULTS = 5
# Ineffective-action streak on the SAME target that forces a visual look even if
# the worker didn't ask (the DOM says "click" but the page never responds).
INEFFECTIVE_STREAK_TRIGGER = 2
# Below this vision confidence we keep the a11y decision rather than override it.
MIN_VISION_CONFIDENCE = 0.55


class VisionVerdict(BaseModel):
    """The vision model's grounded read of the screenshot, mapped back to an
    action on the labelled a11y elements (stable id, not pixels)."""

    observation: str = Field(
        description="What the SCREENSHOT actually shows that's relevant to the question."
    )
    action_type: str = Field(
        description=(
            "The action to take now, refined by what you SEE: one of "
            "'click','type','scroll','press_enter','wait','done','none'. "
            "Use 'none' if the screenshot does not change the a11y plan."
        )
    )
    element_id: str | None = Field(
        default=None,
        description="The element id (e.g. 'e7') from the ELEMENT MAP to act on.",
    )
    text: str | None = Field(default=None, description="Text to type, if action_type='type'.")
    reasoning: str = Field(description="Why the screenshot leads to this action.")
    confidence: float = Field(description="0.0-1.0 — how sure you are from the image.")


VISION_SYSTEM_PROMPT = """You are the VISUAL CORTEX of a browser agent. The agent normally works from
the accessibility DOM (text), but it just hit something it cannot resolve from
text alone and switched to you for ONE look at the screen.

You are given: the objective, the question/ambiguity that triggered this look,
the agent's accessibility ELEMENT MAP (labelled e1,e2,… with positions), and a
SCREENSHOT of the current viewport.

Your job:
1. Look at the screenshot and answer the specific question.
2. Map what you see back to the ELEMENT MAP — return the element_id of the thing
   to act on (NOT pixel coordinates). Correlate by label text and position.
3. Catch what text-only perception misses: which of several look-alike controls
   is the REAL/primary one, whether a prior click actually took effect, whether
   an overlay/popup is covering the target, whether the goal is visually already
   done (a confirmation/toast/cart badge), or whether the layout means the target
   is elsewhere.
4. If the screenshot does NOT change the plan, return action_type='none' — do not
   invent work. Be decisive and ground every claim in what is visibly on screen.
"""


def should_consult_vision(
    *,
    needs_vision: bool,
    state: dict,
    action_type: str = "",
) -> tuple[bool, str]:
    """Decide whether THIS step earns a vision look. Returns (consult, reason).

    Triggers (any one), all gated by the per-task budget:
      1. the worker self-flagged `needs_vision` (the thinker's own call);
      2. the escalation ladder reached its 'vision' rung (state.force_vision);
      3. an ineffective-action streak on the same target (the DOM says act, the
         page doesn't respond) — a classic "the element isn't what the DOM thinks"
         case that only a screenshot disambiguates.
    """
    used = int(state.get("vision_consults", 0) or 0)
    if used >= MAX_VISION_CONSULTS:
        return False, f"budget exhausted ({used}/{MAX_VISION_CONSULTS})"

    # Don't burn a screenshot on a pure 'wait'.
    if action_type == "wait":
        return False, "wait action — no visual gain"

    if needs_vision:
        return True, "worker requested visual confirmation"
    if state.get("force_vision"):
        return True, "escalation ladder vision rung"
    streak = int(state.get("ineffective_streak", 0) or 0)
    if streak >= INEFFECTIVE_STREAK_TRIGGER:
        return True, f"{streak} ineffective actions on the same target"
    return False, ""


def _build_vision_messages(objective, question, a11y_markdown, history_tail, base64_image):
    """Build the multimodal message list (image attached via the model layer's
    base64_image path, mirroring advanced_agent's vision call)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    user = (
        f"═══ OBJECTIVE ═══\n{objective}\n\n"
        f"═══ WHY YOU'RE LOOKING (the ambiguity) ═══\n{question or 'Resolve the current step visually.'}\n\n"
        f"═══ RECENT ACTIONS ═══\n{history_tail or '(none)'}\n\n"
        f"═══ ACCESSIBILITY ELEMENT MAP (map what you see to these ids) ═══\n"
        f"{a11y_markdown[:4000]}\n\n"
        "Look at the attached screenshot and return the grounded action."
    )
    return [SystemMessage(content=VISION_SYSTEM_PROMPT), HumanMessage(content=user)]


async def consult_vision(
    invoke_fn,
    vision_chain: list,
    breaker,
    health_tracker,
    *,
    objective: str,
    question: str,
    a11y_markdown: str,
    history_tail: str = "",
) -> tuple[VisionVerdict | None, str]:
    """Capture one screenshot and ask the vision model to resolve the step.

    Returns (verdict, used_model). verdict is None when vision is unavailable,
    the screenshot fails, or the model can't answer — caller keeps the a11y
    decision (no regression). Never raises.
    """
    if invoke_fn is None or not vision_chain:
        return None, ""

    from mcp_tools import mcp_screenshot
    shot = await mcp_screenshot(full_page=False)
    if not shot.get("ok"):
        logger.debug("vision consult: screenshot failed (%s)", shot.get("error"))
        return None, ""

    messages = _build_vision_messages(
        objective, question, a11y_markdown, history_tail, shot["base64"]
    )
    try:
        verdict, used_model = await invoke_fn(
            vision_chain, messages, VisionVerdict,
            breaker, base64_image=shot["base64"], health_tracker=health_tracker,
        )
        if verdict is None:
            return None, used_model
        logger.info(
            "👁️ Vision consult (%s): %s → %s%s [%.0f%%]",
            used_model, verdict.observation[:70], verdict.action_type,
            f" {verdict.element_id}" if verdict.element_id else "",
            float(verdict.confidence) * 100,
        )
        return verdict, used_model
    except Exception as e:
        logger.warning("vision consult unavailable (%s) — keeping a11y decision",
                       str(e)[:120])
        return None, ""


def apply_vision_verdict(proposed: dict, verdict: VisionVerdict) -> tuple[dict, bool]:
    """Merge a confident vision verdict into the a11y proposed action.

    Returns (proposed, overridden). We override only when the vision model is
    confident AND proposes a concrete action — otherwise the a11y decision stands
    (vision confirmed it, or had nothing to add).
    """
    if verdict is None:
        return proposed, False
    if float(verdict.confidence) < MIN_VISION_CONFIDENCE:
        return proposed, False
    if verdict.action_type in ("", "none"):
        return proposed, False

    proposed = dict(proposed)
    proposed["verb"] = verdict.action_type
    if verdict.element_id:
        proposed["element_id"] = verdict.element_id
        # Coordinates are re-resolved from the id by the V19 registry at action
        # time; drop stale a11y coords so they can't fight the vision choice.
        proposed["x"] = None
        proposed["y"] = None
    if verdict.text is not None:
        proposed["text"] = verdict.text
    proposed["reasoning"] = f"[vision] {verdict.reasoning[:180]}"
    proposed["vision_used"] = True
    return proposed, True
