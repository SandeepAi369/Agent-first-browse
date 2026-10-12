"""Vision-on-demand — the a11y-DOM ⇄ vision "thinker" toggle (V21 + V30 coord fallback).

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
answer comes back as a stable `element_id` (resolved via the V19 registry).

V30 COORDINATE FALLBACK: when the target element is genuinely absent from the
a11y tree (shadow DOM, custom web components), Vision may return pixel coordinates
instead. These coordinates are validated by a second vision check ("Look-Before-
You-Leap") before being sent to Overwatch, ensuring grounded, safe clicks.

If vision is unavailable or unsure, we keep the a11y decision (no regression).
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
    action on the labelled a11y elements (stable id, not pixels).

    V30: When the target is genuinely absent from the element map (shadow DOM,
    custom web component), coord_x/coord_y provide pixel-coordinate fallback."""

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
    coord_x: float | None = Field(
        default=None,
        description=(
            "COORDINATE FALLBACK — only when the target is clearly visible on the "
            "screenshot but genuinely ABSENT from the ELEMENT MAP (e.g. shadow DOM "
            "button, custom web component). Return the center X pixel coordinate "
            "of the visual target. Leave null when element_id is provided."
        ),
    )
    coord_y: float | None = Field(
        default=None,
        description=(
            "COORDINATE FALLBACK — only when the target is clearly visible on the "
            "screenshot but genuinely ABSENT from the ELEMENT MAP. Return the center "
            "Y pixel coordinate of the visual target. Leave null when element_id "
            "is provided."
        ),
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
   to act on. Correlate by label text and position.
3. COORDINATE FALLBACK: If the visual target is clearly visible on the screenshot
   but does NOT appear in the ELEMENT MAP at all (e.g. a shadow DOM button, custom
   web component, or element hidden from the accessibility tree), set element_id
   to null and instead return the target's CENTER pixel coordinates as coord_x and
   coord_y. The coordinate frame matches the screenshot: (0,0) is at the TOP-LEFT,
   x increases rightward, y increases downward. Use this ONLY when you are certain
   the element is genuinely absent from the map — not just hard to match.
4. Catch what text-only perception misses: which of several look-alike controls
   is the REAL/primary one, whether a prior click actually took effect, whether
   an overlay/popup is covering the target, whether the goal is visually already
   done (a confirmation/toast/cart badge), or whether the layout means the target
   is elsewhere.
5. If the screenshot does NOT change the plan, return action_type='none' — do not
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


def _build_vision_messages(objective, question, a11y_markdown, history_tail, base64_image,
                           vp_w=0, vp_h=0):
    """Build the multimodal message list (image attached via the model layer's
    base64_image path, mirroring advanced_agent's vision call)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    bounds = (
        f"═══ VIEWPORT BOUNDS (strict) ═══\n"
        f"The screenshot IS the browser viewport and NOTHING else — {vp_w}×{vp_h} px. "
        f"The only valid coordinate frame is x∈[0,{vp_w}], y∈[0,{vp_h}] with (0,0) at "
        f"the TOP-LEFT of the image. There is no browser chrome, taskbar, or desktop "
        f"in this image — ignore any such notion. Never reference anything outside "
        f"these bounds.\n\n"
        if vp_w and vp_h else ""
    )
    user = (
        f"═══ OBJECTIVE ═══\n{objective}\n\n"
        f"═══ WHY YOU'RE LOOKING (the ambiguity) ═══\n{question or 'Resolve the current step visually.'}\n\n"
        + bounds
        + f"═══ RECENT ACTIONS ═══\n{history_tail or '(none)'}\n\n"
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
        objective, question, a11y_markdown, history_tail, shot["base64"],
        vp_w=shot.get("width", 0), vp_h=shot.get("height", 0),
    )
    try:
        verdict, used_model = await invoke_fn(
            vision_chain, messages, VisionVerdict,
            breaker, base64_image=shot["base64"], health_tracker=health_tracker,
        )
        if verdict is None:
            return None, used_model
        coord_info = ""
        if not verdict.element_id and verdict.coord_x is not None and verdict.coord_y is not None:
            coord_info = f" at ({verdict.coord_x:.0f},{verdict.coord_y:.0f})"
        logger.info(
            "👁️ Vision consult (%s): %s → %s%s%s [%.0f%%]",
            used_model, verdict.observation[:70], verdict.action_type,
            f" {verdict.element_id}" if verdict.element_id else "",
            coord_info,
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

    V30: Handles coordinate fallback — when Vision returns coord_x/coord_y
    instead of element_id (target absent from a11y tree), the coordinates are
    set on the proposed action with a 'vision_coords' flag for downstream
    validation before Overwatch grounding.
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
        proposed["vision_coords"] = False
    elif (getattr(verdict, "coord_x", None) is not None
          and getattr(verdict, "coord_y", None) is not None):
        # V30 COORDINATE FALLBACK: target not in a11y tree — Vision returned
        # pixel coordinates. Clear element_id and set coords for downstream
        # validation ("Look-Before-You-Leap" in the worker pipeline).
        proposed["element_id"] = None
        proposed["x"] = float(verdict.coord_x)
        proposed["y"] = float(verdict.coord_y)
        proposed["vision_coords"] = True
        logger.info(
            "👁️📍 Vision coordinate fallback: target at (%.0f, %.0f) — "
            "element not in a11y tree",
            verdict.coord_x, verdict.coord_y,
        )
    if verdict.text is not None:
        proposed["text"] = verdict.text
    proposed["reasoning"] = f"[vision] {verdict.reasoning[:180]}"
    proposed["vision_used"] = True
    return proposed, True
