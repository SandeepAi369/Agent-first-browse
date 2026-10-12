"""Adaptive Perception Engine — situational, strategy-routed page perception (V29 / P0).

WHY
═══
A single parsing method can't see every page. The agent must adapt: use the fast,
proven a11y scan by default, escalate to a deeper interactive sweep only when that
is insufficient, and fall to vision when the DOM is unresolvable. This module is the
ROUTER that makes that choice — the "thinking" about HOW to look, separate from the
extraction itself.

UNIVERSAL-ONLY (absolute mandate)
═════════════════════════════════
There is NO site/domain logic anywhere in this module — no brand names, no
per-site rules, no commerce keywords. Every decision is made from universal signals:
element counts, geometry, accessibility standards, and the situation on the state.
A source-grep test (test_perception_engine_v29) enforces this mechanically so it can
never regress.

NO DUPLICATION (audited)
════════════════════════
Tier-1 is a THIN ADAPTER over the existing, proven `mcp_tools.mcp_snapshot`
(→ dom_parser.extract) — the only live perception call in the graph. This module
orchestrates strategies; it NEVER re-implements extraction, and `dom_parser` is left
verbatim as the Tier-1 foundation (H3).

P0 SCOPE
════════
The PerceptionStrategy interface + the router scaffold, with Tier-1 wired through it
as a behavior-identical passthrough. The router already exposes the seams a later
phase needs — an ordered, registrable strategy list (modularity) and a universal
`is_sufficient` check (computed and LOGGED now; it gates escalation to the Tier-2
deep CDP sweep and Tier-3 vision in a later phase). Callers never change as tiers
are added.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("perception_engine")

# Universal sufficiency floor: a usable page exposes at least this many interactive
# elements to reason about. Below it, Tier-1 is "sparse" — a later phase escalates
# to the deep sweep; P0 only observes/logs. Pure count heuristic, no site assumptions.
SPARSE_ELEMENT_FLOOR = 2

# ── AP-P1 strict viewport (universal) ──
# A small margin absorbs sub-pixel rounding ONLY; anything meaningfully outside the
# viewport (e.g. centre y = -34) is dropped.
VIEWPORT_MARGIN_PX = 2
# Off-screen elements of these kinds are genuine controls and are ALWAYS preserved
# (tagged offscreen) so a task-critical button/field is never silently dropped.
# Universal element kinds (from dom_parser), not site/commerce labels.
_ACTIONABLE_KINDS = ("button", "input")
_ELEMENT_ID_RE = re.compile(r"\[(e\d+)\]")


@dataclass
class PerceptionResult:
    """One perception pass, in the EXACT canonical contract every node already
    consumes (elements / markdown / selector_map / element_count) plus routing
    metadata. Keeping this identical to the snapshot dict is what lets the engine
    drop in without changing Overwatch, the workers, Target Lock, or the registry."""
    elements: list[dict] = field(default_factory=list)
    markdown: str = ""
    selector_map: dict[str, Any] = field(default_factory=dict)
    element_count: int = 0
    tier: int = 1
    strategy: str = ""
    sufficient: bool = True
    note: str = ""

    @classmethod
    def from_snapshot(cls, snap: dict, *, tier: int, strategy: str) -> "PerceptionResult":
        """Wrap a raw snapshot dict (mcp_snapshot / dom_parser.extract output)."""
        snap = snap or {}
        return cls(
            elements=snap.get("elements", []) or [],
            markdown=snap.get("markdown", "") or "",
            selector_map=snap.get("selector_map", {}) or {},
            element_count=int(snap.get("element_count", 0) or 0),
            tier=tier,
            strategy=strategy,
        )


@runtime_checkable
class PerceptionStrategy(Protocol):
    """A pluggable way to perceive the page. Future tiers (deep CDP sweep, vision)
    and any tool we absorb later implement this same shape — the router and every
    downstream consumer stay untouched (modularity)."""

    name: str
    tier: int

    async def extract(self, page, ctx: dict) -> PerceptionResult: ...


class Tier1A11yStrategy:
    """Tier-1 — the proven a11y/DOM scan, via the existing `mcp_snapshot`
    (→ dom_parser.extract). A thin adapter only: ONE extraction pipeline, no copy."""

    name = "a11y_dom"
    tier = 1

    async def extract(self, page, ctx: dict) -> PerceptionResult:
        from mcp_tools import mcp_snapshot
        snap = await mcp_snapshot()
        return PerceptionResult.from_snapshot(snap, tier=self.tier, strategy=self.name)


def default_strategies() -> list[PerceptionStrategy]:
    """Ordered cheapest-first. Later phases append the deep sweep / vision tiers
    here; nothing else changes."""
    return [Tier1A11yStrategy()]


# ═══════════════════════════════════════════════════════════════════════════════
#  AP-P1 — Strict viewport filter (UNIVERSAL post-pass; dom_parser untouched, H3)
# ═══════════════════════════════════════════════════════════════════════════════
#  "If it isn't physically on screen, it doesn't exist." We drop elements whose
#  CENTRE is outside the viewport, with ONE universal recall exemption: preserve an
#  off-screen element (tag offscreen:true) when it is contextually critical — a
#  genuine interactive control (button/input) OR relevant to the CURRENT goal (its
#  text overlaps the goal tokens). No brand/site/commerce rules — pure DOM kinds,
#  geometry, and the situation. This both declutters and tells the agent what to
#  scroll to, instead of pretending off-screen things are clickable now.

def _on_screen(x, y, vw: int, vh: int, margin: int) -> bool:
    try:
        return (-margin <= float(x) <= vw + margin) and (-margin <= float(y) <= vh + margin)
    except (TypeError, ValueError):
        return True  # no/unparseable coords → cannot judge → KEEP (never drop blindly)


def _goal_relevant(text: str, goal_tokens: set) -> bool:
    """Does this element's text relate to what the agent is trying to do now?
    Reuses the Target-Lock universal tokenizer (no duplication)."""
    if not goal_tokens:
        return False
    try:
        from target_lock import _tokens
        return bool(_tokens(text) & goal_tokens)
    except Exception:  # noqa: BLE001
        return False


def _filter_markdown(markdown: str, kept_ids: set, offscreen_ids: set) -> str:
    """Drop the lines of removed elements; tag the lines of preserved off-screen
    ones. Non-element lines (container headers) pass through untouched."""
    if not markdown:
        return markdown
    out = []
    for line in markdown.split("\n"):
        m = _ELEMENT_ID_RE.search(line)
        if not m:
            out.append(line)
            continue
        eid = m.group(1)
        if eid not in kept_ids:
            continue
        if eid in offscreen_ids:
            line = line.rstrip() + "  (offscreen — scroll to reach)"
        out.append(line)
    return "\n".join(out)


def strict_viewport_filter(elements, selector_map, markdown, *, vw: int, vh: int,
                           goal_tokens: set | None = None,
                           margin: int = VIEWPORT_MARGIN_PX):
    """Apply the universal strict-viewport pass. Returns
    (elements, selector_map, markdown, n_dropped, n_offscreen). On-screen and
    coordless elements pass through UNCHANGED; only preserved off-screen elements
    gain `offscreen: True`."""
    goal_tokens = goal_tokens or set()
    kept_ids: set = set()
    offscreen_ids: set = set()
    out_elements: list[dict] = []
    dropped = 0
    for el in (elements or []):
        eid = el.get("id")
        if _on_screen(el.get("x"), el.get("y"), vw, vh, margin):
            out_elements.append(el)               # unchanged — byte-identical
            if eid:
                kept_ids.add(eid)
            continue
        kind = (el.get("kind") or "").lower()
        critical = (kind in _ACTIONABLE_KINDS
                    or _goal_relevant(f"{el.get('text', '')} {el.get('hint', '')}", goal_tokens))
        if critical:
            out_elements.append({**el, "offscreen": True})
            if eid:
                kept_ids.add(eid)
                offscreen_ids.add(eid)
        else:
            dropped += 1                          # off-screen, non-actionable, off-goal → noise
    if selector_map:
        out_map = {k: v for k, v in selector_map.items() if k in kept_ids}
    else:
        out_map = selector_map or {}
    out_md = _filter_markdown(markdown, kept_ids, offscreen_ids)
    return out_elements, out_map, out_md, dropped, len(offscreen_ids)


def _viewport_of(page) -> tuple[int, int]:
    vw, vh = 1920, 1080
    try:
        vp = page.viewport_size
        if vp:
            vw = int(vp.get("width", vw) or vw)
            vh = int(vp.get("height", vh) or vh)
    except Exception:  # noqa: BLE001
        pass
    return vw, vh


def _goal_tokens(ctx: dict) -> set:
    """Goal-relevance tokens for the recall exemption — reuses Target Lock."""
    try:
        from target_lock import extract_target
        return extract_target(ctx.get("objective", ""), ctx.get("bound_target", "")).tokens
    except Exception:  # noqa: BLE001
        return set()


def is_sufficient(result: PerceptionResult, ctx: dict | None = None) -> bool:
    """Universal sufficiency check — did this pass surface anything actionable to
    reason about? Pure element-count heuristic; no site/domain assumptions. (A later
    phase enriches this with a sub-goal-relevance signal from `ctx`.)"""
    return result.element_count >= SPARSE_ELEMENT_FLOOR


async def perceive(page, ctx: dict | None = None,
                   strategies: list[PerceptionStrategy] | None = None) -> PerceptionResult:
    """Run the perception cascade and return the chosen result.

    P0: Tier-1 only. The sufficiency verdict is computed and LOGGED for
    observability; escalation to deeper tiers is wired in a later phase at exactly
    this seam, so callers never change. Always returns a valid PerceptionResult.
    """
    ctx = ctx or {}
    strategies = strategies or default_strategies()

    result = await strategies[0].extract(page, ctx)

    # AP-P1: universal strict-viewport pass over Tier-1 output (dom_parser untouched).
    try:
        from feature_flags import strict_viewport_enabled
        if strict_viewport_enabled():
            vw, vh = _viewport_of(page)
            els, smap, md, dropped, n_off = strict_viewport_filter(
                result.elements, result.selector_map, result.markdown,
                vw=vw, vh=vh, goal_tokens=_goal_tokens(ctx))
            result.elements, result.selector_map, result.markdown = els, smap, md
            result.element_count = len(els)
            if dropped or n_off:
                logger.info("👁️ Strict viewport: dropped %d off-screen noise; kept %d "
                            "off-screen actionable (tagged offscreen, scroll-to-reach).",
                            dropped, n_off)
    except Exception as e:  # noqa: BLE001 — filtering never breaks perception
        logger.debug("Strict viewport filter skipped (non-fatal): %s", e)

    result.sufficient = is_sufficient(result, ctx)
    if not result.sufficient:
        result.note = f"sparse ({result.element_count} elements)"
        logger.info("👁️ Perception [%s tier-%d] SPARSE (%d elements) — deeper-tier "
                    "escalation lands in a later phase; using Tier-1 result for now.",
                    result.strategy, result.tier, result.element_count)
    else:
        logger.debug("👁️ Perception [%s tier-%d] ok (%d elements)",
                     result.strategy, result.tier, result.element_count)
    return result
