"""DOM Diff — lightweight universal state-change detection (V29 Phase A).

WHY
═══
CriticV12 already diffs the interactive-element SET (count, refs, target state). It
misses SUBTLE changes: a click that opens a small overlay using existing nodes, an
`aria-expanded` toggle, a focus move — where the interactive-set barely changes, so
the action looks like "no progress" and the agent loops (the reported stagnation).

This module adds a CHEAP, UNIVERSAL "page-signal vector": ~8 numbers captured in ONE
`page.evaluate` from pure DOM/ARIA standards (no site/brand rules), diffed in code.
The diff NEVER reaches the LLM — only a one-line verdict does — so it cannot bloat
context. It composes with CriticV12 (raises sensitivity) and feeds a unified
`state_change_score` that Stagnation/Reality can consume.

Pure logic + a tiny JS string. Unit-testable offline (no browser).
"""

from __future__ import annotations

# One evaluate, ~8 universal signals. Deliberately layout-light: querySelectorAll
# length + cheap property reads only — no innerText / getBoundingClientRect /
# getComputedStyle (those force expensive reflow). Universal DOM/ARIA selectors.
PAGE_SIGNAL_JS = r"""
() => {
  const d = document;
  const se = d.scrollingElement || d.documentElement || d.body;
  const n = (s) => { try { return d.querySelectorAll(s).length; } catch(_) { return 0; } };
  let activeTag = '';
  try { activeTag = (d.activeElement && d.activeElement.tagName) || ''; } catch(_) {}
  let path = '';
  try { path = (location.pathname + location.search).slice(0, 120); } catch(_) {}
  return {
    allCount: (d.getElementsByTagName('*') || []).length,
    childCount: (d.body && d.body.childElementCount) || 0,
    dialogs: n('[role="dialog"],[aria-modal="true"],[role="alertdialog"]'),
    expanded: n('[aria-expanded="true"]'),
    interactives: n('a,button,input,select,textarea,[role="button"],[role="link"],[role="menuitem"],[role="tab"],[contenteditable="true"]'),
    activeTag: activeTag,
    scrollH: Math.round((se && se.scrollHeight) || 0),
    path: path,
  };
}
"""

# Numeric signals tolerate tiny jitter (ads/async counters) before counting as changed.
_NUMERIC_TOL = {"allCount": 2, "scrollH": 8, "childCount": 0,
                "dialogs": 0, "expanded": 0, "interactives": 0}
_NUMERIC_KEYS = ("allCount", "childCount", "dialogs", "expanded", "interactives", "scrollH")
_STRING_KEYS = ("activeTag", "path")


def signal_vector_diff(pre: dict | None, post: dict | None) -> dict:
    """Compare two page-signal vectors. Returns
    {changed:int, keys:[...], dialogs_delta:int, expanded_delta:int, meaningful:bool}.
    `meaningful` is the universal "something real happened" signal: an overlay/dialog
    appeared/closed, a section expanded/collapsed, the route changed, or ≥2 signals
    moved at once."""
    if not pre or not post:
        return {"changed": 0, "keys": [], "dialogs_delta": 0,
                "expanded_delta": 0, "meaningful": False}
    keys: list[str] = []
    for k in _NUMERIC_KEYS:
        if abs(int(post.get(k, 0) or 0) - int(pre.get(k, 0) or 0)) > _NUMERIC_TOL.get(k, 0):
            keys.append(k)
    for k in _STRING_KEYS:
        if (pre.get(k) or "") != (post.get(k) or ""):
            keys.append(k)
    dialogs_delta = int(post.get("dialogs", 0) or 0) - int(pre.get("dialogs", 0) or 0)
    expanded_delta = int(post.get("expanded", 0) or 0) - int(pre.get("expanded", 0) or 0)
    meaningful = bool(dialogs_delta != 0 or expanded_delta != 0
                      or "path" in keys or len(keys) >= 2)
    return {"changed": len(keys), "keys": keys, "dialogs_delta": dialogs_delta,
            "expanded_delta": expanded_delta, "meaningful": meaningful}


def state_change_score(*, url_changed: bool = False, semantic_changed: bool = False,
                       element_delta: int = 0, new_count: int = 0,
                       disappeared_count: int = 0, vector: dict | None = None) -> float:
    """Unify all change signals into one [0,1] score (the value/reward signal that
    Stagnation and the world-model later consume). Pure, monotonic, bounded."""
    score = 0.0
    if url_changed:
        score += 0.5
    if semantic_changed:
        score += 0.25
    score += 0.12 * min(1.0, abs(int(element_delta)) / 3.0)
    score += 0.12 * min(1.0, (int(new_count) + int(disappeared_count)) / 3.0)
    if vector:
        if vector.get("dialogs_delta") or vector.get("expanded_delta"):
            score += 0.30
        score += 0.10 * min(1.0, int(vector.get("changed", 0)) / 4.0)
    return max(0.0, min(1.0, score))


def progress_phrase(vector: dict) -> str:
    """One-line, human/LLM-readable description of the vector change (the ONLY thing
    that reaches the LLM — never the raw diff)."""
    if not vector:
        return ""
    if vector.get("dialogs_delta", 0) > 0:
        return "an overlay/dialog appeared"
    if vector.get("dialogs_delta", 0) < 0:
        return "an overlay/dialog closed"
    if vector.get("expanded_delta", 0):
        return "a section expanded/collapsed"
    keys = vector.get("keys", [])
    return f"page state changed ({', '.join(keys[:3])})" if keys else ""
