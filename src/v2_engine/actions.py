"""v2_engine.actions — 2026-Standard Resilient Ref Action Executor.

Wraps the frozen legacy action backends (`mcp_click`, `mcp_type` → `cdp_click`,
`cdp_input`, `ghost_input`) with three reliability upgrades:

  1. Live Ref Pre-Flight + Stale-Ref Semantic Rebind:
     Verifies `window.__aid[eid]` is connected & enabled before acting. If an
     SPA re-render detached the node, deterministically rebinds `window.__aid[eid]`
     to the matching live DOM element by `(kind, label)` instead of blindly
     clicking stale coordinates — or fails cleanly with `stale_ref=True` and
     invalidates the delta compressor.
  2. Fixed V32 Element-State Safety Net:
     Captures pre/post DOM state directly from `window.__aid[eid]` (fixing
     legacy `mcp_tools.py:165` which queried non-existent `[__aid="..."]` DOM
     attributes) to detect checkbox/toggle/accordion/tab state transitions.
  3. Bounded Exponential-Backoff Retry on Transient Browser/DOM Errors:
     Retries transient detachment/interception/timeout errors with fresh ref
     re-resolution between attempts, while never retrying disabled or non-transient
     failures.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from v2_engine.metrics import RunMetrics
from v2_engine.perception import get_shared_compressor

logger = logging.getLogger("v2_engine.actions")

_TRANSIENT_ERROR_TOKENS = (
    "execution context was destroyed",
    "detached",
    "intercepted",
    "timed out",
    "timeout",
    "stale",
    "not attached",
    "node is not",
)

_INSPECT_REF_JS = r"""
(eid) => {
    const reg = window.__aid || {};
    const el = reg[eid];
    if (!el) return { found: false, reason: 'not_in_registry' };
    if (!el.isConnected) return { found: false, reason: 'detached' };
    const disabled = !!el.disabled || el.getAttribute('aria-disabled') === 'true';
    try { el.scrollIntoView({ block: 'center', inline: 'center' }); } catch (_) {}
    let r;
    try { r = el.getBoundingClientRect(); } catch (_) { return { found: false, reason: 'no_rect' }; }
    if (r.width < 1 || r.height < 1) return { found: false, reason: 'zero_size' };
    return {
        found: true,
        disabled,
        x: Math.round(r.left + r.width / 2),
        y: Math.round(r.top + r.height / 2),
        checked: !!el.checked,
        selected: !!el.selected,
        ariaChecked: el.getAttribute('aria-checked') || '',
        ariaExpanded: el.getAttribute('aria-expanded') || '',
        ariaSelected: el.getAttribute('aria-selected') || '',
        ariaPressed: el.getAttribute('aria-pressed') || '',
        open: !!el.open,
    };
}
"""

_REBIND_STALE_REF_JS = r"""
({ eid, kind, label }) => {
    const want = (label || '').replace(/\s+/g, ' ').trim().toLowerCase();
    if (!want) return { ok: false, reason: 'empty_label' };

    const kindSelectors = {
        button: 'button, [role="button"], [role="menuitem"], [role="tab"], '
              + '[role="switch"], [role="checkbox"], [role="radio"], '
              + 'input[type="submit"], input[type="button"]',
        link: 'a[href], [role="link"]',
        input: 'input, textarea, select, [role="textbox"], [role="searchbox"], [role="combobox"], [contenteditable="true"]',
    };
    const sel = kindSelectors[kind] || 'button, a[href], input, textarea, select, [role]';

    function nodeText(el) {
        return (
            el.getAttribute('aria-label')
            || el.textContent
            || el.getAttribute('placeholder')
            || el.getAttribute('title')
            || el.getAttribute('name')
            || ''
        ).replace(/\s+/g, ' ').trim().toLowerCase();
    }

    const matches = [];
    for (const el of document.querySelectorAll(sel)) {
        if (!el.isConnected) continue;
        let r;
        try { r = el.getBoundingClientRect(); } catch (_) { continue; }
        if (r.width < 2 || r.height < 2) continue;
        const st = window.getComputedStyle(el);
        if (st.display === 'none' || st.visibility === 'hidden' || parseFloat(st.opacity) <= 0.01) continue;
        const t = nodeText(el);
        if (t === want) {
            matches.push({ el, r });
        }
    }

    if (matches.length !== 1) {
        return { ok: false, reason: matches.length === 0 ? 'no_live_match' : 'ambiguous_match', count: matches.length };
    }

    const live = matches[0].el;
    try { live.scrollIntoView({ block: 'center', inline: 'center' }); } catch (_) {}
    const r = live.getBoundingClientRect();
    window.__aid = window.__aid || {};
    window.__aid[eid] = live;
    return {
        ok: true,
        x: Math.round(r.left + r.width / 2),
        y: Math.round(r.top + r.height / 2),
    };
}
"""


def _strip_state_suffix(text: str) -> str:
    cleaned = re.sub(r"\[(?:filled:[^\]]+|empty|checked|disabled|offscreen)\]", "", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_transient_error(err: str) -> bool:
    low = (err or "").lower()
    return any(tok in low for tok in _TRANSIENT_ERROR_TOKENS)


def _state_changed(before: dict[str, Any] | None, after: dict[str, Any] | None) -> bool:
    if not before or not after or not before.get("found") or not after.get("found"):
        return False
    keys = ("checked", "selected", "ariaChecked", "ariaExpanded", "ariaSelected", "ariaPressed", "open")
    return any(before.get(k) != after.get(k) for k in keys)


async def inspect_live_ref(page, element_id: str | None) -> dict[str, Any]:
    """Inspect a live element in `window.__aid` without mutating the DOM."""
    if not element_id:
        return {"found": False, "reason": "no_eid"}
    try:
        res = await asyncio.wait_for(page.evaluate(_INSPECT_REF_JS, element_id), timeout=1.5)
        return res if isinstance(res, dict) else {"found": False, "reason": "null_result"}
    except Exception as e:
        return {"found": False, "reason": str(e)[:80]}


async def rebind_stale_ref(
    page,
    element_id: str,
    *,
    kind: str = "",
    label: str = "",
) -> dict[str, Any]:
    """Attempt deterministic live DOM rebinding for a detached/stale element_id."""
    clean_label = _strip_state_suffix(label)
    if not clean_label:
        return {"ok": False, "reason": "no_label"}
    try:
        res = await asyncio.wait_for(
            page.evaluate(_REBIND_STALE_REF_JS, {
                "eid": element_id,
                "kind": (kind or "").lower(),
                "label": clean_label,
            }),
            timeout=1.5,
        )
        return res if isinstance(res, dict) else {"ok": False, "reason": "null_result"}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:80]}


def _lookup_element_meta(
    element_id: str | None,
    elements_list: list[dict[str, Any]] | None,
) -> tuple[str, str]:
    if not element_id or not elements_list:
        return "", ""
    for el in elements_list:
        if isinstance(el, dict) and el.get("id") == element_id:
            return str(el.get("kind") or ""), str(el.get("text") or "")
    return "", ""


async def v2_click(
    page,
    *,
    x: float = 0,
    y: float = 0,
    element_id: str | None = None,
    elements_list: list[dict[str, Any]] | None = None,
    max_retries: int = 2,
    metrics: RunMetrics | None = None,
    click_fn=None,
) -> dict[str, Any]:
    """V2 resilient click with stale-ref rebind, fixed __aid state check, and retry."""
    if click_fn is None:
        from mcp_tools import mcp_click
        click_fn = mcp_click

    kind, label = _lookup_element_meta(element_id, elements_list)
    last_result: dict[str, Any] = {"success": False, "error": "not attempted"}

    for attempt in range(max_retries + 1):
        pre_state: dict[str, Any] | None = None
        cur_x, cur_y = x, y

        if element_id:
            pre_state = await inspect_live_ref(page, element_id)
            if pre_state.get("found"):
                if pre_state.get("disabled"):
                    return {
                        "success": False,
                        "strategy": "",
                        "navigated": False,
                        "dom_changed": False,
                        "error": f"Element {element_id} is disabled",
                    }
                cur_x, cur_y = float(pre_state["x"]), float(pre_state["y"])
            else:
                rebound = await rebind_stale_ref(page, element_id, kind=kind, label=label)
                if rebound.get("ok"):
                    cur_x, cur_y = float(rebound["x"]), float(rebound["y"])
                    pre_state = await inspect_live_ref(page, element_id)
                    if metrics is not None:
                        metrics.record_stale_ref_recovery()
                    logger.info("♻️ V2 stale-ref rebound %s → (%.0f, %.0f)", element_id, cur_x, cur_y)
                elif kind or label:
                    # Element was genuinely detached and has no unique live match:
                    # invalidate delta snapshot and fail cleanly instead of blind coord click.
                    get_shared_compressor().invalidate()
                    return {
                        "success": False,
                        "stale_ref": True,
                        "strategy": "",
                        "navigated": False,
                        "dom_changed": False,
                        "error": (
                            f"STALE_REF: {element_id} detached "
                            f"({rebound.get('reason', 'missing')}); re-snapshot required"
                        ),
                    }

        last_result = await click_fn(x=cur_x, y=cur_y, element_id=element_id)
        if last_result.get("success"):
            return last_result

        # Fixed V32 state safety net: verify live window.__aid[eid] state transition
        if element_id and pre_state and pre_state.get("found"):
            post_state = await inspect_live_ref(page, element_id)
            if _state_changed(pre_state, post_state):
                if metrics is not None:
                    metrics.record_state_safety_override()
                logger.info("✅ V2 __aid state safety net verified transition on %s", element_id)
                return {
                    "success": True,
                    "strategy": "v2_aid_state_verified",
                    "navigated": False,
                    "dom_changed": True,
                    "error": "",
                }

        err = str(last_result.get("error", ""))
        if attempt < max_retries and _is_transient_error(err):
            if metrics is not None:
                metrics.record_action_retry()
            backoff = 0.15 * (2 ** attempt)
            logger.info("🔁 V2 click transient retry %d/%d after %.2fs (%s)",
                        attempt + 1, max_retries, backoff, err[:60])
            await asyncio.sleep(backoff)
            continue
        break

    return last_result


async def v2_type(
    page,
    *,
    text: str,
    x: float = 0,
    y: float = 0,
    element_id: str | None = None,
    elements_list: list[dict[str, Any]] | None = None,
    max_retries: int = 2,
    metrics: RunMetrics | None = None,
    type_fn=None,
) -> dict[str, Any]:
    """V2 resilient type with live ref validation, stale-ref rebind, and transient retry."""
    if type_fn is None:
        from mcp_tools import mcp_type
        type_fn = mcp_type

    kind, label = _lookup_element_meta(element_id, elements_list)
    last_result: dict[str, Any] = {"success": False, "actual_length": 0, "error": "not attempted"}

    for attempt in range(max_retries + 1):
        cur_x, cur_y = x, y
        if element_id:
            info = await inspect_live_ref(page, element_id)
            if info.get("found"):
                if info.get("disabled"):
                    return {
                        "success": False,
                        "actual_length": 0,
                        "strategy": "",
                        "error": f"Input {element_id} is disabled",
                    }
                cur_x, cur_y = float(info["x"]), float(info["y"])
            else:
                rebound = await rebind_stale_ref(page, element_id, kind=kind or "input", label=label)
                if rebound.get("ok"):
                    cur_x, cur_y = float(rebound["x"]), float(rebound["y"])
                    if metrics is not None:
                        metrics.record_stale_ref_recovery()
                    logger.info("♻️ V2 stale input ref rebound %s → (%.0f, %.0f)", element_id, cur_x, cur_y)
                elif kind or label:
                    get_shared_compressor().invalidate()
                    return {
                        "success": False,
                        "stale_ref": True,
                        "actual_length": 0,
                        "strategy": "",
                        "error": (
                            f"STALE_REF: input {element_id} detached "
                            f"({rebound.get('reason', 'missing')}); re-snapshot required"
                        ),
                    }

        last_result = await type_fn(text=text, x=cur_x, y=cur_y, element_id=element_id)
        if last_result.get("success"):
            return last_result

        err = str(last_result.get("error", ""))
        if attempt < max_retries and _is_transient_error(err):
            if metrics is not None:
                metrics.record_action_retry()
            backoff = 0.15 * (2 ** attempt)
            logger.info("🔁 V2 type transient retry %d/%d after %.2fs (%s)",
                        attempt + 1, max_retries, backoff, err[:60])
            await asyncio.sleep(backoff)
            continue
        break

    return last_result
