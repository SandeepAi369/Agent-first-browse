"""Post-action verification with delayed React reversion detection.

This module provides DOM snapshot diffing and delayed recheck capabilities
to verify that browser actions (click, type, etc.) actually took effect.
It specifically catches React/Vue controlled component reversions where
the framework's reconciler overwrites DOM changes after a short delay.

Wraps around the existing CriticV12 and adds:
- Pre/post DOM snapshot comparison
- Structural fingerprint diffing
- Delayed field value rechecking (catches 100-300ms React reversions)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page

try:
    from app.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:
        """Fallback logger factory."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        return logger


logger = get_logger(__name__)

_DEFAULT_EVAL_TIMEOUT_S = 2.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DOMSnapshot:
    """Immutable snapshot of the DOM state at a point in time.

    Attributes:
        url: The page URL when the snapshot was taken.
        title: The document title.
        element_count: Number of interactive elements found.
        semantic_hash: Hash of the structural fingerprint (element states).
        element_states: Mapping of element ref -> state fingerprint string.
        timestamp: Unix timestamp (seconds) when the snapshot was captured.
        field_values: Mapping of element id -> current text/value for
            input, textarea, and contenteditable elements.
    """

    url: str
    title: str
    element_count: int
    semantic_hash: int
    element_states: dict[str, str]
    timestamp: float
    field_values: dict[str, str]


@dataclass
class VerificationResult:
    """Outcome of a post-action verification check.

    Attributes:
        verified: ``True`` if the action appears to have succeeded.
        effect_type: Category of the observed effect.  One of
            ``'navigation'``, ``'state_change'``, ``'element_appeared'``,
            ``'element_disappeared'``, ``'content_change'``,
            ``'no_effect'``, or ``'reverted'``.
        diagnosis: Human-readable explanation of what was detected.
        suggested_recovery: Recommended recovery action if the effect was
            unexpected or missing.
        confidence: Confidence score in [0.0, 1.0].
        dom_diff: Summary dict of what changed between snapshots.
    """

    verified: bool
    effect_type: str
    diagnosis: str
    suggested_recovery: str
    confidence: float
    dom_diff: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JavaScript snippet executed inside the page
# ---------------------------------------------------------------------------

_SNAPSHOT_JS = """
() => {
    const inputs = document.querySelectorAll('input, textarea, [contenteditable]');
    const fieldValues = {};
    const elementStates = {};
    inputs.forEach((el, i) => {
        const id = el.getAttribute('data-element-id') || el.id || `field_${i}`;
        if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
            fieldValues[id] = el.value || '';
        } else {
            fieldValues[id] = el.textContent || '';
        }
    });
    const interactives = document.querySelectorAll(
        'a, button, input, textarea, select, [role="button"], [role="link"]'
    );
    interactives.forEach((el, i) => {
        const ref = el.getAttribute('data-ref') || `e${i}`;
        elementStates[ref] = [
            el.tagName,
            (el.textContent || '').trim().slice(0, 30),
            el.disabled,
            el.value || ''
        ].join('|');
    });
    return {
        url: location.href,
        title: document.title,
        elementCount: interactives.length,
        fieldValues,
        elementStates
    };
}
""".strip()

_READ_FIELD_JS = """
(selector) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
        return el.value || '';
    }
    return el.textContent || '';
}
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_semantic_hash(element_states: dict[str, str]) -> int:
    """Deterministic hash of the structural fingerprint.

    The hash is order-independent: keys are sorted before hashing so that
    DOM enumeration order differences don't produce false positives.
    """
    parts = sorted(f"{k}={v}" for k, v in element_states.items())
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:16], 16)


def _build_css_selector(target_ref: str) -> str:
    """Best-effort CSS selector from a *target_ref* string.

    ``target_ref`` may be:
    - A CSS selector already (starts with ``#``, ``.``, ``[``, or a tag name)
    - A bare ``data-ref`` value  →  ``[data-ref="<value>"]``
    - A bare ``id`` value        →  ``#<value>``

    Falls back to ``[data-ref="<target_ref>"]`` when ambiguous.
    """
    if not target_ref:
        return ""
    ref = target_ref.strip()
    # Already looks like a selector
    if ref.startswith(("#", ".", "[")) or ref.lower().startswith(("input", "textarea", "select", "button")):
        return ref
    # Looks like a data-ref token (e.g. "e4")
    return f'[data-ref="{ref}"]'


# ---------------------------------------------------------------------------
# ActionVerifier
# ---------------------------------------------------------------------------


class ActionVerifier:
    """Verifies that a browser action had the intended DOM effect.

    Usage::

        verifier = ActionVerifier(page)
        await verifier.snapshot_before()
        # … perform the action …
        result = await verifier.verify_after("type", target_ref="#email", expected_text="a@b.com")
        if not result.verified:
            logger.warning("Action failed: %s", result.diagnosis)

    For React-managed inputs, follow up with :meth:`delayed_recheck`::

        still_ok = await verifier.delayed_recheck("#email", "a@b.com", delay_ms=300)
    """

    def __init__(self, page: Page) -> None:
        self._page = page
        self._before: DOMSnapshot | None = None

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    async def snapshot(self) -> DOMSnapshot:
        """Capture a full DOM snapshot via a single ``page.evaluate()`` call.

        Returns:
            A :class:`DOMSnapshot` representing the current page state.

        The call is wrapped in :func:`asyncio.wait_for` with a 2-second
        timeout so a hung page never blocks the agent loop.
        """
        try:
            raw: dict[str, Any] = await asyncio.wait_for(
                self._page.evaluate(_SNAPSHOT_JS),
                timeout=_DEFAULT_EVAL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("snapshot: page.evaluate timed out after %.1fs", _DEFAULT_EVAL_TIMEOUT_S)
            return DOMSnapshot(
                url="",
                title="",
                element_count=0,
                semantic_hash=0,
                element_states={},
                timestamp=time.time(),
                field_values={},
            )
        except Exception:
            logger.exception("snapshot: page.evaluate raised an unexpected error")
            return DOMSnapshot(
                url="",
                title="",
                element_count=0,
                semantic_hash=0,
                element_states={},
                timestamp=time.time(),
                field_values={},
            )

        element_states: dict[str, str] = raw.get("elementStates", {})
        field_values: dict[str, str] = raw.get("fieldValues", {})
        semantic_hash = _compute_semantic_hash(element_states)

        snap = DOMSnapshot(
            url=raw.get("url", ""),
            title=raw.get("title", ""),
            element_count=raw.get("elementCount", 0),
            semantic_hash=semantic_hash,
            element_states=element_states,
            timestamp=time.time(),
            field_values=field_values,
        )
        logger.debug(
            "snapshot: url=%s elements=%d fields=%d hash=%x",
            snap.url,
            snap.element_count,
            len(snap.field_values),
            snap.semantic_hash,
        )
        return snap

    async def snapshot_before(self) -> None:
        """Capture pre-action state.

        Must be called *before* the action is performed.  The captured
        snapshot is stored internally and used by :meth:`verify_after`.
        """
        self._before = await self.snapshot()
        logger.debug("snapshot_before: captured (url=%s)", self._before.url)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    async def verify_after(
        self,
        action: str,
        target_ref: str = "",
        expected_text: str | None = None,
    ) -> VerificationResult:
        """Verify that an action had the expected effect.

        Compares the pre-action snapshot (from :meth:`snapshot_before`) with
        a fresh post-action snapshot.

        Args:
            action: The action that was performed, e.g. ``"click"``,
                ``"type"``, ``"select"``.
            target_ref: An identifier for the target element (CSS selector,
                ``data-ref`` value, or element ``id``).
            expected_text: For ``type`` actions, the text that should now
                appear in the target field.

        Returns:
            A :class:`VerificationResult` describing what happened.
        """
        if self._before is None:
            logger.warning("verify_after: no pre-action snapshot — calling snapshot_before implicitly")
            return VerificationResult(
                verified=False,
                effect_type="no_effect",
                diagnosis="No pre-action snapshot was captured; cannot compare.",
                suggested_recovery="Call snapshot_before() before performing the action.",
                confidence=0.0,
                dom_diff={},
            )

        after = await self.snapshot()
        diff = self._compute_diff(self._before, after)

        action_lower = action.strip().lower()

        # --- Type action ------------------------------------------------
        if action_lower in ("type", "fill", "insert_text", "inserttext"):
            return self._verify_type_action(
                before=self._before,
                after=after,
                diff=diff,
                target_ref=target_ref,
                expected_text=expected_text,
            )

        # --- Click action -----------------------------------------------
        if action_lower == "click":
            return self._verify_click_action(
                before=self._before,
                after=after,
                diff=diff,
                target_ref=target_ref,
            )

        # --- Generic / other actions ------------------------------------
        return self._verify_generic_action(
            before=self._before,
            after=after,
            diff=diff,
            action=action_lower,
        )

    # ------------------------------------------------------------------
    # Delayed recheck  (React / Vue reversion detection)
    # ------------------------------------------------------------------

    async def delayed_recheck(
        self,
        target_ref: str,
        expected_text: str,
        delay_ms: int = 300,
    ) -> bool:
        """Wait *delay_ms* then recheck if the field still holds *expected_text*.

        This catches React/Vue controlled-component reversions where:

        1. CDP ``insertText`` sets the DOM value.
        2. Immediate verification reads it back → **PASS**.
        3. The framework's ``setState`` reconciler overwrites it 100-300 ms
           later → **FAIL**.

        Args:
            target_ref: CSS selector or ``data-ref`` identifying the field.
            expected_text: The text that should still be present.
            delay_ms: Milliseconds to wait before rechecking (default 300).

        Returns:
            ``True`` if the text is still intact; ``False`` if it was reverted.
        """
        delay_s = max(delay_ms, 0) / 1000.0
        logger.debug(
            "delayed_recheck: waiting %.0f ms before rechecking target=%s",
            delay_ms,
            target_ref,
        )
        await asyncio.sleep(delay_s)

        selector = _build_css_selector(target_ref)
        if not selector:
            logger.warning("delayed_recheck: empty target_ref — cannot build selector")
            return False

        try:
            current_value: str | None = await asyncio.wait_for(
                self._page.evaluate(_READ_FIELD_JS, selector),
                timeout=_DEFAULT_EVAL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("delayed_recheck: page.evaluate timed out")
            return False
        except Exception:
            logger.exception("delayed_recheck: page.evaluate raised an unexpected error")
            return False

        if current_value is None:
            logger.warning(
                "delayed_recheck: element not found for selector '%s'",
                selector,
            )
            return False

        match = current_value == expected_text
        if not match:
            logger.warning(
                "delayed_recheck: REVERTED — expected %r but found %r (selector=%s)",
                expected_text,
                current_value,
                selector,
            )
        else:
            logger.debug("delayed_recheck: text intact for selector=%s", selector)

        return match

    # ------------------------------------------------------------------
    # Internal diff / classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_diff(before: DOMSnapshot, after: DOMSnapshot) -> dict[str, Any]:
        """Compute a structured diff between two DOM snapshots.

        Returns a dict with keys:
        - ``url_changed``: bool
        - ``title_changed``: bool
        - ``element_count_delta``: int (positive = elements added)
        - ``hash_changed``: bool
        - ``added_elements``: list of refs present only in *after*
        - ``removed_elements``: list of refs present only in *before*
        - ``changed_elements``: list of refs whose fingerprint changed
        - ``field_changes``: dict of field_id -> (old, new)
        """
        before_refs = set(before.element_states.keys())
        after_refs = set(after.element_states.keys())

        added = sorted(after_refs - before_refs)
        removed = sorted(before_refs - after_refs)
        changed: list[str] = []
        for ref in sorted(before_refs & after_refs):
            if before.element_states[ref] != after.element_states[ref]:
                changed.append(ref)

        field_changes: dict[str, tuple[str, str]] = {}
        all_field_ids = set(before.field_values.keys()) | set(after.field_values.keys())
        for fid in sorted(all_field_ids):
            old_val = before.field_values.get(fid, "")
            new_val = after.field_values.get(fid, "")
            if old_val != new_val:
                field_changes[fid] = (old_val, new_val)

        return {
            "url_changed": before.url != after.url,
            "title_changed": before.title != after.title,
            "element_count_delta": after.element_count - before.element_count,
            "hash_changed": before.semantic_hash != after.semantic_hash,
            "added_elements": added,
            "removed_elements": removed,
            "changed_elements": changed,
            "field_changes": field_changes,
        }

    # -- Type action -----------------------------------------------------

    @staticmethod
    def _verify_type_action(
        *,
        before: DOMSnapshot,
        after: DOMSnapshot,
        diff: dict[str, Any],
        target_ref: str,
        expected_text: str | None,
    ) -> VerificationResult:
        """Classify the outcome of a type / fill action."""
        field_changes: dict[str, tuple[str, str]] = diff.get("field_changes", {})

        # If we have an expected text, look for it in field values
        if expected_text is not None:
            # Check all fields for the expected text
            for fid, new_val in after.field_values.items():
                if expected_text in new_val:
                    return VerificationResult(
                        verified=True,
                        effect_type="content_change",
                        diagnosis=f"Field '{fid}' contains expected text.",
                        suggested_recovery="",
                        confidence=0.95,
                        dom_diff=diff,
                    )

            # Check if any field changed to contain the expected text
            for fid, (old, new) in field_changes.items():
                if expected_text in new:
                    return VerificationResult(
                        verified=True,
                        effect_type="content_change",
                        diagnosis=f"Field '{fid}' changed and contains expected text.",
                        suggested_recovery="",
                        confidence=0.95,
                        dom_diff=diff,
                    )

            # Expected text not found anywhere
            if field_changes:
                # Something changed, but not what we expected
                return VerificationResult(
                    verified=False,
                    effect_type="content_change",
                    diagnosis=(
                        f"Fields changed but expected text not found. "
                        f"Changed fields: {list(field_changes.keys())}"
                    ),
                    suggested_recovery="Re-focus the target element and retype the text.",
                    confidence=0.7,
                    dom_diff=diff,
                )

            # Nothing changed at all
            return VerificationResult(
                verified=False,
                effect_type="no_effect",
                diagnosis="No field values changed after type action; input may not have been focused.",
                suggested_recovery="Click the target field to focus it, then retry the type action.",
                confidence=0.85,
                dom_diff=diff,
            )

        # No expected_text — just check if *any* field changed
        if field_changes:
            return VerificationResult(
                verified=True,
                effect_type="content_change",
                diagnosis=f"{len(field_changes)} field(s) changed.",
                suggested_recovery="",
                confidence=0.75,
                dom_diff=diff,
            )

        return VerificationResult(
            verified=False,
            effect_type="no_effect",
            diagnosis="No field changes detected after type action.",
            suggested_recovery="Ensure the target element is focused and editable.",
            confidence=0.8,
            dom_diff=diff,
        )

    # -- Click action ----------------------------------------------------

    @staticmethod
    def _verify_click_action(
        *,
        before: DOMSnapshot,
        after: DOMSnapshot,
        diff: dict[str, Any],
        target_ref: str,
    ) -> VerificationResult:
        """Classify the outcome of a click action."""

        # Navigation
        if diff["url_changed"]:
            return VerificationResult(
                verified=True,
                effect_type="navigation",
                diagnosis=f"Navigation detected: {before.url} → {after.url}",
                suggested_recovery="",
                confidence=0.95,
                dom_diff=diff,
            )

        added = diff.get("added_elements", [])
        removed = diff.get("removed_elements", [])
        delta = diff["element_count_delta"]

        # New elements appeared (e.g. dropdown, modal, accordion)
        if delta > 0 and added:
            return VerificationResult(
                verified=True,
                effect_type="element_appeared",
                diagnosis=f"{len(added)} new element(s) appeared (delta={delta}).",
                suggested_recovery="",
                confidence=0.85,
                dom_diff=diff,
            )

        # Elements disappeared (e.g. closing a modal)
        if delta < 0 and removed:
            return VerificationResult(
                verified=True,
                effect_type="element_disappeared",
                diagnosis=f"{len(removed)} element(s) disappeared (delta={delta}).",
                suggested_recovery="",
                confidence=0.85,
                dom_diff=diff,
            )

        # Structural hash changed (attribute toggles, class changes, etc.)
        if diff["hash_changed"]:
            changed = diff.get("changed_elements", [])
            return VerificationResult(
                verified=True,
                effect_type="state_change",
                diagnosis=f"DOM structure changed ({len(changed)} element(s) mutated).",
                suggested_recovery="",
                confidence=0.7,
                dom_diff=diff,
            )

        # Title changed (e.g. SPA route change without URL update)
        if diff["title_changed"]:
            return VerificationResult(
                verified=True,
                effect_type="state_change",
                diagnosis=f"Page title changed: '{before.title}' → '{after.title}'.",
                suggested_recovery="",
                confidence=0.65,
                dom_diff=diff,
            )

        # Nothing observable changed
        return VerificationResult(
            verified=False,
            effect_type="no_effect",
            diagnosis="No observable DOM change after click.",
            suggested_recovery="Retry the click, or try a different selector / coordinates.",
            confidence=0.8,
            dom_diff=diff,
        )

    # -- Generic action --------------------------------------------------

    @staticmethod
    def _verify_generic_action(
        *,
        before: DOMSnapshot,
        after: DOMSnapshot,
        diff: dict[str, Any],
        action: str,
    ) -> VerificationResult:
        """Classify the outcome of any non-click, non-type action."""

        any_change = (
            diff["url_changed"]
            or diff["title_changed"]
            or diff["hash_changed"]
            or diff["element_count_delta"] != 0
            or diff.get("field_changes")
        )

        if diff["url_changed"]:
            effect = "navigation"
        elif diff["element_count_delta"] > 0:
            effect = "element_appeared"
        elif diff["element_count_delta"] < 0:
            effect = "element_disappeared"
        elif diff["hash_changed"] or diff.get("changed_elements"):
            effect = "state_change"
        elif diff.get("field_changes"):
            effect = "content_change"
        else:
            effect = "no_effect"

        if any_change:
            return VerificationResult(
                verified=True,
                effect_type=effect,
                diagnosis=f"Action '{action}' produced a DOM change (effect={effect}).",
                suggested_recovery="",
                confidence=0.7,
                dom_diff=diff,
            )

        return VerificationResult(
            verified=False,
            effect_type="no_effect",
            diagnosis=f"No DOM change detected after '{action}' action.",
            suggested_recovery=f"Retry the '{action}' action or try an alternative approach.",
            confidence=0.75,
            dom_diff=diff,
        )
