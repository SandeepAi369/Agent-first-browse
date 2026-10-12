"""THE CRITIC V12 — Task-Aware Progress Detection + Circuit Breaker.

V11 Problem: The old Critic compared screenshot hashes. JPEG artifacts
caused false positives ("page changed" when it didn't), so the agent
clicked "Post" 8+ times in a loop without ever detecting the lack of
real progress.

V12 Solution: Multi-signal progress detection that answers the question:
"Did this action move us CLOSER to the goal?" — not just "Did pixels change?"

Signals used (ranked by reliability):
  1. Element State Change — Did the target element's state mutate?
     (e.g., button became disabled, textbox got filled, modal appeared/disappeared)
  2. A11y Semantic Hash — Did the accessibility tree structure change?
     (deterministic, no JPEG artifacts, catches real DOM mutations)
  3. URL Change — Did we navigate to a new page?
  4. Element Count Delta — Did interactive elements appear/disappear?
  5. Action Repetition — Are we doing the same action on the same target?

Circuit Breaker: After MAX_NO_PROGRESS consecutive actions with no
detected progress, the Critic HALTS the loop immediately and tells
the CEO to replan with a different strategy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

from playwright.async_api import Page

logger = logging.getLogger("orchestrator.critic_v12")


# ═══════════════════════════════════════════════════════════════════════════════
#  Verdict — The Critic's judgment
# ═══════════════════════════════════════════════════════════════════════════════

class Verdict:
    """The Critic's judgment on whether an action achieved progress."""

    __slots__ = (
        "success", "reason", "confidence",
        "url_changed", "semantic_changed", "element_delta",
        "circuit_breaker_triggered", "state_change_score",
    )

    def __init__(
        self,
        success: bool,
        reason: str,
        confidence: float = 1.0,
        url_changed: bool = False,
        semantic_changed: bool = False,
        element_delta: int = 0,
        circuit_breaker_triggered: bool = False,
        state_change_score: float = 0.0,
    ):
        self.success = success
        self.reason = reason
        self.confidence = confidence
        self.url_changed = url_changed
        self.semantic_changed = semantic_changed
        self.element_delta = element_delta
        self.circuit_breaker_triggered = circuit_breaker_triggered
        # V29 Diffing: unified [0,1] "how much did the page state change" signal.
        self.state_change_score = state_change_score

    def __repr__(self) -> str:
        icon = "✅" if self.success else ("🛑" if self.circuit_breaker_triggered else "❌")
        return f"Verdict({icon} {self.reason} [{self.confidence:.0%}])"


# ═══════════════════════════════════════════════════════════════════════════════
#  Action Fingerprint — For repetition detection
# ═══════════════════════════════════════════════════════════════════════════════

class ActionFingerprint:
    """Fingerprint of a single action for repetition detection."""
    __slots__ = ("action", "target_ref", "target_name", "url", "timestamp")

    def __init__(self, action: str, target_ref: str = "", target_name: str = "", url: str = ""):
        self.action = action
        self.target_ref = target_ref
        self.target_name = target_name
        self.url = url
        self.timestamp = time.monotonic()

    @property
    def identity(self) -> str:
        """Unique identity string for dedup."""
        return f"{self.action}|{self.target_ref}|{self.target_name}|{self.url}"


# ═══════════════════════════════════════════════════════════════════════════════
#  PageState — Snapshot of the page for comparison
# ═══════════════════════════════════════════════════════════════════════════════

class PageState:
    """Captured state of a page at a point in time."""
    __slots__ = (
        "url", "semantic_hash", "element_count",
        "element_refs", "element_states", "title", "timestamp",
    )

    def __init__(
        self,
        url: str = "",
        semantic_hash: int = 0,
        element_count: int = 0,
        element_refs: frozenset[str] | None = None,
        element_states: dict[str, str] | None = None,
        title: str = "",
    ):
        self.url = url
        self.semantic_hash = semantic_hash
        self.element_count = element_count
        self.element_refs = element_refs or frozenset()
        self.element_states = element_states or {}
        self.title = title
        self.timestamp = time.monotonic()


# ═══════════════════════════════════════════════════════════════════════════════
#  CriticV12 — The Brain's Self-Correction Engine
# ═══════════════════════════════════════════════════════════════════════════════

class CriticV12:
    """Task-aware progress detection with automatic circuit breaking.

    Usage:
        critic = CriticV12(page)
        await critic.snapshot_before(a11y_data)
        # ... execute action ...
        verdict = await critic.evaluate(action, target_ref, a11y_data_after)
    """

    MAX_NO_PROGRESS = 2       # Circuit breaker threshold
    MAX_ACTION_HISTORY = 10   # Rolling window for repetition detection

    def __init__(self, page: Page) -> None:
        self._page = page
        self._prev_state: PageState | None = None
        self._prev_signals: dict | None = None   # V29 Diffing: page-signal vector
        self._no_progress_streak: int = 0
        self._action_history: deque[ActionFingerprint] = deque(maxlen=self.MAX_ACTION_HISTORY)
        self._total_actions: int = 0
        self._total_progress: int = 0

    # ── Pre-Action Snapshot ───────────────────────────────────────────────

    async def snapshot_before(self, a11y_data: dict | None = None) -> None:
        """Capture pre-action state. Call this BEFORE executing any action.

        Args:
            a11y_data: The result dict from a11y_parser.extract().
                       If None, only URL is captured.
        """
        url = ""
        try:
            url = self._page.url
        except Exception:
            pass

        title = ""
        try:
            title = await self._page.title()
        except Exception:
            pass

        semantic_hash = 0
        element_count = 0
        element_refs: frozenset[str] = frozenset()
        element_states: dict[str, str] = {}

        if a11y_data:
            semantic_hash = a11y_data.get("semantic_hash", 0)
            element_count = a11y_data.get("element_count", 0)
            elements = a11y_data.get("elements", [])
            element_refs = frozenset(e.get("ref", "") for e in elements if e.get("ref"))
            # Capture per-element state for fine-grained change detection
            for e in elements:
                ref = e.get("ref", "")
                if ref:
                    props = e.get("properties", {})
                    state_str = f"{e.get('role','')}|{e.get('name','')}|{sorted(props.items())}"
                    element_states[ref] = state_str

        self._prev_state = PageState(
            url=url,
            semantic_hash=semantic_hash,
            element_count=element_count,
            element_refs=element_refs,
            element_states=element_states,
            title=title,
        )

        # V29 Diffing: capture the pre-action page-signal vector (cheap, gated).
        self._prev_signals = await self._capture_signals()

    async def _capture_signals(self) -> dict | None:
        """One cheap page.evaluate → the ~8-number universal page-signal vector.
        Returns None when diffing is disabled or the eval fails (never raises)."""
        try:
            from feature_flags import diffing_enabled
            if not diffing_enabled():
                return None
            from dom_diff import PAGE_SIGNAL_JS
            return await asyncio.wait_for(self._page.evaluate(PAGE_SIGNAL_JS), timeout=2.0)
        except Exception as e:  # noqa: BLE001 — diffing never breaks the verdict
            logger.debug("page-signal capture skipped: %s", e)
            return None

    # ── Post-Action Evaluation ────────────────────────────────────────────

    async def evaluate(
        self,
        action: str,
        target_ref: str = "",
        target_name: str = "",
        a11y_data_after: dict | None = None,
        skill_success: bool = True,
        skill_error: str = "",
    ) -> Verdict:
        """Evaluate whether an action achieved real task progress.

        This is the core intelligence of V12. Instead of just checking
        "did the page change?", it checks "did we make PROGRESS?"

        Args:
            action: The action type (click, type, goto, scroll, wait, done)
            target_ref: The RefID of the target element (e.g., "e5")
            target_name: Human-readable name (e.g., "Post button")
            a11y_data_after: A11y extraction result AFTER the action
            skill_success: Whether the skill/executor reported success
            skill_error: Error message from the skill (if any)

        Returns:
            Verdict with success, reason, and circuit_breaker status.
        """
        self._total_actions += 1

        # ── Signal 0: Skill-level failure ──
        if not skill_success:
            return self._make_verdict(
                success=False,
                reason=f"Skill reported failure: {skill_error or 'unknown'}",
                confidence=0.95,
                is_progress=False,
            )

        # ── Capture current state ──
        current_url = ""
        try:
            current_url = self._page.url
        except Exception:
            pass

        current_hash = 0
        current_count = 0
        current_refs: frozenset[str] = frozenset()
        current_states: dict[str, str] = {}

        if a11y_data_after:
            current_hash = a11y_data_after.get("semantic_hash", 0)
            current_count = a11y_data_after.get("element_count", 0)
            elements = a11y_data_after.get("elements", [])
            current_refs = frozenset(e.get("ref", "") for e in elements if e.get("ref"))
            for e in elements:
                ref = e.get("ref", "")
                if ref:
                    props = e.get("properties", {})
                    state_str = f"{e.get('role','')}|{e.get('name','')}|{sorted(props.items())}"
                    current_states[ref] = state_str

        prev = self._prev_state or PageState()

        # ── Signal 1: URL Change ──
        url_changed = current_url != prev.url and current_url != ""

        # ── Signal 2: Semantic Hash Change (A11y tree structure) ──
        semantic_changed = (
            current_hash != prev.semantic_hash
            and current_hash != 0
            and prev.semantic_hash != 0
        )

        # ── Signal 3: Element Count Delta ──
        element_delta = current_count - prev.element_count

        # ── Signal 4: Target Element State Change ──
        target_state_changed = False
        if target_ref and target_ref in prev.element_states:
            old_state = prev.element_states.get(target_ref, "")
            new_state = current_states.get(target_ref, "")
            if old_state != new_state:
                target_state_changed = True
            # Element disappeared = likely submitted/closed (strong progress signal)
            if target_ref in prev.element_refs and target_ref not in current_refs:
                target_state_changed = True

        # ── Signal 5: New Elements Appeared ──
        new_elements = current_refs - prev.element_refs
        disappeared_elements = prev.element_refs - current_refs

        # ── Signal 6: Action Repetition Detection ──
        fingerprint = ActionFingerprint(
            action=action,
            target_ref=target_ref,
            target_name=target_name,
            url=current_url,
        )
        repetition_count = sum(
            1 for fp in self._action_history
            if fp.identity == fingerprint.identity
        )
        self._action_history.append(fingerprint)

        # ═══════════════════════════════════════════════════════════════════
        #  PROGRESS DETERMINATION ENGINE
        #  Combines all signals to determine if real progress was made.
        # ═══════════════════════════════════════════════════════════════════

        progress_signals: list[str] = []

        # URL change is strong progress for any action
        if url_changed:
            progress_signals.append(f"URL changed → {current_url}")

        # Semantic change = the page structure genuinely mutated
        if semantic_changed:
            progress_signals.append("Page structure changed (A11y hash)")

        # Target element changed state or disappeared
        if target_state_changed:
            progress_signals.append(f"Target '{target_ref}' state changed")

        # New elements appeared (e.g., compose modal opened, results loaded)
        if len(new_elements) >= 2:
            progress_signals.append(f"{len(new_elements)} new elements appeared")

        # Elements disappeared (e.g., modal closed, form submitted)
        if len(disappeared_elements) >= 2:
            progress_signals.append(f"{len(disappeared_elements)} elements disappeared")

        # Significant element count change
        if abs(element_delta) >= 3:
            progress_signals.append(f"Element count changed by {element_delta:+d}")

        # ── Action-specific logic ──
        if action == "goto":
            if url_changed:
                return self._make_verdict(
                    success=True,
                    reason=f"Navigation confirmed: {current_url}",
                    confidence=0.95,
                    is_progress=True,
                    url_changed=True,
                    semantic_changed=semantic_changed,
                    element_delta=element_delta,
                )
            else:
                return self._make_verdict(
                    success=False,
                    reason="URL did not change after navigation",
                    confidence=0.8,
                    is_progress=False,
                )

        if action == "done":
            return self._make_verdict(
                success=True,
                reason="Task marked as done",
                confidence=1.0,
                is_progress=True,
            )

        if action == "wait":
            # Wait always "succeeds" but doesn't count as progress
            return self._make_verdict(
                success=True,
                reason="Wait completed",
                confidence=0.9,
                is_progress=semantic_changed or url_changed,
                semantic_changed=semantic_changed,
            )

        # ── V29 Diffing: fold subtle overlay/panel/focus changes into the signals.
        #    A click that opens a small overlay using existing nodes (so element_delta
        #    ≈ 0) used to read as "no progress" → stagnation loop. The page-signal
        #    vector catches it. The diff stays in code; only this one phrase reaches
        #    the LLM. Computes the unified state_change_score either way.
        sc_score = 0.0
        try:
            from dom_diff import signal_vector_diff, state_change_score, progress_phrase
            post_signals = await self._capture_signals()
            vector = signal_vector_diff(self._prev_signals, post_signals)
            if vector.get("meaningful") and not progress_signals:
                phrase = progress_phrase(vector)
                if phrase:
                    progress_signals.append(phrase)
            sc_score = state_change_score(
                url_changed=url_changed, semantic_changed=semantic_changed,
                element_delta=element_delta, new_count=len(new_elements),
                disappeared_count=len(disappeared_elements), vector=vector)
        except Exception as e:  # noqa: BLE001 — diffing never breaks the verdict
            logger.debug("diffing fold skipped: %s", e)

        # ── For click/type/scroll: require actual progress signals ──
        made_progress = len(progress_signals) > 0

        if made_progress:
            reason = f"Progress detected: {'; '.join(progress_signals[:3])}"
            return self._make_verdict(
                success=True,
                reason=reason,
                confidence=min(0.6 + 0.1 * len(progress_signals), 0.98),
                is_progress=True,
                url_changed=url_changed,
                semantic_changed=semantic_changed,
                element_delta=element_delta,
                state_change_score=sc_score,
            )

        # ── NO PROGRESS DETECTED ──
        # The action executed but nothing meaningful changed.
        # This is the critical path that prevents infinite loops.

        if repetition_count >= 1:
            reason = (
                f"No progress: '{action}' on '{target_name or target_ref}' "
                f"repeated {repetition_count + 1}x with no effect"
            )
        else:
            reason = (
                f"No progress: '{action}' on '{target_name or target_ref}' "
                f"had no visible effect on page state"
            )

        return self._make_verdict(
            success=False,
            reason=reason,
            confidence=0.85,
            is_progress=False,
            url_changed=False,
            semantic_changed=False,
            element_delta=element_delta,
            state_change_score=sc_score,
        )

    # ── Internal: Verdict Construction + Circuit Breaker ──────────────────

    def _make_verdict(
        self,
        success: bool,
        reason: str,
        confidence: float,
        is_progress: bool,
        url_changed: bool = False,
        semantic_changed: bool = False,
        element_delta: int = 0,
        state_change_score: float = 0.0,
    ) -> Verdict:
        """Build a Verdict and update the circuit breaker state."""

        if is_progress:
            self._no_progress_streak = 0
            self._total_progress += 1
        else:
            self._no_progress_streak += 1

        circuit_breaker = False

        if self._no_progress_streak >= self.MAX_NO_PROGRESS:
            circuit_breaker = True
            reason = (
                f"🛑 CIRCUIT BREAKER: {self._no_progress_streak} consecutive actions "
                f"with no detected progress. Last: {reason}. "
                f"The CEO must replan with a DIFFERENT strategy."
            )
            confidence = 0.95
            success = False
            logger.warning(reason)

        return Verdict(
            success=success,
            reason=reason,
            confidence=confidence,
            url_changed=url_changed,
            semantic_changed=semantic_changed,
            element_delta=element_delta,
            circuit_breaker_triggered=circuit_breaker,
            state_change_score=state_change_score,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def no_progress_streak(self) -> int:
        return self._no_progress_streak

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_actions": self._total_actions,
            "total_progress": self._total_progress,
            "no_progress_streak": self._no_progress_streak,
            "progress_rate": (
                self._total_progress / self._total_actions
                if self._total_actions > 0 else 1.0
            ),
            "action_history_size": len(self._action_history),
        }

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker after a successful replan."""
        self._no_progress_streak = 0
        logger.info("Circuit breaker reset after replan")

    def reset_for_task(self) -> None:
        """Full reset of per-task state (V18 clean handoff).

        Called at task finalize so a reused CriticV12 instance never carries a
        previous task's progress streak, action history, or snapshot into a new
        one.
        """
        self._prev_state = None
        self._no_progress_streak = 0
        self._action_history.clear()
        self._total_actions = 0
        self._total_progress = 0
        logger.info("CriticV12 reset for new task")
