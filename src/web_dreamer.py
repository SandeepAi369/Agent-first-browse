"""WebDreamer — LLM-as-World-Model Simulate-Before-Acting Planning Engine.

Implements Algorithm 1 from "Is Your LLM Secretly a World Model of the Internet?"
(Gu et al., arXiv:2411.06559, TMLR 2025).

Instead of committing to the first action the LLM suggests, WebDreamer:
  1. Generates k candidate actions for the current state
  2. For each candidate, uses the LLM to SIMULATE what would happen
  3. Scores each simulated outcome on a three-bucket scale
  4. Executes argmax(scores) — the action most likely to make progress

This gives the look-ahead benefit of tree search without executing any
irreversible actions during planning. Only the winning action is executed
on the live browser.

Key design decisions:
  - Horizon H=1 (single-step look-ahead) — H=2,3 degrade due to hallucination
  - k=3 candidates by default (configurable)
  - Three-bucket scoring: COMPLETE (1.0), ON_TRACK (0.5), WRONG (0.0)
  - Averaged over 2 simulation runs per candidate (reduces variance)
  - Complexity gate: only invoked for ambiguous/risky steps (not every step)

Integration:
  Sits between the DOM parser and the action executor in the main loop.
  Uses the same LLM failover chain as the main agent.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from app.logger import get_logger
    logger = get_logger("web_dreamer")
except ImportError:
    import logging
    logger = logging.getLogger("web_dreamer")


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateAction:
    """A proposed action with its reasoning."""
    action_type: str
    element_id: str | None = None
    text: str | None = None
    url: str | None = None
    x: float | None = None
    y: float | None = None
    reasoning: str = ""
    
    def describe(self) -> str:
        """Human-readable description of this action."""
        parts = [self.action_type]
        if self.element_id:
            parts.append(f"on [{self.element_id}]")
        if self.text:
            parts.append(f"text='{self.text[:40]}'")
        if self.url:
            parts.append(f"url='{self.url[:60]}'")
        return " ".join(parts)


@dataclass
class SimulatedOutcome:
    """Predicted state change after executing a candidate action."""
    predicted_changes: str  # Natural language description of what would change
    new_elements: str       # Elements that would appear
    disappeared_elements: str  # Elements that would disappear
    url_change: str         # Expected URL change (if any)
    error_state: str        # Any error conditions predicted


@dataclass
class CandidateEvaluation:
    """Full evaluation of a candidate action."""
    candidate: CandidateAction
    simulated_outcome: SimulatedOutcome | None = None
    score: float = 0.0      # 0.0 (WRONG), 0.5 (ON_TRACK), 1.0 (COMPLETE)
    confidence: float = 0.0  # Average confidence across simulation runs
    score_label: str = ""    # "COMPLETE" | "ON_TRACK" | "WRONG"


@dataclass
class DreamerResult:
    """Result of WebDreamer planning."""
    best_action: CandidateAction
    best_score: float
    all_evaluations: list[CandidateEvaluation]
    planning_time_ms: float
    candidates_generated: int
    simulations_run: int


# ═══════════════════════════════════════════════════════════════════════════════
#  Pydantic Schemas for Structured LLM Output
# ═══════════════════════════════════════════════════════════════════════════════

class Candidate(BaseModel):
    """One candidate action proposed by the planner.

    V17.0: typed nested model (was a free-form dict) — strict-mode providers
    like Groq require `additionalProperties: false` on every nested object,
    which Pydantic emits via extra='forbid'. The untyped dict caused 400s.
    """
    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(
        description="One of: goto, click, type, scroll, press_enter, wait, done."
    )
    element_id: str | None = Field(
        default=None, description="The element ID from the page structure (e.g., 'e5')."
    )
    text: str | None = Field(
        default=None, description="The full text to type (for type actions)."
    )
    url: str | None = Field(
        default=None, description="URL to navigate to (for goto actions)."
    )
    reasoning: str = Field(
        default="", description="Why this action might work, 1 sentence."
    )


class CandidateSet(BaseModel):
    """LLM output: k candidate actions for the current state."""
    model_config = ConfigDict(extra="forbid")

    candidates: list[Candidate] = Field(
        description=(
            "A list of 3 distinct candidate actions. "
            "Make candidates genuinely DIFFERENT strategies, not variations of one."
        )
    )


class WorldModelPrediction(BaseModel):
    """LLM output: predicted state changes after an action."""
    model_config = ConfigDict(extra="forbid")

    predicted_changes: str = Field(
        description=(
            "Describe what will change on the webpage after this action. "
            "Be specific: new elements appearing, existing elements disappearing, "
            "content changes, URL changes, or error states. "
            "If the action is a click on a button, predict what happens next."
        )
    )
    url_change: str = Field(
        default="",
        description="The expected new URL after this action, or empty if URL won't change."
    )
    error_prediction: str = Field(
        default="",
        description="Any error conditions you predict (e.g., 'button is disabled', 'element not found')."
    )


class ValueJudgment(BaseModel):
    """LLM output: score for a simulated outcome."""
    model_config = ConfigDict(extra="forbid")

    verdict: str = Field(
        description=(
            "Rate whether the predicted state after the action moves toward the goal. "
            "Must be exactly one of: COMPLETE, ON_TRACK, WRONG. "
            "COMPLETE = the goal appears fully achieved. "
            "ON_TRACK = meaningful progress toward the goal. "
            "WRONG = no progress or moves away from the goal."
        )
    )
    explanation: str = Field(
        description="Brief explanation for your verdict (1 sentence)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompts
# ═══════════════════════════════════════════════════════════════════════════════

CANDIDATE_GENERATION_PROMPT = """You are an expert web automation planner. Given the current page state and objective, propose exactly 3 DISTINCT candidate actions.

Each candidate should represent a DIFFERENT strategy:
- Candidate 1: The most obvious/direct action
- Candidate 2: An alternative approach (different element or strategy)
- Candidate 3: A cautious/exploratory action (scroll, wait, or navigate differently)

Rules:
- Each candidate must be a single atomic action (click, type, goto, scroll, press_enter, wait, or done)
- Use element_id (e.g., 'e5') when available for click/type actions
- For type actions, include the full text to type
- Make candidates genuinely DIFFERENT — not 3 variations of clicking the same button
- If the goal appears already achieved, one candidate should be 'done'"""


WORLD_MODEL_PROMPT = """You are a world model for web browser interactions. You predict what will happen when an action is performed on a webpage.

Given:
- The current page state (DOM structure, visible elements)
- A proposed action (click, type, navigate, etc.)

Predict ONLY the changes that will occur:
- New elements appearing (modals, dropdowns, new page content)
- Existing elements disappearing (closed modals, navigated away)
- Content changes (text field filled, button state changed)
- URL changes (page navigation)
- Error conditions (element not clickable, form validation error)

Be SPECIFIC and GROUNDED in the current page structure. Do NOT hallucinate elements that aren't related to the action. If you're uncertain about what will happen, say so."""


VALUE_FUNCTION_PROMPT = """You are a value function for a web automation agent. Given the original objective and the predicted state after an action, rate whether the action moves toward achieving the goal.

Rating scale:
- COMPLETE (1.0): The goal appears to be FULLY achieved in the predicted state
- ON_TRACK (0.5): The action makes MEANINGFUL progress toward the goal (correct direction)
- WRONG (0.0): The action does NOT help, is irrelevant, or moves AWAY from the goal

Be strict: only rate COMPLETE if the goal is truly finished, not just partially done.
Rate ON_TRACK for incremental progress (e.g., navigated to the right page, filled a required field).
Rate WRONG for actions that waste steps or go in the wrong direction."""


# ═══════════════════════════════════════════════════════════════════════════════
#  WebDreamer Engine
# ═══════════════════════════════════════════════════════════════════════════════

class WebDreamer:
    """LLM-as-world-model planning engine.
    
    Algorithm (from the paper, adapted):
      1. get_candidates(state, objective) → [Action1, Action2, Action3]
      2. For each candidate:
         a. simulate(state, action) → predicted_next_state (natural language)
         b. score(predicted_state, objective) → {1.0, 0.5, 0.0}
      3. Execute argmax(scores)
    
    Args:
        invoke_fn: Async function with signature:
            (failover_chain, messages, schema, breaker, health_tracker) → (result, model_name)
        failover_chain: List of LLM model instances
        breaker: Circuit breaker instance
        health_tracker: Provider health tracker
        num_candidates: Number of candidate actions to generate (default 3)
        num_simulations: Number of simulation runs per candidate (default 1, max 3)
    """
    
    def __init__(
        self,
        invoke_fn,
        failover_chain: list,
        breaker,
        health_tracker,
        num_candidates: int = 3,
        num_simulations: int = 1,
    ):
        self._invoke = invoke_fn
        self._chain = failover_chain
        self._breaker = breaker
        self._health = health_tracker
        self._k = num_candidates
        self._n_sims = min(num_simulations, 3)
        self._total_calls = 0
        self._cache: dict[str, float] = {}  # action_desc → score cache

    # ── Step 1: Generate Candidate Actions ────────────────────────────────
    
    async def _get_candidates(
        self,
        dom_markdown: str,
        objective: str,
        plan_context: str,
        action_history: str,
        current_url: str,
        situation_note: str = "",
    ) -> list[CandidateAction]:
        """Generate k candidate actions from the current state."""

        messages = [
            SystemMessage(content=CANDIDATE_GENERATION_PROMPT),
            HumanMessage(content=(
                f"═══ OBJECTIVE ═══\n{objective}\n\n"
                f"═══ PLAN STATUS ═══\n{plan_context}\n\n"
                f"═══ CURRENT URL ═══\n{current_url}\n\n"
                f"═══ ACTION HISTORY ═══\n{action_history}\n\n"
                f"═══ PAGE STRUCTURE ═══\n{dom_markdown[:3000]}\n"
                f"{situation_note}\n"
                f"Now propose exactly {self._k} distinct candidate actions."
            )),
        ]
        
        try:
            result, model = await self._invoke(
                self._chain, messages, CandidateSet,
                self._breaker, health_tracker=self._health,
            )
            self._total_calls += 1
            
            candidates = []
            raw_candidates = result.candidates if hasattr(result, 'candidates') else []
            
            for c in raw_candidates[:self._k]:
                candidates.append(CandidateAction(
                    action_type=c.action_type or "wait",
                    element_id=c.element_id,
                    text=c.text,
                    url=c.url,
                    reasoning=c.reasoning or "",
                ))
            
            if not candidates:
                logger.warning("WebDreamer: LLM returned 0 candidates, using fallback")
                candidates = [CandidateAction(action_type="wait", reasoning="Fallback: no candidates generated")]
            
            logger.info(
                "WebDreamer: %d candidates generated by %s: %s",
                len(candidates), model,
                " | ".join(c.describe() for c in candidates),
            )
            return candidates
            
        except Exception as e:
            logger.warning("WebDreamer candidate generation failed: %s", e)
            return [CandidateAction(action_type="wait", reasoning=f"Fallback: generation error: {e}")]

    # ── Step 2: Simulate Outcome ──────────────────────────────────────────
    
    async def _simulate(
        self,
        dom_markdown: str,
        candidate: CandidateAction,
        objective: str,
    ) -> SimulatedOutcome:
        """Use the LLM as a world model to predict the outcome of an action."""
        
        action_desc = candidate.describe()
        
        # V15.0 F3: Extract specific element context from DOM markdown
        # When the DOM is truncated to 2500 chars, the target element [eN] may be cut.
        # The world model then hallucinates "element does not exist" and scores 0.0.
        # Fix: find the element's line and inject it explicitly into the prompt.
        element_context = ""
        if candidate.element_id:
            for line in dom_markdown.split("\n"):
                if f"[{candidate.element_id}]" in line:
                    element_context = line.strip()
                    break
        
        element_section = ""
        if element_context:
            element_section = (
                f"\n═══ TARGET ELEMENT (from DOM) ═══\n{element_context}\n"
            )
        
        messages = [
            SystemMessage(content=WORLD_MODEL_PROMPT),
            HumanMessage(content=(
                f"═══ CURRENT PAGE STATE ═══\n{dom_markdown[:2500]}\n"
                f"{element_section}\n"
                f"═══ PROPOSED ACTION ═══\n{action_desc}\n"
                f"Reasoning: {candidate.reasoning}\n\n"
                f"═══ CONTEXT ═══\nObjective: {objective[:200]}\n\n"
                "Predict what will change on the page after this action is executed."
            )),
        ]
        
        try:
            result, model = await self._invoke(
                self._chain, messages, WorldModelPrediction,
                self._breaker, health_tracker=self._health,
            )
            self._total_calls += 1
            
            return SimulatedOutcome(
                predicted_changes=result.predicted_changes if hasattr(result, 'predicted_changes') else "",
                new_elements="",
                disappeared_elements="",
                url_change=result.url_change if hasattr(result, 'url_change') else "",
                error_state=result.error_prediction if hasattr(result, 'error_prediction') else "",
            )
            
        except Exception as e:
            logger.warning("WebDreamer simulation failed for '%s': %s", action_desc, e)
            return SimulatedOutcome(
                predicted_changes=f"Simulation failed: {e}",
                new_elements="", disappeared_elements="",
                url_change="", error_state=str(e),
            )

    # ── Step 3: Score Simulated Outcome ───────────────────────────────────
    
    async def _score(
        self,
        simulated_outcome: SimulatedOutcome,
        objective: str,
        candidate: CandidateAction,
    ) -> tuple[float, str]:
        """Score the simulated outcome on a three-bucket scale.
        
        Returns: (score, label) where score is 1.0/0.5/0.0 and label is the verdict.
        """
        
        messages = [
            SystemMessage(content=VALUE_FUNCTION_PROMPT),
            HumanMessage(content=(
                f"═══ OBJECTIVE ═══\n{objective[:200]}\n\n"
                f"═══ ACTION TAKEN ═══\n{candidate.describe()}\n\n"
                f"═══ PREDICTED STATE AFTER ACTION ═══\n"
                f"{simulated_outcome.predicted_changes}\n"
                f"URL change: {simulated_outcome.url_change or 'none'}\n"
                f"Errors: {simulated_outcome.error_state or 'none'}\n\n"
                "Rate this outcome: COMPLETE, ON_TRACK, or WRONG."
            )),
        ]
        
        try:
            result, model = await self._invoke(
                self._chain, messages, ValueJudgment,
                self._breaker, health_tracker=self._health,
            )
            self._total_calls += 1
            
            verdict = result.verdict.strip().upper() if hasattr(result, 'verdict') else "WRONG"
            
            score_map = {
                "COMPLETE": 1.0,
                "ON_TRACK": 0.5,
                "ON TRACK": 0.5,
                "WRONG": 0.0,
            }
            score = score_map.get(verdict, 0.0)
            
            return score, verdict
            
        except Exception as e:
            logger.warning("WebDreamer scoring failed: %s", e)
            return 0.25, "UNKNOWN"  # Slightly above WRONG to not penalize errors

    # ── Main Entry Point ──────────────────────────────────────────────────
    
    async def plan_and_select(
        self,
        dom_markdown: str,
        objective: str,
        plan_context: str,
        action_history: str,
        current_url: str,
        proposed_action: CandidateAction | None = None,
        situation: dict | None = None,
    ) -> DreamerResult:
        """Run the full WebDreamer planning loop.
        
        1. Generate k candidates
        2. Simulate each (H=1)
        3. Score each simulation
        4. Return the best action
        
        Args:
            dom_markdown: Current page DOM as markdown
            objective: The task objective
            plan_context: Current plan state (from PlanState.render())
            action_history: Compressed action history
            current_url: Current page URL
        
        Returns:
            DreamerResult with the best action, its score, and all evaluations
        """
        start_time = time.monotonic()
        total_sims = 0

        # V29: situational tuning — flag-gated; None ⇒ vacuum-scoring baseline.
        situ = None
        situ_note = ""
        try:
            from feature_flags import webdreamer_situational_enabled
            if webdreamer_situational_enabled() and situation is not None:
                situ = extract_situation(situation)
                situ_note = situational_candidate_note(situ)
        except Exception as e:  # noqa: BLE001 — tuning never breaks planning
            logger.debug("situational tuning skipped: %s", e)
            situ = None

        # Step 1: Generate candidates (situationally hinted when tuning is active)
        candidates = await self._get_candidates(
            dom_markdown, objective, plan_context, action_history, current_url,
            situation_note=situ_note,
        )
        
        # Guarantee the original proposed action is in the candidate set
        if proposed_action:
            exists = any(
                c.action_type == proposed_action.action_type and 
                c.element_id == proposed_action.element_id 
                for c in candidates
            )
            if not exists:
                candidates.insert(0, proposed_action)
        
        # Step 2+3: Simulate and score each candidate
        evaluations: list[CandidateEvaluation] = []
        
        for candidate in candidates:
            action_desc = candidate.describe()
            
            # Check cache first
            cache_key = f"{current_url}|{action_desc}"
            if cache_key in self._cache:
                cached_score = self._cache[cache_key]
                logger.debug("WebDreamer: cache hit for '%s' → %.1f", action_desc, cached_score)
                evaluations.append(CandidateEvaluation(
                    candidate=candidate,
                    score=cached_score,
                    confidence=0.8,
                    score_label="CACHED",
                ))
                continue
            
            # Run n_sims simulations and average the scores
            sim_scores: list[float] = []
            last_outcome: SimulatedOutcome | None = None
            last_label = ""
            
            for sim_run in range(self._n_sims):
                outcome = await self._simulate(dom_markdown, candidate, objective)
                score, label = await self._score(outcome, objective, candidate)
                sim_scores.append(score)
                last_outcome = outcome
                last_label = label
                total_sims += 1
            
            avg_score = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
            
            # Cache the result
            self._cache[cache_key] = avg_score
            
            evaluation = CandidateEvaluation(
                candidate=candidate,
                simulated_outcome=last_outcome,
                score=avg_score,
                confidence=1.0 - (max(sim_scores) - min(sim_scores)) if len(sim_scores) > 1 else 0.7,
                score_label=last_label,
            )
            evaluations.append(evaluation)
            
            logger.info(
                "  📊 Candidate '%s' → score=%.2f (%s): %s",
                action_desc,
                avg_score,
                last_label,
                last_outcome.predicted_changes[:80] if last_outcome else "?",
            )
        
        # Step 4: Select argmax (situationally-adjusted when tuning is active)
        if evaluations:
            best_eval = select_best_evaluation(evaluations, situ)
            if situ:
                base_best = max(evaluations, key=lambda e: e.score)
                if base_best is not best_eval:
                    logger.info("🌙↻ Situational tuning changed pick: '%s' → '%s'",
                                base_best.candidate.describe(), best_eval.candidate.describe())
        else:
            best_eval = CandidateEvaluation(
                candidate=CandidateAction(action_type="wait", reasoning="No evaluations"),
                score=0.0,
                score_label="FALLBACK",
            )

        planning_time = (time.monotonic() - start_time) * 1000

        # Report the situationally-adjusted score so the override gate sees real value.
        best_final_score = adjusted_score(best_eval.score, best_eval.candidate.action_type, situ)

        result = DreamerResult(
            best_action=best_eval.candidate,
            best_score=best_final_score,
            all_evaluations=evaluations,
            planning_time_ms=planning_time,
            candidates_generated=len(candidates),
            simulations_run=total_sims,
        )
        
        logger.info(
            "🌙 WebDreamer: best='%s' score=%.2f (%s) | %d candidates, %d sims, %.0fms | LLM calls=%d",
            best_eval.candidate.describe(),
            best_eval.score,
            best_eval.score_label,
            len(candidates),
            total_sims,
            planning_time,
            self._total_calls,
        )
        
        return result

    def clear_cache(self) -> None:
        """Clear the simulation cache (e.g., after navigation)."""
        self._cache.clear()

    @property
    def total_llm_calls(self) -> int:
        """Total LLM calls made by the dreamer across all planning rounds."""
        return self._total_calls


# ═══════════════════════════════════════════════════════════════════════════════
#  V29 Situational scoring tuning (flag V29_WEBDREAMER_SITUATIONAL)
# ═══════════════════════════════════════════════════════════════════════════════
#  Vacuum-scoring rated `scroll` as universally "safe progress" and feared `goto`.
#  This layers a CONSERVATIVE, UNIVERSAL, additive (~±0.15) situational delta on top
#  of the LLM value score — keyed ONLY to signals already on the LangGraph state
#  (no new computation, no registry/Overwatch access) and to verb CLASS, never to
#  any site. It is applied solely in candidate SELECTION, so flag-off is identical
#  to the baseline.

SITU_REVEAL_THRESHOLD = 0.4      # last action's state_change_score ≥ this + static URL = a "reveal"
SITU_REVEAL_BONUS = 0.15         # engage the just-revealed content
SITU_REVEAL_PENALTY = 0.15       # scroll/goto away right after a reveal
SITU_EXPLORE_BONUS = 0.15        # max `goto` bonus when fully stuck
SITU_SCROLL_DEADEND_PENALTY = 0.15  # scroll that reveals nothing new (NOT productive scroll)
SITU_SCROLL_DECAY_AT = 2         # scroll_stuck_streak at which scroll productivity → 0
SITU_STUCK_URL_FULL = 4          # same_url_streak at which stuckness = 1.0
SITU_STAGNATION_FULL = 3         # stagnation_level at which stuckness = 1.0
_ENGAGE_VERBS = ("click", "type", "select_option", "hover", "press_enter")


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def extract_situation(state: dict | None) -> dict:
    """Read the ALREADY-computed situational signals from the LangGraph state.
    Pure read — no new computation, no `window.__aid`/Overwatch access."""
    s = state or {}
    scs = float(s.get("state_change_score", 0.0) or 0.0)
    same_url = int(s.get("same_url_streak", 0) or 0)
    scroll_stuck = int(s.get("scroll_stuck_streak", 0) or 0)
    stagnation = int(s.get("stagnation_level", 0) or 0)
    return {
        # a big DOM change with a STATIC url = a JS toggle/accordion/SPA reveal
        "reveal": (scs >= SITU_REVEAL_THRESHOLD) and (same_url >= 1),
        # stuckness ∈ [0,1] from same-URL persistence or stagnation
        "stuckness": _clamp01(max(same_url / SITU_STUCK_URL_FULL,
                                  stagnation / SITU_STAGNATION_FULL)),
        # scroll productivity ∈ [0,1] — 1 on an infinite feed, → 0 when scrolling stalls
        "scroll_productive": _clamp01(1.0 - scroll_stuck / max(1, SITU_SCROLL_DECAY_AT)),
        "scroll_stuck": scroll_stuck,
    }


def situational_adjustment(verb: str, situation: dict) -> float:
    """Conservative additive score delta (≈ ±0.15) for an action given the situation.
    Universal: keyed to verb CLASS + signals, never to any site/path."""
    v = (verb or "").strip().lower()
    adj = 0.0
    if situation.get("reveal"):
        if v in _ENGAGE_VERBS:
            adj += SITU_REVEAL_BONUS         # look at what you just revealed
        elif v in ("scroll", "goto"):
            adj -= SITU_REVEAL_PENALTY       # don't abandon the reveal before reading it
    if v == "scroll":
        # penalize ONLY unproductive scroll → a productive infinite feed is untouched
        adj -= SITU_SCROLL_DEADEND_PENALTY * (1.0 - float(situation.get("scroll_productive", 1.0)))
    if v == "goto":
        adj += SITU_EXPLORE_BONUS * float(situation.get("stuckness", 0.0))
    return adj


def adjusted_score(base: float, verb: str, situation: dict | None) -> float:
    """Base LLM value + situational delta, clamped. `situation=None` ⇒ base (baseline)."""
    if not situation:
        return _clamp01(base)
    return _clamp01(base + situational_adjustment(verb, situation))


def select_best_evaluation(evaluations: list, situation: dict | None = None):
    """argmax over evaluations by situationally-adjusted score (pure, testable).
    `situation=None` ⇒ plain argmax (vacuum baseline)."""
    if not evaluations:
        return None
    return max(evaluations,
               key=lambda e: adjusted_score(e.score, e.candidate.action_type, situation))


def situational_candidate_note(situation: dict) -> str:
    """Short UNIVERSAL hint injected into candidate generation (no site/path logic)."""
    notes = []
    if situation.get("reveal"):
        notes.append("Your last action just revealed NEW on-page content WITHOUT changing the "
                     "URL (a toggle/accordion/expand). Strongly prefer a candidate that ENGAGES "
                     "or reads the newly visible content over scrolling or navigating away.")
    if situation.get("stuckness", 0.0) >= 0.75:
        notes.append("Repeated interactions have NOT advanced the page (stuck on a static URL). "
                     "Consider navigating directly toward where the goal likely lives, using the "
                     "current URL's own structure.")
    if float(situation.get("scroll_productive", 1.0)) <= 0.1:
        notes.append("Recent scrolling has revealed nothing new — further scrolling is a dead end.")
    return ("\n═══ SITUATIONAL HINT ═══\n" + " ".join(notes)) if notes else ""


def should_override_with_dreamer(
    best_score: float,
    best_verb: str,
    best_element_id: str | None,
    cur_verb: str,
    cur_element_id: str | None,
    min_score: float = 0.6,
) -> bool:
    """Pure decision: should WebDreamer's best imagined action REPLACE the worker's
    pick? Only when it scored well AND is genuinely DIFFERENT (otherwise the dreamer
    merely CONFIRMS the original — keep it, no churn). Unit-testable."""
    if best_score < min_score:
        return False
    return (best_verb or "") != (cur_verb or "") or (best_element_id or "") != (cur_element_id or "")


# ═══════════════════════════════════════════════════════════════════════════════
#  Complexity Gate — Decides when to invoke WebDreamer
# ═══════════════════════════════════════════════════════════════════════════════

def should_invoke_dreamer(
    element_count: int,
    action_risk_level: str,
    consecutive_no_progress: int,
    step_number: int,
    same_url_streak: int,
) -> bool:
    """Decide whether the current step warrants WebDreamer simulation.
    
    WebDreamer is expensive (~9 LLM calls per step with k=3), so we only
    invoke it when the agent actually needs help deciding.
    
    Returns True if:
    - The action is IRREVERSIBLE (always simulate before irreversible actions)
    - Multiple interactive elements visible AND agent has been stuck
    - Agent has been on the same URL for 3+ steps (confusion/loop)
    - Agent has had 2+ consecutive no-progress actions
    
    Returns False for:
    - Simple navigations (few elements, low risk)
    - First few steps of a task (agent usually knows what to do)
    - Steps immediately after successful progress
    """
    # Always simulate before irreversible actions
    if action_risk_level == "IRREVERSIBLE":
        return True
    
    # Agent is stuck — invoke dreamer for help
    if consecutive_no_progress >= 2:
        return True
    
    # Agent is confused (same URL, many elements)
    if same_url_streak >= 3 and element_count > 5:
        return True
    
    # Complex page with CAUTIOUS action — dreamer can help
    if action_risk_level == "CAUTIOUS" and element_count > 15 and step_number > 3:
        return True
    
    # Simple case — skip dreamer
    return False
