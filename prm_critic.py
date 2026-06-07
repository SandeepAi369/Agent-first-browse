"""PRM Critic — Checklist-Based Process Reward Model for Step-Level Progress Scoring.

Implements the Web-Shepherd pattern (Kim et al., arXiv:2505.15277):
  1. At task start: decompose the objective into a checklist of verifiable sub-goals
  2. After each action: score which checklist items are now satisfied
  3. Reward = weighted sum of checklist scores

This provides dense, goal-aware progress signal that the heuristic CriticV12
cannot offer. CriticV12 detects "did something change?" — this module detects
"did we get CLOSER TO THE GOAL?"

Integration:
  - Augments (not replaces) CriticV12 for ambiguous verdicts
  - Serves as the value function inside WebDreamer's scoring step
  - Generates the checklist alongside _decompose_objective at task start
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from app.logger import get_logger
    logger = get_logger("prm_critic")
except ImportError:
    import logging
    logger = logging.getLogger("prm_critic")


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChecklistItem:
    """A single verifiable sub-goal in the objective checklist."""
    id: int
    description: str
    status: str = "pending"     # pending | in_progress | done | failed
    confidence: float = 0.0     # 0.0 to 1.0 — how confident we are this is done
    step_completed: int = -1    # Which step completed this item (-1 = not done)
    
    @property
    def score(self) -> float:
        """Numeric score for this item."""
        if self.status == "done":
            return 1.0
        elif self.status == "in_progress":
            return 0.5 * self.confidence
        else:
            return 0.0


@dataclass
class StepScore:
    """Score for a single agent step against the checklist."""
    total_score: float          # Sum of checklist scores / num items (0.0 to 1.0)
    items_done: int             # Number of checklist items completed
    items_total: int            # Total checklist items
    items_in_progress: int      # Items partially done
    newly_completed: list[str]  # Descriptions of items completed in this step
    progress_delta: float       # Change in score from previous step
    verdict: str                # "PROGRESS" | "NO_CHANGE" | "REGRESSION"


@dataclass  
class TrajectoryScore:
    """Score for a predicted trajectory (used by WebDreamer)."""
    score: float                # 0.0 to 1.0
    items_likely_done: int      # Predicted completed items
    explanation: str            # Why this score


# ═══════════════════════════════════════════════════════════════════════════════
#  Pydantic Schemas for LLM Structured Output
# ═══════════════════════════════════════════════════════════════════════════════

class ChecklistGeneration(BaseModel):
    """LLM output: checklist of verifiable sub-goals."""
    items: list[str] = Field(
        description=(
            "A list of 4-8 verifiable sub-goals that must be completed to achieve "
            "the objective. Each item should be specific and independently verifiable. "
            "Order them in the expected sequence of completion. "
            "Example for 'Add Ryzen 7 to cart on Amazon': "
            "['Navigate to amazon.in', 'Search for Ryzen 7 9700X', "
            "'Identify non-sponsored results', 'Select the cheapest valid option', "
            "'Click Add to Cart button', 'Verify cart confirmation message']"
        )
    )


class ChecklistEvaluation(BaseModel):
    """LLM output: evaluation of checklist items against current state."""
    evaluations: list[dict] = Field(
        description=(
            "For each checklist item, evaluate its status based on the current page state. "
            "Each dict must have: 'id' (int, 0-indexed), 'status' ('done', 'in_progress', or 'pending'), "
            "'confidence' (float 0.0-1.0), 'evidence' (brief reason, 1 sentence). "
            "Base your evaluation ONLY on what you can see in the current page state."
        )
    )


class TrajectoryEvaluation(BaseModel):
    """LLM output: evaluation of a predicted future state against checklist."""
    items_likely_done: int = Field(
        description="How many checklist items would be completed in the predicted state."
    )
    score: float = Field(
        description="Overall progress score from 0.0 (no progress) to 1.0 (all done)."
    )
    explanation: str = Field(
        description="Brief explanation of why this predicted state earns this score."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompts
# ═══════════════════════════════════════════════════════════════════════════════

CHECKLIST_GEN_PROMPT = """You are a task decomposition expert. Break the given browser automation objective into 4-8 verifiable sub-goals (checklist items).

Rules:
- Each item must be specific and independently verifiable from the page state
- Items should be in chronological order of expected completion
- Include verification steps (e.g., "Verify the item was added to cart")
- Don't include meta-steps like "Open browser" or "Close browser"
- Each item should correspond to a meaningful milestone in the task"""


CHECKLIST_EVAL_PROMPT = """You are a progress evaluator for a browser automation agent. Given the current page state and a checklist of sub-goals, evaluate which items have been completed.

Rules:
- Base your evaluation ONLY on the evidence in the current page state
- 'done' = you can see clear evidence this step is completed
- 'in_progress' = partial evidence (e.g., on the right page but haven't found the element yet)
- 'pending' = no evidence of progress on this item
- Set confidence 0.0-1.0 based on how certain you are
- Be conservative: only mark 'done' if you're quite sure"""


TRAJECTORY_EVAL_PROMPT = """You are a value function for a web automation agent. Given the objective's checklist and a PREDICTED future state (not the current state), estimate how many checklist items would be completed.

Be realistic: predicted states may be optimistic. Score conservatively."""


# ═══════════════════════════════════════════════════════════════════════════════
#  PRM Critic Engine
# ═══════════════════════════════════════════════════════════════════════════════

class PRMCritic:
    """Checklist-based Process Reward Model.
    
    Usage:
        prm = PRMCritic(invoke_fn, chain, breaker, health)
        checklist = await prm.generate_checklist("Add Ryzen to cart on Amazon")
        
        # After each step:
        step_score = await prm.score_step(checklist, dom_markdown)
        
        # For WebDreamer (predicted states):
        traj_score = await prm.score_trajectory(checklist, predicted_state_text)
    """
    
    def __init__(self, invoke_fn, failover_chain, breaker, health_tracker):
        self._invoke = invoke_fn
        self._chain = failover_chain
        self._breaker = breaker
        self._health = health_tracker
        self._prev_score: float = 0.0
        self._total_calls: int = 0
    
    async def generate_checklist(self, objective: str) -> list[ChecklistItem]:
        """Decompose the objective into a checklist of verifiable sub-goals.
        
        Called once at task start. The checklist persists throughout execution.
        """
        messages = [
            SystemMessage(content=CHECKLIST_GEN_PROMPT),
            HumanMessage(content=(
                f"Objective: {objective}\n\n"
                "Break this into 4-8 verifiable sub-goals."
            )),
        ]
        
        try:
            result, model = await self._invoke(
                self._chain, messages, ChecklistGeneration,
                self._breaker, health_tracker=self._health,
            )
            self._total_calls += 1
            
            items_raw = result.items if hasattr(result, 'items') else []
            checklist = [
                ChecklistItem(id=i, description=desc.strip())
                for i, desc in enumerate(items_raw)
                if desc.strip()
            ]
            
            if not checklist:
                logger.warning("PRM: LLM returned empty checklist, creating fallback")
                checklist = [ChecklistItem(id=0, description=objective)]
            
            logger.info(
                "📋 PRM Checklist (%d items by %s): %s",
                len(checklist), model,
                " → ".join(item.description[:40] for item in checklist),
            )
            return checklist
            
        except Exception as e:
            logger.warning("PRM checklist generation failed: %s — using objective as single item", e)
            return [ChecklistItem(id=0, description=objective)]
    
    async def score_step(
        self,
        checklist: list[ChecklistItem],
        dom_markdown: str,
        current_url: str = "",
        step_number: int = 0,
    ) -> StepScore:
        """Score the current page state against the checklist.
        
        Compares item statuses before and after to detect progress.
        """
        # Build checklist context
        checklist_text = "\n".join(
            f"  [{item.id}] {item.description} (currently: {item.status})"
            for item in checklist
        )
        
        messages = [
            SystemMessage(content=CHECKLIST_EVAL_PROMPT),
            HumanMessage(content=(
                f"═══ CHECKLIST ═══\n{checklist_text}\n\n"
                f"═══ CURRENT URL ═══\n{current_url}\n\n"
                f"═══ CURRENT PAGE STATE ═══\n{dom_markdown[:2500]}\n\n"
                "Evaluate each checklist item against the current page state."
            )),
        ]
        
        try:
            result, model = await self._invoke(
                self._chain, messages, ChecklistEvaluation,
                self._breaker, health_tracker=self._health,
            )
            self._total_calls += 1
            
            evaluations = result.evaluations if hasattr(result, 'evaluations') else []
            newly_completed = []
            
            for eval_item in evaluations:
                item_id = eval_item.get("id", -1)
                if 0 <= item_id < len(checklist):
                    item = checklist[item_id]
                    new_status = eval_item.get("status", "pending")
                    new_confidence = float(eval_item.get("confidence", 0.0))
                    
                    # Track newly completed items
                    if new_status == "done" and item.status != "done":
                        newly_completed.append(item.description)
                        item.step_completed = step_number
                    
                    item.status = new_status
                    item.confidence = new_confidence
            
        except Exception as e:
            logger.warning("PRM step scoring failed: %s", e)
        
        # Calculate aggregate score
        total_score = sum(item.score for item in checklist) / len(checklist) if checklist else 0.0
        items_done = sum(1 for item in checklist if item.status == "done")
        items_in_progress = sum(1 for item in checklist if item.status == "in_progress")
        
        progress_delta = total_score - self._prev_score
        self._prev_score = total_score
        
        if progress_delta > 0.05:
            verdict = "PROGRESS"
        elif progress_delta < -0.05:
            verdict = "REGRESSION"
        else:
            verdict = "NO_CHANGE"
        
        step_score = StepScore(
            total_score=total_score,
            items_done=items_done,
            items_total=len(checklist),
            items_in_progress=items_in_progress,
            newly_completed=newly_completed,
            progress_delta=progress_delta,
            verdict=verdict,
        )
        
        logger.info(
            "📊 PRM: %d/%d done (%.0f%%) %s%s | %s",
            items_done, len(checklist),
            total_score * 100,
            "↑" if progress_delta > 0 else ("↓" if progress_delta < 0 else "→"),
            f" +{', '.join(c[:30] for c in newly_completed)}" if newly_completed else "",
            verdict,
        )
        
        return step_score
    
    async def score_trajectory(
        self,
        checklist: list[ChecklistItem],
        predicted_state: str,
        objective: str = "",
    ) -> TrajectoryScore:
        """Score a PREDICTED future state against the checklist.
        
        Used by WebDreamer as the value function for ranking candidates.
        """
        checklist_text = "\n".join(
            f"  [{item.id}] {item.description} (currently: {item.status})"
            for item in checklist
        )
        
        messages = [
            SystemMessage(content=TRAJECTORY_EVAL_PROMPT),
            HumanMessage(content=(
                f"═══ OBJECTIVE ═══\n{objective[:200]}\n\n"
                f"═══ CHECKLIST ═══\n{checklist_text}\n\n"
                f"═══ PREDICTED STATE AFTER ACTION ═══\n{predicted_state}\n\n"
                "Estimate the score for this predicted state."
            )),
        ]
        
        try:
            result, model = await self._invoke(
                self._chain, messages, TrajectoryEvaluation,
                self._breaker, health_tracker=self._health,
            )
            self._total_calls += 1
            
            return TrajectoryScore(
                score=float(result.score) if hasattr(result, 'score') else 0.0,
                items_likely_done=int(result.items_likely_done) if hasattr(result, 'items_likely_done') else 0,
                explanation=result.explanation if hasattr(result, 'explanation') else "",
            )
            
        except Exception as e:
            logger.warning("PRM trajectory scoring failed: %s", e)
            return TrajectoryScore(score=0.25, items_likely_done=0, explanation=f"Error: {e}")

    @property
    def total_llm_calls(self) -> int:
        return self._total_calls
