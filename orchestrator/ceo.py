"""THE CEO — Master Planner + DAG Builder.

Takes a raw user prompt, classifies the task, and builds a
Directed Acyclic Graph (DAG) of sub-tasks using LLM structured output.
Can replan when the Critic reports failures.

Inspired by Magentic-One's Orchestrator (Task Ledger + Progress Ledger)
and LangGraph's conditional routing.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from orchestrator.actions import (
    TaskDAG, TaskNode, ActionType, NodeStatus,
    CEOPlan, PlannedStep,
)
from orchestrator.event_log import EventLog
from orchestrator.state import OrchestratorState

logger = logging.getLogger("orchestrator.ceo")

# ═══════════════════════════════════════════════════════════════════════════════
#  CEO System Prompt — The brain's operating instructions
# ═══════════════════════════════════════════════════════════════════════════════

_CEO_SYSTEM_PROMPT = """\
You are the CEO — an elite task planner for a browser automation system.

Given a user's objective, you MUST:
1. CLASSIFY the task type (browsing, scraping, posting, research, login, automation, other).
2. REASON about the best execution strategy in 2-3 sentences.
3. DECOMPOSE the objective into 2-8 atomic steps.

Each step MUST use exactly ONE of these actions:
  navigate  — Go to a URL. Params: {url: "..."}
  click     — Click an element. Params: {interaction: "click", x: N, y: N} or {interaction: "click", selector: "css"}
  type      — Type text. Params: {interaction: "type", text: "...", x: N, y: N}
  scroll    — Scroll the page. Params: {interaction: "scroll", delta: 500}
  extract   — Extract data from the page. Params: {mode: "dom_map|text_content|query|form_fields", selector: "..."}
  execute_js — Run JavaScript. Params: {js_code: "..."}
  wait      — Pause. Params: {duration_ms: 1000}
  screenshot — Capture page state. Params: {}
  done      — Signal completion. Params: {result_summary: "..."}

RULES:
- Steps are 0-indexed. Use depends_on to specify ordering.
- The first step usually has depends_on: [].
- Always start with a navigate or extract step.
- After navigation, include an extract step to map the DOM before interacting.
- End with a done step.
- For scraping: navigate → extract (query) → done with extracted data.
- For posting: navigate → extract (form_fields) → type/click → done.
- For research: navigate → extract (text_content) → done with summary.
- Be specific in descriptions. No vague steps.
"""


class CEO:
    """Master Planner — Decomposes any objective into a structured DAG.

    The CEO does NOT execute anything. It only plans.
    The Spawner and Executor handle execution.
    """

    def __init__(
        self,
        failover_chain: list,
        health_tracker: Any = None,
        circuit_breaker: Any = None,
    ) -> None:
        self._failover_chain = failover_chain
        self._health_tracker = health_tracker
        self._breaker = circuit_breaker

    async def plan(self, objective: str, context: dict | None = None) -> TaskDAG:
        """Decompose an objective into a TaskDAG.

        Args:
            objective: The raw user prompt.
            context: Optional context (current URL, DOM state, etc.).

        Returns:
            A TaskDAG ready for the Executor.
        """
        logger.info("CEO planning for: %s", objective[:120])

        # Build the planning prompt
        context_str = ""
        if context:
            context_str = f"\n\nCurrent context:\n{json.dumps(context, indent=2, default=str)[:2000]}"

        user_prompt = (
            f"Objective: {objective}\n"
            f"{context_str}\n\n"
            "Produce your plan now."
        )

        messages = [
            SystemMessage(content=_CEO_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # Try structured output first, fall back to JSON parsing
        plan = await self._get_plan(messages)

        # Convert CEOPlan → TaskDAG
        dag = self._plan_to_dag(plan, objective)
        logger.info("CEO plan created: %d nodes, type=%s", len(dag.nodes), plan.task_type)
        logger.info("\n%s", dag.summary())

        return dag

    async def replan(
        self,
        state: OrchestratorState,
        failed_node: TaskNode,
        error: str,
        event_log: EventLog,
    ) -> TaskDAG | None:
        """Replan after a node failure.

        Uses the Progress Ledger and Event Log to provide rich context
        about what went wrong, so the CEO can devise an alternative.

        Returns:
            New TaskDAG, or None if max retries exceeded.
        """
        replan_count = state.progress_ledger.replan_count
        if replan_count >= state.max_replan_attempts:
            logger.warning(
                "CEO: Max replan attempts (%d) reached. Requesting human intervention.",
                state.max_replan_attempts,
            )
            return None

        state.progress_ledger.replan_count += 1

        logger.info(
            "CEO replanning (attempt %d/%d) after node '%s' failed: %s",
            replan_count + 1, state.max_replan_attempts,
            failed_node.id, error[:150],
        )

        # Build rich context for replanning
        recent_events = event_log.to_context(last_n=10)
        progress = state.progress_ledger.summary()

        replan_prompt = (
            f"The previous plan FAILED at step '{failed_node.description}'.\n"
            f"Error: {error}\n\n"
            f"Original objective: {state.objective}\n"
            f"Progress so far: {progress}\n"
            f"Current URL: {state.current_url}\n\n"
            f"Recent execution history:\n{recent_events}\n\n"
            f"DOM state:\n{state.dom_tree[:2000]}\n\n"
            "Create a NEW plan that avoids the previous failure. "
            "You may skip steps that already succeeded. "
            "Think about WHY the previous approach failed and use a DIFFERENT strategy."
        )

        messages = [
            SystemMessage(content=_CEO_SYSTEM_PROMPT),
            HumanMessage(content=replan_prompt),
        ]

        try:
            plan = await self._get_plan(messages)
            dag = self._plan_to_dag(plan, state.objective)
            logger.info("CEO replan created: %d nodes", len(dag.nodes))
            return dag
        except Exception as e:
            logger.error("CEO replan failed: %s", e)
            return None

    # ── Internal: LLM Invocation ──────────────────────────────────────────

    async def _get_plan(self, messages: list) -> CEOPlan:
        """Invoke the LLM with structured output to get a CEOPlan."""
        import asyncio

        last_error = None
        model_names = [
            getattr(m, 'model_name', getattr(m, 'model', str(m)))
            for m in self._failover_chain
        ]

        for idx, llm in enumerate(self._failover_chain):
            model_name = model_names[idx]

            # Skip quarantined models
            if self._health_tracker and not self._health_tracker.is_available(model_name):
                if idx < len(self._failover_chain) - 1:
                    continue

            try:
                structured = llm.with_structured_output(CEOPlan)
                result = await asyncio.wait_for(
                    structured.ainvoke(messages),
                    timeout=30.0,
                )

                if self._health_tracker:
                    self._health_tracker.record_success(model_name)
                if self._breaker:
                    self._breaker.record_success()

                return result

            except Exception as e:
                last_error = e
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()

                if self._health_tracker:
                    self._health_tracker.record_failure(model_name, is_rate_limit=is_rate_limit)
                if self._breaker:
                    self._breaker.record_failure()

                logger.warning(
                    "CEO model [%d/%d] %s failed: %s",
                    idx + 1, len(self._failover_chain), model_name, error_str[:120],
                )

        # All models failed — try unstructured JSON parsing as last resort
        return await self._fallback_json_plan(messages, last_error)

    async def _fallback_json_plan(self, messages: list, last_error: Exception | None) -> CEOPlan:
        """Fallback: ask for raw JSON if structured output failed on all models."""
        import asyncio

        fallback_msg = messages.copy()
        fallback_msg[-1] = HumanMessage(
            content=str(fallback_msg[-1].content) + "\n\nReturn your answer as raw JSON matching this schema: "
            '{"task_type": "...", "reasoning": "...", "steps": [{"description": "...", "action": "...", "params": {...}, "depends_on": [...]}]}'
        )

        for llm in self._failover_chain:
            try:
                result = await asyncio.wait_for(llm.ainvoke(fallback_msg), timeout=30.0)
                text = result.content if hasattr(result, "content") else str(result)
                text = text.strip()

                # Extract JSON from markdown code blocks
                if "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                    if text.startswith("json"):
                        text = text[4:].strip()

                data = json.loads(text)
                return CEOPlan(**data)
            except Exception:
                continue

        raise RuntimeError(f"CEO: ALL models failed to produce a plan. Last error: {last_error}")

    # ── Internal: Plan → DAG Conversion ───────────────────────────────────

    def _plan_to_dag(self, plan: CEOPlan, objective: str) -> TaskDAG:
        """Convert a CEOPlan (LLM output) into a TaskDAG (executable graph)."""
        dag = TaskDAG(goal=objective, metadata={"task_type": plan.task_type, "reasoning": plan.reasoning})

        # Create nodes with stable IDs
        node_ids: list[str] = []
        for i, step in enumerate(plan.steps):
            action = self._normalize_action(step.action)
            node = TaskNode(
                id=f"step_{i}",
                description=step.description,
                action=action,
                params=step.params,
                dependencies=[f"step_{d}" for d in step.depends_on if d < i],
            )
            dag.add_node(node)
            node_ids.append(node.id)

        return dag

    @staticmethod
    def _normalize_action(action_str: str) -> str:
        """Normalize LLM action strings to valid ActionType values."""
        mapping = {
            "navigate": "navigate", "goto": "navigate", "open": "navigate",
            "click": "click", "press": "click",
            "type": "type", "input": "type", "fill": "type",
            "scroll": "scroll",
            "extract": "extract", "scrape": "extract", "read": "extract", "get": "extract",
            "execute_js": "execute_js", "eval": "execute_js", "javascript": "execute_js",
            "wait": "wait", "pause": "wait", "sleep": "wait",
            "screenshot": "screenshot", "capture": "screenshot",
            "done": "done", "finish": "done", "complete": "done",
        }
        return mapping.get(action_str.lower(), action_str.lower())
