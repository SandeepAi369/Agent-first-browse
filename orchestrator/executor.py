"""THE EXECUTOR — Runtime Engine + Critic Loop.

Walks the TaskDAG in topological order, dispatching each node
to the Spawner for execution, then passing the result through
the Critic for evaluation. If a node fails, it asks the CEO
to replan (up to max_replan_attempts).

This is the main entry point for running any task through
the Multi-Agent Orchestrator.

Inspired by:
  - OpenHands' Action-Observation loop
  - Magentic-One's Progress Ledger reflection
  - LangGraph's conditional routing with checkpoint/resume
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from playwright.async_api import Page

from orchestrator.actions import TaskDAG, TaskNode, NodeStatus
from orchestrator.ceo import CEO
from orchestrator.critic import Critic, Verdict
from orchestrator.event_log import EventLog
from orchestrator.spawner import Spawner
from orchestrator.state import OrchestratorState, ProgressEntry

logger = logging.getLogger("orchestrator.executor")


class ExecutionResult:
    """Final result of the entire orchestration run."""

    def __init__(
        self,
        success: bool,
        summary: str,
        data: Any = None,
        state: OrchestratorState | None = None,
    ):
        self.success = success
        self.summary = summary
        self.data = data
        self.state = state

    def __repr__(self) -> str:
        icon = "✅" if self.success else "❌"
        return f"ExecutionResult({icon} {self.summary})"


class Executor:
    """Runtime Engine — Walks the DAG and executes each node.

    The Executor is the bridge between the CEO's plan and reality.
    It handles:
    1. Topological traversal of the DAG
    2. Node execution via the Spawner
    3. Result evaluation via the Critic
    4. Failure recovery via CEO replanning
    5. State persistence via the Event Log
    """

    def __init__(
        self,
        page: Page,
        ceo: CEO,
        persistence_dir: Path | None = None,
    ) -> None:
        self._page = page
        self._ceo = ceo
        self._spawner = Spawner(page)
        self._critic = Critic(page)
        self._event_log = EventLog()
        self._state = OrchestratorState()
        self._persistence_dir = persistence_dir or Path("persistence/orchestrator")

    @property
    def state(self) -> OrchestratorState:
        return self._state

    @property
    def event_log(self) -> EventLog:
        return self._event_log

    async def run(self, objective: str) -> ExecutionResult:
        """Execute an objective from start to finish.

        This is the main entry point for the orchestrator.

        Steps:
        1. CEO plans the task → produces a DAG
        2. Walk the DAG in topological order
        3. For each node: Snapshot → Execute → Critique → Record
        4. If a node fails: Replan (up to 3 times)
        5. If all nodes succeed: Return final result
        6. If max replans exceeded: Pause for human intervention
        """
        self._state.objective = objective
        self._state.task_ledger.objective = objective
        self._event_log.system(f"Orchestrator started: {objective[:120]}")

        logger.info("━" * 60)
        logger.info("🚀 ORCHESTRATOR: %s", objective[:120])
        logger.info("━" * 60)

        # ── Phase 1: CEO Plans ────────────────────────────────────────────
        try:
            dag = await self._ceo.plan(objective)
        except Exception as e:
            logger.error("CEO planning failed: %s", e)
            self._event_log.error(f"Planning failed: {e}")
            return ExecutionResult(
                success=False,
                summary=f"Planning failed: {e}",
                state=self._state,
            )

        self._state.dag = dag
        self._state.task_ledger.task_type = dag.metadata.get("task_type", "unknown")
        self._event_log.append(
            __import__("orchestrator.event_log", fromlist=["Event"]).Event(
                event_type="plan",
                summary=f"CEO created plan: {len(dag.nodes)} steps ({dag.metadata.get('task_type', '?')})",
                data={"dag_summary": dag.summary()},
            )
        )

        # ── Phase 2: Execute DAG ──────────────────────────────────────────
        result = await self._execute_dag(dag)

        # ── Phase 3: Persist & Return ─────────────────────────────────────
        self._save_event_log()

        elapsed = self._state.elapsed_seconds()
        total = self._state.progress_ledger.total_actions
        success_rate = self._state.progress_ledger.success_rate()

        logger.info("")
        logger.info("━" * 60)
        if result.success:
            logger.info("🎯 MISSION ACCOMPLISHED")
        else:
            logger.info("⚠️  MISSION INCOMPLETE")
        logger.info("━" * 60)
        logger.info("  Objective:    %s", objective[:80])
        logger.info("  Steps:        %d", total)
        logger.info("  Success Rate: %.0f%%", success_rate * 100)
        logger.info("  Elapsed:      %.1fs", elapsed)
        logger.info("━" * 60)
        logger.info("")

        return result

    # ── Core DAG Execution Loop ───────────────────────────────────────────

    async def _execute_dag(self, dag: TaskDAG) -> ExecutionResult:
        """Walk the DAG in topological order and execute each node."""
        self._state.dag = dag
        collected_data: dict[str, Any] = {}

        for node in dag.topological_order():
            if node.status in (NodeStatus.DONE, NodeStatus.SKIPPED):
                continue

            # Mark running
            node.status = NodeStatus.RUNNING
            self._state.current_node_id = node.id

            logger.info("")
            logger.info(
                "━━━ Node %s [%s] ━━━",
                node.id, node.action,
            )
            logger.info("  📝 %s", node.description)

            # Update current URL
            try:
                self._state.current_url = self._page.url
            except Exception:
                pass

            # ── Snapshot pre-action state ──
            await self._critic.snapshot_before()

            # ── Execute ──
            try:
                result = await self._spawner.execute_node(node, self._event_log)
            except Exception as e:
                logger.error("Node execution crashed: %s", e)
                result = __import__("skills.base", fromlist=["SkillResult"]).SkillResult(
                    success=False,
                    summary=f"Execution crashed: {e}",
                    error=str(e),
                )

            # ── Critique ──
            await asyncio.sleep(0.3)  # Let DOM settle
            verdict = await self._critic.evaluate(node, result)

            # ── Record in Progress Ledger ──
            entry = ProgressEntry(
                node_id=node.id,
                action_taken=f"{node.action}: {node.description[:80]}",
                observation=result.summary,
                success=verdict.success,
                reasoning=verdict.reason,
            )
            self._state.progress_ledger.record(entry)

            self._event_log.critique(
                node.id,
                f"{'✅' if verdict.success else '❌'} {verdict.reason}",
                confidence=verdict.confidence,
            )

            if verdict.success:
                node.status = NodeStatus.DONE
                node.result = result.data
                logger.info("  ✅ %s", verdict.reason)

                # Collect extracted data
                if node.action == "extract" and result.data:
                    collected_data[node.id] = result.data

                # Check for done node
                if node.action == "done":
                    self._state.mark_complete(collected_data)
                    return ExecutionResult(
                        success=True,
                        summary="Task completed successfully",
                        data=collected_data,
                        state=self._state,
                    )
            else:
                node.status = NodeStatus.FAILED
                node.error = verdict.reason
                node.retries += 1
                logger.warning("  ❌ %s", verdict.reason)

                # ── Replan ──
                if node.retries <= self._state.max_replan_attempts:
                    logger.info("  🔄 Requesting CEO replan (attempt %d/%d)...",
                                node.retries, self._state.max_replan_attempts)

                    # Refresh DOM state for CEO context
                    await self._refresh_dom_state()

                    new_dag = await self._ceo.replan(
                        self._state, node, verdict.reason, self._event_log,
                    )

                    if new_dag:
                        return await self._execute_dag(new_dag)
                    else:
                        self._state.request_human_intervention(
                            f"CEO replan returned empty after node '{node.id}' failed: {verdict.reason}"
                        )
                        return ExecutionResult(
                            success=False,
                            summary=f"Replan failed at node: {node.description}",
                            data=collected_data,
                            state=self._state,
                        )
                else:
                    self._state.request_human_intervention(
                        f"Max retries ({self._state.max_replan_attempts}) exceeded for: {node.description}"
                    )
                    return ExecutionResult(
                        success=False,
                        summary=f"Max retries exceeded at: {node.description}",
                        data=collected_data,
                        state=self._state,
                    )

        # All nodes processed
        if dag.is_complete():
            self._state.mark_complete(collected_data)
            return ExecutionResult(
                success=True,
                summary="All DAG nodes completed",
                data=collected_data,
                state=self._state,
            )

        return ExecutionResult(
            success=False,
            summary="DAG execution ended with incomplete nodes",
            data=collected_data,
            state=self._state,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _refresh_dom_state(self) -> None:
        """Refresh DOM state in the orchestrator state for CEO context."""
        try:
            import dom_parser
            dom_data = await dom_parser.extract(self._page, timeout=5.0)
            self._state.dom_tree = dom_data.get("markdown", "")
            self._state.dom_elements = dom_data.get("elements", [])
        except Exception as e:
            logger.warning("DOM refresh failed: %s", e)

    def _save_event_log(self) -> None:
        """Persist the event log to disk."""
        try:
            log_path = self._persistence_dir / "event_log.json"
            self._event_log.save(log_path)
            logger.info("Event log saved: %s (%d events)", log_path, len(self._event_log))
        except Exception as e:
            logger.warning("Event log save failed: %s", e)
