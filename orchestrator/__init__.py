"""Multi-Agent Orchestrator — The Brain of Agent First IDE.

A 3-tier architecture for executing ANY browser or system task:
  Tier 1: CEO        — Master Planner + DAG Builder
  Tier 2: Spawner    — Dynamic Agent Factory
  Tier 3: Executor   — Runtime Engine + Critic Loop

Inspired by OpenHands (event streams), Magentic-One (dual ledgers),
and LangGraph (stateful routing with conditional edges).
"""

from orchestrator.actions import ActionType, TaskNode, TaskDAG
from orchestrator.event_log import Event, EventLog
from orchestrator.state import OrchestratorState
from orchestrator.ceo import CEO
from orchestrator.spawner import Spawner
from orchestrator.executor import Executor
from orchestrator.critic import Critic

__all__ = [
    "ActionType", "TaskNode", "TaskDAG",
    "Event", "EventLog",
    "OrchestratorState",
    "CEO", "Spawner", "Executor", "Critic",
]
