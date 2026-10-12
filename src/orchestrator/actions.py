"""Unified Action Definitions & Task DAG Schema.

Every possible action the system can take is defined here as a typed,
serializable Pydantic model. The CEO produces TaskDAGs made of TaskNodes;
the Spawner reads them; the Executor runs them.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
#  Action Types — The universal vocabulary of the system
# ═══════════════════════════════════════════════════════════════════════════════

class ActionType(str, Enum):
    """Every atomic action the orchestrator can perform."""
    NAVIGATE    = "navigate"       # Go to a URL
    CLICK       = "click"          # Click an element by coordinates or selector
    TYPE        = "type"           # Type text into a focused/targeted element
    SCROLL      = "scroll"         # Scroll the page
    EXTRACT     = "extract"        # Extract structured data from the DOM
    EXECUTE_JS  = "execute_js"     # Run arbitrary JavaScript on the page
    WAIT        = "wait"           # Wait for a condition or fixed delay
    SCREENSHOT  = "screenshot"     # Capture current page state
    DONE        = "done"           # Signal task completion


# ═══════════════════════════════════════════════════════════════════════════════
#  Task Node — A single unit of work in the DAG
# ═══════════════════════════════════════════════════════════════════════════════

class NodeStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    SKIPPED  = "skipped"


class TaskNode(BaseModel):
    """A single node in the Task DAG.

    Each node represents one atomic operation with a specific skill.
    Nodes can depend on other nodes (via `dependencies`), forming
    the directed acyclic graph that the Executor walks.
    """
    id: str = Field(default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}")
    description: str = Field(..., description="Human-readable description of what this step does")
    action: ActionType = Field(..., description="The action type to execute")
    params: dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters")
    dependencies: list[str] = Field(default_factory=list, description="Node IDs that must complete first")
    status: NodeStatus = Field(default=NodeStatus.PENDING)
    result: Any = Field(default=None, description="Output from execution")
    error: str | None = Field(default=None, description="Error message if failed")
    retries: int = Field(default=0, description="Number of times this node has been retried")

    class Config:
        use_enum_values = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Task DAG — The full execution plan
# ═══════════════════════════════════════════════════════════════════════════════

class TaskDAG(BaseModel):
    """Directed Acyclic Graph representing a complete execution plan.

    Built by the CEO from a raw user objective. Walked by the Executor
    in topological order. Can be replanned by the CEO when nodes fail.
    """
    id: str = Field(default_factory=lambda: f"dag_{uuid.uuid4().hex[:8]}")
    goal: str = Field(..., description="The original user objective")
    nodes: dict[str, TaskNode] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_node(self, node: TaskNode) -> None:
        """Add a node to the DAG."""
        self.nodes[node.id] = node

    def topological_order(self) -> list[TaskNode]:
        """Return nodes in dependency-respecting execution order.

        Uses Kahn's algorithm for topological sorting. Nodes with no
        dependencies execute first; nodes whose deps are all satisfied
        execute next, and so on.
        """
        # Build in-degree map
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.dependencies:
                if dep in self.nodes:
                    in_degree[nid] += 1

        # Start with zero-dependency nodes
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        ordered: list[TaskNode] = []

        while queue:
            # Sort for deterministic ordering
            queue.sort()
            current = queue.pop(0)
            ordered.append(self.nodes[current])

            # Reduce in-degree for dependents
            for nid, node in self.nodes.items():
                if current in node.dependencies:
                    in_degree[nid] -= 1
                    if in_degree[nid] == 0:
                        queue.append(nid)

        return ordered

    def get_ready_nodes(self) -> list[TaskNode]:
        """Return nodes whose dependencies are all DONE and that are still PENDING."""
        done_ids = {nid for nid, n in self.nodes.items() if n.status == NodeStatus.DONE}
        ready = []
        for nid, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
            if all(dep in done_ids for dep in node.dependencies):
                ready.append(node)
        return ready

    def is_complete(self) -> bool:
        """True if all nodes are DONE or SKIPPED."""
        return all(
            n.status in (NodeStatus.DONE, NodeStatus.SKIPPED)
            for n in self.nodes.values()
        )

    def has_failures(self) -> bool:
        """True if any node is FAILED."""
        return any(n.status == NodeStatus.FAILED for n in self.nodes.values())

    def summary(self) -> str:
        """Human-readable summary of DAG state."""
        lines = [f"DAG: {self.goal}"]
        for node in self.topological_order():
            status_icon = {
                "pending": "⬜", "running": "🔄",
                "done": "✅", "failed": "❌", "skipped": "⏭️",
            }.get(node.status, "❓")
            deps = f" (after: {', '.join(node.dependencies)})" if node.dependencies else ""
            lines.append(f"  {status_icon} [{node.id}] {node.action}: {node.description}{deps}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  CEO Output Schema — What the LLM must produce when planning
# ═══════════════════════════════════════════════════════════════════════════════

class PlannedStep(BaseModel):
    """A single step in the CEO's plan (LLM structured output)."""
    description: str = Field(..., description="What this step accomplishes")
    action: str = Field(..., description="Action type: navigate, click, type, scroll, extract, execute_js, wait, screenshot, done")
    params: dict[str, Any] = Field(default_factory=dict, description="Action parameters (url, selector, text, etc.)")
    depends_on: list[int] = Field(default_factory=list, description="Zero-indexed step numbers this depends on")


class CEOPlan(BaseModel):
    """The CEO's full plan — structured output from the planning LLM."""
    task_type: str = Field(..., description="Classification: browsing, scraping, posting, research, login, automation, other")
    reasoning: str = Field(..., description="Why this plan structure was chosen")
    steps: list[PlannedStep] = Field(..., description="Ordered list of execution steps")
