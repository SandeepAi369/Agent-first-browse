from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.browser_promoter.nodes import (
    browser_controller_node,
    router_function,
    router_node,
    supervisor_node,
    worker_node,
)
from app.browser_promoter.state import AgentState


def _default_checkpoint_path() -> Path:
    """Return default sqlite checkpoint path and ensure parent directory exists."""
    workspace_root = Path(__file__).resolve().parents[3]
    persistence_dir = workspace_root / "persistence"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    return persistence_dir / "checkpoints.sqlite"


def build_graph(
    *,
    checkpointer: Any | None = None,
    enable_hitl_interrupt: bool = False,
) -> CompiledStateGraph:
    """Build and compile graph with optional checkpointer and HITL interrupt hooks."""
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("worker", worker_node)
    graph.add_node("browser_controller", browser_controller_node)
    graph.add_node("router", router_node)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "worker")

    # Worker output is routed through router logic for circuit-break safety.
    graph.add_edge("worker", "router")

    # Browser always loops back to Worker to observe result state.
    graph.add_edge("browser_controller", "worker")

    graph.add_conditional_edges(
        "router",
        router_function,
        {
            "browser_controller": "browser_controller",
            "supervisor": "supervisor",
            "end": END,
        },
    )

    interrupt_before = ["browser_controller"] if enable_hitl_interrupt else None
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )


@asynccontextmanager
async def build_graph_with_default_checkpointer(
    checkpoint_path: Path | None = None,
    *,
    enable_hitl_interrupt: bool = False,
) -> AsyncIterator[CompiledStateGraph]:
    """
    Build graph with AsyncSqliteSaver persisted to local sqlite checkpoint file.

    This helper manages saver lifecycle and yields a compiled graph bound to the
    provided or default checkpoint database.
    """
    db_path = checkpoint_path or _default_checkpoint_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        yield build_graph(
            checkpointer=checkpointer,
            enable_hitl_interrupt=enable_hitl_interrupt,
        )
