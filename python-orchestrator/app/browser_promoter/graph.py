from __future__ import annotations

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


def build_graph() -> CompiledStateGraph:
    """Build and compile the Step-5 graph with strict router-controlled flow."""
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

    return graph.compile()
