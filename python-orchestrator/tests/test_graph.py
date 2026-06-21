"""Tests for LangGraph compilation and topology."""

from __future__ import annotations

from app.browser_promoter.graph import build_graph
from app.browser_promoter.supervisor_subgraph import build_supervisor_subgraph


class TestBuildGraph:
    """Validate main orchestration graph compilation."""

    def test_compiles_without_error(self) -> None:
        graph = build_graph()
        assert graph is not None

    def test_has_expected_nodes(self) -> None:
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        # LangGraph adds __start__ and __end__ pseudo-nodes
        assert "supervisor" in node_names
        assert "reasoning_agent" in node_names
        assert "browser_controller" in node_names
        assert "router" in node_names

    def test_compiles_with_hitl_interrupt(self) -> None:
        graph = build_graph(enable_hitl_interrupt=True)
        assert graph is not None


class TestBuildSupervisorSubgraph:
    """Validate supervisor subgraph compilation."""

    def test_compiles_without_error(self) -> None:
        subgraph = build_supervisor_subgraph()
        assert subgraph is not None

    def test_has_expected_nodes(self) -> None:
        subgraph = build_supervisor_subgraph()
        node_names = set(subgraph.get_graph().nodes.keys())
        assert "context_analyzer" in node_names
        assert "strategy_planner" in node_names
        assert "strategy_challenger" in node_names
        assert "copywriter" in node_names
        assert "risk_stealth_assessor" in node_names
        assert "final_merge" in node_names
