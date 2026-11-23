"""Smoke tests for True Brain v16.0 core modules and LangGraph assembly."""

from __future__ import annotations

from brain_graph import build_brain_graph
from brain_state import BrainState, ProposedAction, StepRecord
from moe_router import route_to_worker, verdict_router
from workers.base_worker import WorkerAction, build_system_prompt


def test_brain_state_models_and_renderers():
    state = BrainState(objective="Test import chain")
    assert len(BrainState.model_fields) > 0
    assert isinstance(state.get_plan_render(), str)
    assert isinstance(state.compress_history(), str)
    assert ProposedAction is not None
    assert StepRecord is not None


def test_moe_router_and_verdict_router_defaults():
    state = BrainState(objective="Test import chain")
    route = route_to_worker(state.model_dump())
    assert route in {"navigator", "interactor", "extractor", "done_check", "recovery", "finalize"}
    assert verdict_router({"overwatch_verdict": "pass"}) == "commit"


def test_worker_prompts_and_schema_available():
    nav_prompt = build_system_prompt("navigator", "test plan", "test facts")
    int_prompt = build_system_prompt("interactor")
    ext_prompt = build_system_prompt("extractor")
    assert len(nav_prompt) > 50
    assert len(int_prompt) > 50
    assert len(ext_prompt) > 50
    assert "action_type" in WorkerAction.model_fields


def test_build_brain_graph_registers_core_nodes():
    graph = build_brain_graph()
    expected = {
        "goal_compiler",
        "planner",
        "perceive",
        "router",
        "navigator",
        "interactor",
        "extractor",
        "overwatch",
        "done_check",
        "commit",
        "retry",
        "rollback",
        "recovery",
        "finalize",
    }
    assert expected.issubset(set(graph.nodes.keys()))
