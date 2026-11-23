"""Offline graph & state validation tests for the legacy form-fill workflow."""

from __future__ import annotations

import asyncio
import logging
import sys

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.browser_promoter.nodes import (
    auth_check_node,
    browser_controller_node,
    housekeeping_node,
    reasoning_agent_node,
    router_function,
    router_node,
    stuck_evaluator_node,
    task_logging_node,
    vision_agent_node,
)
from app.browser_promoter.state import (
    AgentState,
    BrowserConfig,
    CampaignContext,
    HighLevelCommand,
)

logger = logging.getLogger("test_form_fill")


def _build_form_fill_state() -> AgentState:
    campaign = CampaignContext(
        campaign_id="test-form-fill-001",
        campaign_name="DemoQA Form Fill Test",
        objective="TEST FORM FILL TASK",
        target_platforms=["demoqa"],
        session_id="test-form-fill-session",
        github_username="None",
    )
    return AgentState(
        campaign=campaign,
        thread_id="test-form-fill-session",
        browser_config=BrowserConfig(headless=True),
        dry_run_mode=True,
        autonomous_continuation=True,
        max_cycles=15,
        worker_confidence_threshold=0.4,
        high_level_command=HighLevelCommand(
            action_type="Form Fill",
            target_description="DemoQA Automation Practice Form at https://demoqa.com/automation-practice-form",
            draft_text="Name: John Doe, Email: john@example.com, Gender: Male, Mobile: 1234567890",
            behavior_plan=(
                "Navigate to https://demoqa.com/automation-practice-form, fill inputs, "
                "and submit."
            ),
            confidence=0.9,
        ),
    )


def _build_form_fill_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("vision_agent", vision_agent_node)
    graph.add_node("reasoning_agent", reasoning_agent_node)
    graph.add_node("browser_controller", browser_controller_node)
    graph.add_node("stuck_evaluator", stuck_evaluator_node)
    graph.add_node("auth_check", auth_check_node)
    graph.add_node("router", router_node)
    graph.add_node("task_logging_node", task_logging_node)
    graph.add_node("housekeeping_node", housekeeping_node)

    graph.add_edge(START, "vision_agent")
    graph.add_edge("vision_agent", "reasoning_agent")
    graph.add_edge("reasoning_agent", "stuck_evaluator")
    graph.add_edge("stuck_evaluator", "auth_check")
    graph.add_edge("auth_check", "router")
    graph.add_edge("browser_controller", "vision_agent")
    graph.add_conditional_edges(
        "router",
        router_function,
        {
            "browser_controller": "browser_controller",
            "supervisor": "task_logging_node",
            "end": "task_logging_node",
        },
    )
    graph.add_edge("task_logging_node", "housekeeping_node")
    graph.add_edge("housekeeping_node", END)
    return graph


def test_form_fill_state_serializes_cleanly():
    state = _build_form_fill_state()
    dumped = state.model_dump(mode="json")
    assert dumped["campaign"]["campaign_id"] == "test-form-fill-001"
    assert dumped["high_level_command"]["action_type"] == "Form Fill"
    assert dumped["high_level_command"]["confidence"] == 0.9


def test_form_fill_graph_compiles_with_memory_checkpointer():
    graph = _build_form_fill_graph()
    compiled = graph.compile(checkpointer=MemorySaver())
    assert compiled is not None
    assert {
        "vision_agent",
        "reasoning_agent",
        "browser_controller",
        "router",
        "task_logging_node",
        "housekeeping_node",
    }.issubset(set(graph.nodes.keys()))


async def run_form_fill() -> None:
    """Manual live runner (not invoked during offline pytest runs)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    state = _build_form_fill_state()
    compiled = _build_form_fill_graph().compile(checkpointer=MemorySaver())
    await compiled.ainvoke(
        state.model_dump(mode="json"),
        config=RunnableConfig(
            configurable={"thread_id": "test-form-fill-session"},
            run_name="swarm.test",
        ),
    )


if __name__ == "__main__":
    asyncio.run(run_form_fill())
