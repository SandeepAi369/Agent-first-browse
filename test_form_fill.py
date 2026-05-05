import asyncio
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

from app import config
from app.browser_promoter.state import AgentState, CampaignContext, BrowserConfig, HighLevelCommand
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from app.browser_promoter.nodes import (
    strategic_planner_node,
    tactical_selector_node,
    vision_tool_node,
    browser_controller_node,
    stuck_evaluator_node,
    auth_check_node,
    router_node,
    task_logging_node,
    housekeeping_node,
    router_function
)

# Force logging visible
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("test_form_fill")

async def run_form_fill():
    campaign = CampaignContext(
        campaign_id="test-form-fill-001",
        campaign_name="DemoQA Form Fill Test",
        objective="TEST FORM FILL TASK",
        target_platforms=["demoqa"],
        session_id="test-form-fill-session",
        github_username="None",
    )

    state = AgentState(
        campaign=campaign,
        thread_id="test-form-fill-session",
        browser_config=BrowserConfig(headless=False),
        dry_run_mode=False,
        autonomous_continuation=True,
        max_cycles=15,
        worker_confidence_threshold=0.4,
        high_level_command=HighLevelCommand(
            action_type="Form Fill",
            target_description="DemoQA Automation Practice Form at https://demoqa.com/automation-practice-form",
            draft_text="Name: John Doe, Email: john@example.com, Gender: Male, Mobile: 1234567890",
            behavior_plan="Navigate to the URL https://demoqa.com/automation-practice-form, find the form fields, type the draft text into the correct inputs, and click the Submit button.",
            confidence=0.9
        )
    )

    graph = StateGraph(AgentState)
    graph.add_node("strategic_planner", strategic_planner_node)
    graph.add_node("tactical_selector", tactical_selector_node)
    graph.add_node("vision_tool", vision_tool_node)
    graph.add_node("browser_controller", browser_controller_node)
    graph.add_node("stuck_evaluator", stuck_evaluator_node)
    graph.add_node("auth_check", auth_check_node)
    graph.add_node("router", router_node)
    graph.add_node("task_logging_node", task_logging_node)
    graph.add_node("housekeeping_node", housekeeping_node)

    # Start directly at the Swarm (Strategic Planner)
    graph.add_edge(START, "strategic_planner")
    graph.add_edge("strategic_planner", "tactical_selector")
    
    def collaborative_router(state: AgentState) -> str:
        if state.routing.vision_requested:
            return "vision_tool"
        return "stuck_evaluator"

    graph.add_conditional_edges("tactical_selector", collaborative_router, {
        "vision_tool": "vision_tool",
        "stuck_evaluator": "stuck_evaluator",
    })
    
    graph.add_edge("vision_tool", "strategic_planner")
    graph.add_edge("stuck_evaluator", "auth_check")
    graph.add_edge("auth_check", "router")
    graph.add_edge("browser_controller", "strategic_planner")
    
    graph.add_conditional_edges(
        "router",
        router_function,
        {
            "browser_controller": "browser_controller",
            "supervisor": "task_logging_node", # Skip supervisor, end test
            "end": "task_logging_node",
        },
    )
    graph.add_edge("task_logging_node", "housekeeping_node")
    graph.add_edge("housekeeping_node", END)

    compiled_graph = graph.compile(checkpointer=MemorySaver())

    logger.info("Starting form fill test directly with Swarm...")
    try:
        await compiled_graph.ainvoke(
            state.model_dump(mode="json"),
            config=RunnableConfig(
                configurable={"thread_id": "test-form-fill-session"},
                run_name="swarm.test"
            )
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

    logger.info("Test finished.")

if __name__ == "__main__":
    asyncio.run(run_form_fill())
