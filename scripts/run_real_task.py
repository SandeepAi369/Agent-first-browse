#!/usr/bin/env python3
import sys
from pathlib import Path
import asyncio

_ORCHESTRATOR_DIR = Path(__file__).resolve().parents[1] / "python-orchestrator"
if str(_ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_DIR))

from app.observability import build_run_config, configure_langsmith
from app import config

# Configure tracing before importing graph/LLM modules.
configure_langsmith()

from app.browser_promoter.database import initialize_persistence_database
from app.browser_promoter.graph import build_graph_with_default_checkpointer
from app.browser_promoter.browser_runtime import BrowserRuntime
from app.browser_promoter.state import AgentState, BrowserConfig, BrowserMode, CampaignContext
from app.logger import get_logger

logger = get_logger("run_real_task")


def _resolve_browser_mode() -> BrowserMode:
    if config.FORCE_VISIBLE_LOCAL_BROWSER:
        return BrowserMode.LOCAL_NATIVE_CDP

    try:
        return BrowserMode(config.BROWSER_MODE)
    except ValueError:
        logger.warning("Unsupported BROWSER_MODE '%s'. Falling back to ISOLATED_SERVER.", config.BROWSER_MODE)
        return BrowserMode.ISOLATED_SERVER

async def main():
    initialize_persistence_database()

    try:
        # Define a custom campaign asking to go to Reddit and test posting
        campaign = CampaignContext(
            campaign_id="trial-run-1",
            campaign_name="Reddit Posting Trial",
            objective=(
                "Open Reddit and find a relevant software-engineering discussion. "
                "Prepare a concise helpful draft comment based on what is visible, but do not submit it."
            ),
            target_platforms=["reddit"],
            session_id="trial-session-1",
        )

        state = AgentState(
            campaign=campaign,
            thread_id=config.PERSISTENT_THREAD_ID or campaign.session_id,
            browser_config=BrowserConfig(
                mode=_resolve_browser_mode(),
                headless=False if config.FORCE_VISIBLE_LOCAL_BROWSER else config.BROWSER_HEADLESS,
            ),
            dry_run_mode=config.DRY_RUN_MODE,
            autonomous_continuation=config.AUTONOMOUS_CONTINUATION,
            max_cycles=config.MAX_CYCLES,
            worker_confidence_threshold=config.WORKER_CONFIDENCE_THRESHOLD,
        )

        logger.info(
            "Visible mode=%s headless=%s cdp_endpoint=%s",
            state.browser_config.mode.value,
            state.browser_config.headless,
            state.browser_config.local_cdp_endpoint,
        )
        logger.info(
            "Human pacing active: action delay %.1fs-%.1fs | LLM gaps worker=%.1fs supervisor=%.1fs",
            config.HUMAN_ACTION_DELAY_MIN_SECONDS,
            config.HUMAN_ACTION_DELAY_MAX_SECONDS,
            config.WORKER_LLM_MIN_GAP_SECONDS,
            config.SUPERVISOR_LLM_MIN_GAP_SECONDS,
        )
        logger.info("Starting Trial Workflow on target: Reddit...")
        async with build_graph_with_default_checkpointer() as graph:
            final_state = await graph.ainvoke(
                state.model_dump(mode="json"),
                config=build_run_config(
                    state=state,
                    run_name="campaign.real_task",
                    tags=["launcher", "real-task", "visible-browser"],
                    metadata={"entrypoint": "scripts/run_real_task.py"},
                ),
            )
        logger.info("Task Finished.")
        if final_state.get("worker_last_confused"):
            logger.error(f"Worker Confusion Reason: {final_state.get('worker_last_confusion_reason')}")
    finally:
        await BrowserRuntime.shutdown()
    
if __name__ == "__main__":
    asyncio.run(main())
