#!/usr/bin/env python3
"""
Agent First IDE — Campaign Launcher
Uses Playwright's own Chromium with a persistent profile.
Your logins are saved — log in once, it remembers forever.
"""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path

# Load .env FIRST — before any module reads os.getenv()
from dotenv import load_dotenv
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

_ORCHESTRATOR_DIR = _PROJECT_ROOT / "python-orchestrator"
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

logger = get_logger("campaign_launcher")


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

    print("\n=======================================================")
    print("  AGENT FIRST IDE — CAMPAIGN STARTING")
    print("=======================================================")
    print(f"  Browser: {_resolve_browser_mode().value}")
    print(f"  Brain:   {config.SUPERVISOR_LLM_MODEL}")
    print(f"  Eyes:    {config.WORKER_VLM_MODEL}")
    print(f"  DryRun:  {config.DRY_RUN_MODE}")
    print(f"  Auto:    {config.AUTONOMOUS_CONTINUATION}")
    print(f"  Human:   {config.HUMAN_ACTION_DELAY_MIN_SECONDS:.1f}s - {config.HUMAN_ACTION_DELAY_MAX_SECONDS:.1f}s")
    print("=======================================================\n")

    campaign = CampaignContext(
        campaign_id="real-task-002",
        campaign_name="Live Community Recon Task",
        objective=(
            "Open Reddit and find one active software-engineering discussion where practical browser "
            "automation or scraping tips are being requested. Read the top context and prepare a short, helpful "
            "draft response without submitting it."
        ),
        target_platforms=["reddit"],
        session_id="recon-session-002",
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

    logger.info("Executing campaign: %s", campaign.campaign_name)
    try:
        async with build_graph_with_default_checkpointer() as graph:
            final_state = await graph.ainvoke(
                state.model_dump(mode="json"),
                config=build_run_config(
                    state=state,
                    run_name="campaign.start",
                    tags=["launcher", "real-task"],
                    metadata={"entrypoint": "start_campaign.py"},
                ),
            )

        print("\n=======================================================")
        print("  CAMPAIGN FINISHED")
        print("=======================================================")
        url = final_state.get("current_url", "N/A")
        cycles = final_state.get("cycle_count", "?")
        summary = final_state.get("current_scene_summary", "No summary.")
        print(f"  Final URL:    {url}")
        print(f"  Cycles used:  {cycles}/{config.MAX_CYCLES}")
        print(f"  Scene:        {summary[:300]}")

        if final_state.get("worker_last_confused"):
            print(f"\n  Worker issue: {final_state.get('worker_last_confusion_reason')}")

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await BrowserRuntime.shutdown()

    print("\n  Browser tab stays open. Press ENTER to exit.")
    try:
        input()
    except EOFError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
