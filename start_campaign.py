#!/usr/bin/env python3
"""
Agent First IDE — Campaign Launcher
Uses Playwright's own Chromium with a persistent profile.
Your logins are saved — log in once, it remembers forever.

Supports running natively on Windows or from WSL.
"""

from __future__ import annotations

import os
import platform
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
from app.browser_promoter.state import AgentState, BrowserConfig, CampaignContext
from app.logger import get_logger

logger = get_logger("campaign_launcher")


def _detect_environment() -> str:
    """Detect runtime environment for diagnostic display."""
    if platform.system() == "Windows":
        return "Windows Native"
    if "WSL_DISTRO_NAME" in os.environ:
        return f"WSL ({os.environ['WSL_DISTRO_NAME']})"
    if platform.system() == "Linux":
        return f"Linux ({platform.release().split('-')[0]})"
    return platform.system()


async def main():
    initialize_persistence_database()

    env_name = _detect_environment()

    print("\n=======================================================")
    print("  AGENT FIRST IDE — CAMPAIGN STARTING")
    print("=======================================================")
    print(f"  Runtime:  {env_name}")
    print(f"  Browser:  Playwright Chromium (persistent context)")
    print(f"  Brain:    {config.SUPERVISOR_LLM_MODEL}")
    print(f"  Eyes:     {config.WORKER_VLM_MODEL}")
    print(f"  DryRun:   {config.DRY_RUN_MODE}")
    print(f"  Auto:     {config.AUTONOMOUS_CONTINUATION}")
    print(f"  Human:    {config.HUMAN_ACTION_DELAY_MIN_SECONDS:.1f}s - {config.HUMAN_ACTION_DELAY_MAX_SECONDS:.1f}s")
    print("=======================================================\n")

    campaign = CampaignContext(
        campaign_id="test-wiki-001",
        campaign_name="Wikipedia Search Test",
        objective=(
            "TEST SIMPLE TASK:\n"
            "1. Navigate to https://en.wikipedia.org\n"
            "2. Find the main search bar and type a name like 'Alan Turing'.\n"
            "3. Click the search button or press enter.\n"
            "4. Verify that you have reached the article and read the title."
        ),
        target_platforms=["wikipedia"],
        session_id="test-wiki-session",
        github_username="None",
    )

    state = AgentState(
        campaign=campaign,
        thread_id=config.PERSISTENT_THREAD_ID or campaign.session_id,
        browser_config=BrowserConfig(headless=False),
        dry_run_mode=False,
        autonomous_continuation=config.AUTONOMOUS_CONTINUATION,
        max_cycles=15,
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
