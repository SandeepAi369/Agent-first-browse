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
from app.browser_promoter.state import AgentState, BrowserConfig, BrowserMode, CampaignContext
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


def _resolve_browser_mode() -> BrowserMode:
    """Resolve initial browser mode based on BROWSER_STRATEGY config.

    The actual smart auto-detection happens in browser_runtime.py's
    _auto_resolve_browser_mode(). Here we just set the initial mode
    that the runtime will further resolve when strategy='auto'.
    """
    strategy = config.BROWSER_STRATEGY

    if strategy == "cdp":
        return BrowserMode.LOCAL_NATIVE_CDP
    if strategy == "playwright":
        return BrowserMode.ISOLATED_SERVER

    # strategy == "auto": let the runtime decide based on environment
    # Start with ISOLATED_SERVER — runtime will upgrade to CDP if appropriate
    return BrowserMode.ISOLATED_SERVER


async def main():
    initialize_persistence_database()

    env_name = _detect_environment()

    print("\n=======================================================")
    print("  AGENT FIRST IDE — CAMPAIGN STARTING")
    print("=======================================================")
    print(f"  Runtime:  {env_name}")
    print(f"  Strategy: {config.BROWSER_STRATEGY} (auto-detect platform)")
    print(f"  Input:    {config.INPUT_DRIVER} (Playwright native API)")
    print(f"  Brain:    {config.SUPERVISOR_LLM_MODEL}")
    print(f"  Eyes:     {config.WORKER_VLM_MODEL}")
    print(f"  DryRun:   {config.DRY_RUN_MODE}")
    print(f"  Auto:     {config.AUTONOMOUS_CONTINUATION}")
    print(f"  Human:    {config.HUMAN_ACTION_DELAY_MIN_SECONDS:.1f}s - {config.HUMAN_ACTION_DELAY_MAX_SECONDS:.1f}s")
    print("=======================================================\n")

    campaign = CampaignContext(
        campaign_id="real-task-test-playwright",
        campaign_name="Test Playwright Native Input Driver",
        objective=(
            "Open https://github.com. Find the search bar in the header, type 'Playwright Input Test' "
            "into it, and press Enter to search. Verify the search results page loads."
        ),
        target_platforms=["github"],
        session_id="recon-session-test",
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
