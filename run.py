"""Unified CLI Entry Point — run.py

Replaces all old run_*.py scripts with a single, generalized interface.
Accepts ANY objective as a command-line argument or interactive prompt.

Usage:
  python run.py "Find the top 5 trending repos on GitHub and save them"
  python run.py "Post this article on Dev.to: ..."
  python run.py "Go to twitter.com and tweet: Hello World"
  python run.py --interactive
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "python-orchestrator"))

from app.browser_promoter.cdp_stealth_launcher import (
    STEALTH_INIT_SCRIPT,
    STEALTH_LAUNCH_ARGS,
    STEALTH_USER_AGENT,
    apply_page_stealth,
    get_random_viewport,
    VISUAL_CURSOR_INIT_SCRIPT,
)
from app.browser_promoter.browser_warmup import run_warmup, extract_target_url_from_objective
from app.logger import get_logger
from model_registry import ModelRegistry

import dom_parser

logger = get_logger("run")

# ═══════════════════════════════════════════════════════════════════════════════
#  Browser Launcher (reused from advanced_agent.py — KEPT infrastructure)
# ═══════════════════════════════════════════════════════════════════════════════

async def _launch_browser():
    """Launch a stealth Playwright browser with persistent profile."""
    from playwright.async_api import async_playwright

    profile_dir = Path(__file__).parent / "persistence" / "browser_sessions" / "agent_main"
    profile_dir.mkdir(parents=True, exist_ok=True)

    vp = get_random_viewport()
    pw = await async_playwright().start()

    context = await pw.chromium.launch_persistent_context(
        str(profile_dir),
        headless=True,
        viewport=vp,
        user_agent=STEALTH_USER_AGENT,
        args=STEALTH_LAUNCH_ARGS + dom_parser.TLS_STEALTH_ARGS,
        ignore_https_errors=True,
        bypass_csp=True,
        java_script_enabled=True,
        locale="en-US",
        timezone_id="America/New_York",
    )

    # Install stealth + shadow piercer
    await context.add_init_script(STEALTH_INIT_SCRIPT)
    await context.add_init_script(VISUAL_CURSOR_INIT_SCRIPT)
    await dom_parser.install_shadow_piercer(context)

    page = context.pages[0] if context.pages else await context.new_page()
    await apply_page_stealth(page)

    logger.info("Browser launched (viewport: %dx%d)", vp["width"], vp["height"])
    return context, page


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Orchestration Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

async def execute(objective: str) -> None:
    """Execute any objective through the Multi-Agent Orchestrator."""
    from orchestrator.ceo import CEO
    from orchestrator.executor import Executor

    # ── Launch Browser ──
    context, page = await _launch_browser()

    try:
        # ── Human Warm-Up ──
        target_hint = extract_target_url_from_objective(objective)
        try:
            await run_warmup(page, target_url=target_hint)
        except Exception as e:
            logger.warning("Warm-up failed (non-fatal): %s", e)

        # ── Initialize Infrastructure ──
        registry = ModelRegistry.get_instance()
        from app.browser_promoter.worker_planner import ReasoningAgent
        reasoning_agent = ReasoningAgent()

        failover_chain = reasoning_agent.get_failover_chain()
        chain_names = reasoning_agent.get_chain_names()

        if not failover_chain:
            logger.error("No LLM clients available. Check .env API keys.")
            return

        logger.info(
            "LLM Chain (%d models): %s",
            len(failover_chain), " → ".join(chain_names),
        )

        # ── Create Orchestrator Components ──
        ceo = CEO(
            failover_chain=failover_chain,
            health_tracker=registry.health,
            circuit_breaker=registry.breaker,
        )

        executor = Executor(
            page=page,
            ceo=ceo,
            persistence_dir=Path(__file__).parent / "persistence" / "orchestrator",
        )

        # ── Execute ──
        logger.info("Objective: %s", objective[:200])
        result = await executor.run(objective)

        if result.success:
            print(f"\n{'═' * 60}")
            print(f"  ✅  MISSION COMPLETE — {result.summary}")
            print(f"{'═' * 60}\n")

            if result.data:
                print("Collected data:")
                import json
                print(json.dumps(result.data, indent=2, default=str)[:3000])
        else:
            print(f"\n{'═' * 60}")
            print(f"  ⚠️  MISSION INCOMPLETE — {result.summary}")
            if result.state and result.state.is_paused:
                print(f"  🛑 Human intervention needed: {result.state.pause_reason}")
            print(f"{'═' * 60}\n")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error("Fatal error: %s", e)
        import traceback
        traceback.print_exc()
    finally:
        try:
            await context.close()
            logger.info("Browser closed. Session persisted.")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Orchestrator — Execute ANY browser task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py "Find the top 10 AI news on Hacker News"
  python run.py "Go to reddit.com/r/test and post: Hello World"
  python run.py "Log into dev.to and create a new article"
  python run.py --interactive
        """,
    )
    parser.add_argument(
        "objective",
        nargs="?",
        default=None,
        help="The task objective to execute",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (prompt for objectives)",
    )

    args = parser.parse_args()

    if args.interactive:
        print("Multi-Agent Orchestrator — Interactive Mode")
        print("Type your objective and press Enter. Type 'exit' to quit.\n")
        while True:
            try:
                objective = input("🎯 Objective: ").strip()
                if objective.lower() in ("exit", "quit", "q"):
                    break
                if not objective:
                    continue
                asyncio.run(execute(objective))
            except (KeyboardInterrupt, EOFError):
                break
    elif args.objective:
        asyncio.run(execute(args.objective))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
