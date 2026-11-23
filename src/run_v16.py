"""True Brain v16.0 — New Entry Point.

This is the replacement for run.py that uses the LangGraph StateGraph
architecture instead of the monolithic advanced_agent.py loop.

Usage:
    python src/run_v16.py run "Navigate to https://example.com and click Login"
    python src/run_v16.py login    # Manual login mode (unchanged)

Both entry points coexist:
    run.py      → Old monolithic loop (advanced_agent.py)
    run_v16.py  → New LangGraph brain (brain_graph.py)
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.logger import get_logger

logger = get_logger("run_v16")


def main():
    parser = argparse.ArgumentParser(
        description="True Brain v16.0 — LangGraph-based Browser Agent"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("login", help="Open browser for manual login.")
    run_parser = sub.add_parser("run", help="Run the brain with a task objective.")
    run_parser.add_argument(
        "objective",
        nargs="?",
        default=(
            "1. Go to https://the-internet.herokuapp.com/login\n"
            "2. Type 'tomsmith' into the username field.\n"
            "3. Type 'SuperSecretPassword!' into the password field.\n"
            "4. Click the Login button.\n"
            "5. Wait to see the secure area success message and then finish."
        ),
        help="The task objective for the agent.",
    )
    run_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode.",
    )
    run_parser.add_argument(
        "--mode",
        choices=["legacy", "v2"],
        default=None,
        help="Browser engine mode: 'legacy' (frozen baseline) or 'v2' (2026 upgraded engine).",
    )

    args = parser.parse_args()

    if args.command == "login":
        from advanced_agent import manual_login_mode
        asyncio.run(manual_login_mode())
    elif args.command == "run":
        from brain_graph import run_brain
        from v2_engine import get_engine_mode, set_engine_mode
        selected_mode = set_engine_mode(args.mode) if args.mode else get_engine_mode()
        logger.info("🧠 True Brain — LangGraph Architecture [mode=%s]", selected_mode)
        logger.info("Objective: %s", args.objective[:200])
        try:
            asyncio.run(run_brain(
                args.objective,
                headless=getattr(args, "headless", False),
                mode=selected_mode,
            ))
        except (KeyboardInterrupt, asyncio.CancelledError):
            # run_brain already closed the browser and stopped the driver.
            logger.warning("Interrupted by user — exiting.")
            sys.exit(130)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
