"""True Brain v16.0 — New Entry Point.

This is the replacement for run.py that uses the LangGraph StateGraph
architecture instead of the monolithic advanced_agent.py loop.

Usage:
    python run_v16.py run "Navigate to https://example.com and click Login"
    python run_v16.py login    # Manual login mode (unchanged)

Both entry points coexist:
    run.py      → Old monolithic loop (advanced_agent.py)
    run_v16.py  → New LangGraph brain (brain_graph.py)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure app imports work
sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

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

    args = parser.parse_args()

    if args.command == "login":
        from advanced_agent import manual_login_mode
        asyncio.run(manual_login_mode())
    elif args.command == "run":
        from brain_graph import run_brain
        logger.info("🧠 True Brain v16.0 — LangGraph Architecture")
        logger.info("Objective: %s", args.objective[:200])
        asyncio.run(run_brain(args.objective, headless=getattr(args, "headless", False)))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
