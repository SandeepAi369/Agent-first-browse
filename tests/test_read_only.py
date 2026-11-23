"""Offline validation tests for the read-only browsing mission script."""

from __future__ import annotations

import asyncio
import inspect

from advanced_agent import run_agent
from execution_safety import is_domain_allowed

READ_ONLY_OBJECTIVE = (
    "\nMISSION: READ-ONLY HUMAN BROWSING TEST\n\n"
    "TARGET PLATFORM: https://www.reddit.com/r/rust\n"
    "OBJECTIVE: Navigate to the subreddit, scroll around to simulate a human reading, "
    "and then finish.\n\n"
    "CRITICAL RULES FOR THIS RUN:\n"
    '1. STRICTLY READ-ONLY: DO NOT click "Create Post", DO NOT type anything, '
    "DO NOT submit any forms, and DO NOT log in if prompted.\n"
    "2. HUMAN BEHAVIOR: Just navigate to the URL, scroll down the page 2 or 3 times "
    'to simulate reading the feed, and then output action_type="done".\n'
)


def test_read_only_objective_contains_strict_non_mutation_rules():
    assert "STRICTLY READ-ONLY" in READ_ONLY_OBJECTIVE
    assert "https://www.reddit.com/r/rust" in READ_ONLY_OBJECTIVE
    assert 'action_type="done"' in READ_ONLY_OBJECTIVE


def test_read_only_target_domain_allowed_and_runner_is_async():
    assert is_domain_allowed("https://www.reddit.com/r/rust") is True
    assert inspect.iscoroutinefunction(run_agent)


if __name__ == "__main__":
    print("Starting Read-Only Browsing Test on Reddit...")
    asyncio.run(run_agent(READ_ONLY_OBJECTIVE))
