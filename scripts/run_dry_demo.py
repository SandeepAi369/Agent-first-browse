#!/usr/bin/env python3
"""Quick dry-run demonstration of the Agent First IDE orchestration pipeline.

This script validates that:
1. All imports resolve correctly
2. State models construct without errors
3. LangGraph compiles the main graph + supervisor subgraph
4. SQLite persistence initializes and creates tables

No API keys or browser required — this is a pure structural validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the python-orchestrator package is on sys.path
_ORCHESTRATOR_DIR = Path(__file__).resolve().parents[1] / "python-orchestrator"
if str(_ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_DIR))

from app.browser_promoter.database import initialize_persistence_database
from app.browser_promoter.graph import build_graph
from app.browser_promoter.state import (
    AgentState,
    BrowserConfig,
    CampaignContext,
)
from app.browser_promoter.supervisor_subgraph import build_supervisor_subgraph
from app.logger import get_logger

logger = get_logger("dry_demo")


def main() -> None:
    logger.info("=" * 60)
    logger.info("Agent First IDE — Dry Run Validation")
    logger.info("=" * 60)

    # 1. State construction
    logger.info("[1/4] Constructing AgentState...")
    campaign = CampaignContext(
        campaign_id="demo-001",
        campaign_name="Dry Run Demo",
        objective="Validate pipeline without API keys or browser.",
        target_platforms=["reddit"],
        session_id="demo-session-001",
    )
    state = AgentState(
        campaign=campaign,
        browser_config=BrowserConfig(headless=True),
        dry_run_mode=True,
    )
    logger.info("  ✓ AgentState created (thread=%s, max_cycles=%d)", state.thread_id, state.max_cycles)

    # 2. Graph compilation
    logger.info("[2/4] Compiling main LangGraph...")
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    logger.info("  ✓ Main graph compiled — nodes: %s", sorted(node_names))

    # 3. Supervisor subgraph compilation
    logger.info("[3/4] Compiling Supervisor subgraph...")
    subgraph = build_supervisor_subgraph()
    sub_nodes = set(subgraph.get_graph().nodes.keys())
    logger.info("  ✓ Supervisor subgraph compiled — nodes: %s", sorted(sub_nodes))

    # 4. Database initialization
    logger.info("[4/4] Initializing SQLite persistence...")
    db_path = initialize_persistence_database()
    logger.info("  ✓ Database initialized at: %s", db_path)

    logger.info("=" * 60)
    logger.info("All validations passed. Pipeline is structurally sound.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
