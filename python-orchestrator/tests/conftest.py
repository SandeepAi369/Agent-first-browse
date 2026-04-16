"""Shared pytest fixtures for Agent First IDE tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.browser_promoter.state import (
    AgentState,
    BrowserConfig,
    CampaignContext,
    HighLevelCommand,
)


@pytest.fixture
def campaign_context() -> CampaignContext:
    """Minimal valid campaign context for testing."""
    return CampaignContext(
        campaign_id="test-campaign-001",
        campaign_name="Test Campaign",
        objective="Validate orchestration pipeline in dry-run mode.",
        target_platforms=["reddit"],
        session_id="test-session-001",
    )


@pytest.fixture
def agent_state(campaign_context: CampaignContext) -> AgentState:
    """Minimal valid AgentState for testing."""
    return AgentState(
        campaign=campaign_context,
        browser_config=BrowserConfig(headless=True),
        dry_run_mode=True,
    )


@pytest.fixture
def high_level_command() -> HighLevelCommand:
    """Sample high-level command for testing."""
    return HighLevelCommand(
        action_type="reconnaissance",
        target_description="Find relevant developer communities",
        behavior_plan="Search Google for active communities, read discussion quality.",
        confidence=0.65,
        stealth_adjustments=["Randomize delays between 1-4 seconds."],
    )


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Temporary SQLite database path for isolated testing."""
    return tmp_path / "test_agent.db"
