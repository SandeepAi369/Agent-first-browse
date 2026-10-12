"""Tests for state models and data structures."""

from __future__ import annotations

from app.browser_promoter.state import (
    AgentState,
    BrowserAction,
    BrowserConfig,
    BrowserMode,
    CampaignContext,
    HighLevelCommand,
    ScreenshotFrame,
    WorkerFeedback,
)


class TestCampaignContext:
    """Validate CampaignContext schema constraints."""

    def test_valid_creation(self, campaign_context: CampaignContext) -> None:
        assert campaign_context.campaign_id == "test-campaign-001"
        assert campaign_context.campaign_name == "Test Campaign"
        assert "reddit" in campaign_context.target_platforms

    def test_rejects_empty_campaign_id(self) -> None:
        import pytest

        with pytest.raises(Exception):
            CampaignContext(
                campaign_id="",
                campaign_name="Test",
                objective="Test",
                session_id="s1",
            )


class TestBrowserAction:
    """Validate BrowserAction schema for all supported action types."""

    def test_goto_action(self) -> None:
        action = BrowserAction(action="goto", url="https://reddit.com")
        assert action.action == "goto"
        assert action.url == "https://reddit.com"

    def test_click_action_with_selector(self) -> None:
        action = BrowserAction(action="click", selector="#submit-btn")
        assert action.action == "click"
        assert action.selector == "#submit-btn"

    def test_click_action_with_coordinates(self) -> None:
        action = BrowserAction(action="click", x=100.0, y=200.0)
        assert action.x == 100.0
        assert action.y == 200.0

    def test_type_action(self) -> None:
        action = BrowserAction(action="type", selector="input[name='q']", text="hello world")
        assert action.text == "hello world"
        assert action.clear_before_type is True

    def test_scroll_action(self) -> None:
        action = BrowserAction(action="scroll", delta_y=500)
        assert action.delta_y == 500

    def test_wait_action(self) -> None:
        action = BrowserAction(action="wait", wait_ms=2000)
        assert action.wait_ms == 2000

    def test_screenshot_action(self) -> None:
        action = BrowserAction(action="screenshot")
        assert action.action == "screenshot"

    def test_manual_intervention_action(self) -> None:
        action = BrowserAction(action="manual_intervention_required")
        assert action.action == "manual_intervention_required"

    def test_auto_generated_action_id(self) -> None:
        action = BrowserAction(action="wait")
        assert action.action_id.startswith("act_")
        assert len(action.action_id) == 16  # "act_" + 12 hex chars


class TestBrowserConfig:
    """Validate BrowserConfig defaults and modes."""

    def test_defaults(self) -> None:
        config = BrowserConfig()
        assert config.mode == BrowserMode.PERSISTENT_CONTEXT
        assert config.headless is False
        assert config.viewport_width == 1440

    def test_persistent_context_mode(self) -> None:
        config = BrowserConfig(
            mode=BrowserMode.PERSISTENT_CONTEXT,
            headless=True,
        )
        assert config.mode == BrowserMode.PERSISTENT_CONTEXT
        assert config.headless is True


class TestAgentState:
    """Validate AgentState construction and rolling window methods."""

    def test_minimal_creation(self, agent_state: AgentState) -> None:
        assert agent_state.campaign.campaign_id == "test-campaign-001"
        assert agent_state.dry_run_mode is True
        assert agent_state.should_continue is True
        assert agent_state.cycle_count == 0

    def test_push_screenshot_rolling_window(self, agent_state: AgentState) -> None:
        frames = [
            ScreenshotFrame(url=f"https://example.com/page{i}")
            for i in range(10)
        ]
        history = []
        for frame in frames:
            state_with_history = agent_state.model_copy(update={"screenshot_history": history})
            history = state_with_history.push_screenshot(frame)

        assert len(history) == agent_state.screenshot_window_size
        assert history[-1].url == "https://example.com/page9"

    def test_push_feedback_rolling_window(self, agent_state: AgentState) -> None:
        feedbacks = [
            WorkerFeedback(command_id=f"cmd_{i}", status="completed")
            for i in range(60)
        ]
        history = []
        for fb in feedbacks:
            state_with_history = agent_state.model_copy(update={"worker_feedback": history})
            history = state_with_history.push_feedback(fb)

        assert len(history) == agent_state.feedback_window_size

    def test_thread_id_auto_generated(self, agent_state: AgentState) -> None:
        assert agent_state.thread_id.startswith("thread_")

    def test_max_cycles_default(self, agent_state: AgentState) -> None:
        assert agent_state.max_cycles == 15

    def test_confidence_threshold_default(self, agent_state: AgentState) -> None:
        assert agent_state.worker_confidence_threshold == 0.4


class TestHighLevelCommand:
    """Validate HighLevelCommand schema."""

    def test_valid_creation(self, high_level_command: HighLevelCommand) -> None:
        assert high_level_command.action_type == "reconnaissance"
        assert high_level_command.confidence == 0.65
        assert len(high_level_command.stealth_adjustments) == 1

    def test_confidence_bounds(self) -> None:
        import pytest

        with pytest.raises(Exception):
            HighLevelCommand(
                action_type="engage",
                target_description="test",
                behavior_plan="test",
                confidence=1.5,  # exceeds max
            )
