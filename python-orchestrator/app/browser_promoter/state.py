from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class CampaignContext(BaseModel):
    """Stable campaign-level context shared across all graph nodes."""

    campaign_id: str = Field(min_length=1)
    campaign_name: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    target_platforms: list[Literal["x", "reddit", "github"]] = Field(default_factory=list)
    session_id: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class SupervisorCommand(BaseModel):
    """High-level command emitted by the Supervisor model."""

    command_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    priority: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


class HighLevelCommand(BaseModel):
    """Final structured command produced by the Supervisor subgraph."""

    action_type: str = Field(min_length=1)
    target_description: str = Field(min_length=1)
    draft_text: str | None = None
    behavior_plan: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    stealth_adjustments: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class BrowserAction(BaseModel):
    """Executable browser action prepared by the Worker model."""

    action_id: str = Field(default_factory=lambda: f"act_{uuid4().hex[:12]}", min_length=1)
    action: Literal["goto", "scroll", "click", "type", "wait", "screenshot"]
    url: str = ""
    selector: str = ""
    text: str = ""
    x: float | None = None
    y: float | None = None
    delta_y: int = 700
    wait_ms: int = Field(default=400, ge=0, le=120000)
    timeout_ms: int = Field(default=15000, ge=1, le=120000)
    typing_delay_ms: int = Field(default=45, ge=0, le=500)
    clear_before_type: bool = True

    model_config = ConfigDict(extra="forbid")


class InteractiveRegion(BaseModel):
    """Interactive UI element with spatial metadata for vision grounding."""

    tag: str = ""
    role: str = ""
    text: str = ""
    selector: str = ""
    x: int
    y: int
    width: int
    height: int

    model_config = ConfigDict(extra="forbid")


class WorkerFeedback(BaseModel):
    """Execution result from browser controller back to Worker/Supervisor."""

    command_id: str = Field(min_length=1)
    status: Literal["completed", "failed"]
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RoutingFlags(BaseModel):
    """Routing controls for Supervisor -> Worker <-> Browser -> Router loop."""

    next_hop: Literal["worker", "browser_controller", "supervisor", "router", "end"] = "worker"
    requires_browser_action: bool = False
    requires_supervisor_review: bool = False
    stop_requested: bool = False

    model_config = ConfigDict(extra="forbid")


class ScreenshotFrame(BaseModel):
    """Compressed screenshot + interaction metadata payload."""

    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    url: str = ""
    screenshot_base64: str = ""
    screenshot_encoding: str = "zlib+base64:jpeg"
    original_bytes: int = 0
    compressed_bytes: int = 0
    accessibility_tree: dict[str, Any] = Field(default_factory=dict)
    interactive_regions: list[InteractiveRegion] = Field(default_factory=list)
    scene_summary: str = ""

    model_config = ConfigDict(extra="forbid")


class AgentState(BaseModel):
    """
    Canonical state for the dual-model browser promoter loop.

    Includes campaign context, rolling screenshot history, Supervisor commands,
    Worker feedback, and routing flags.
    """

    campaign: CampaignContext
    high_level_command: HighLevelCommand | None = None

    supervisor_commands: list[SupervisorCommand] = Field(default_factory=list)
    worker_action_queue: list[BrowserAction] = Field(default_factory=list)
    worker_feedback: list[WorkerFeedback] = Field(default_factory=list)

    screenshot_history: list[ScreenshotFrame] = Field(default_factory=list)
    screenshot_window_size: int = Field(default=6, ge=1, le=64)
    feedback_window_size: int = Field(default=48, ge=1, le=256)

    current_url: str = ""
    current_screenshot_base64: str = ""
    current_screenshot_encoding: str = "none"
    current_accessibility_tree: dict[str, Any] = Field(default_factory=dict)
    current_interactive_regions: list[InteractiveRegion] = Field(default_factory=list)
    current_scene_summary: str = ""

    worker_last_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    worker_last_confused: bool = False
    worker_last_confusion_reason: str = ""

    should_continue: bool = True
    cycle_count: int = Field(default=0, ge=0)
    max_cycles: int = Field(default=15, ge=1, le=1000)
    worker_confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    routing: RoutingFlags = Field(default_factory=RoutingFlags)
    ephemeral: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def push_screenshot(self, frame: ScreenshotFrame) -> list[ScreenshotFrame]:
        """Return screenshot history with bounded rolling window."""
        history = [*self.screenshot_history, frame]
        return history[-self.screenshot_window_size :]

    def push_feedback(self, feedback: WorkerFeedback) -> list[WorkerFeedback]:
        """Return feedback history with bounded rolling window."""
        feedback_list = [*self.worker_feedback, feedback]
        return feedback_list[-self.feedback_window_size :]
