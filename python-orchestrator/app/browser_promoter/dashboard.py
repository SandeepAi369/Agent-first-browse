"""Terminal dashboard rendering for the browser promoter orchestration loop.

Extracted from nodes.py to keep the main graph nodes focused on orchestration
logic rather than display concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.browser_promoter.database import count_locked_target_communities
from app.logger import get_logger

if TYPE_CHECKING:
    from app.browser_promoter.state import AgentState

logger = get_logger(__name__)


def print_terminal_dashboard(state: AgentState, cycle_value: int, hard_max: int) -> None:
    """Render concise real-time orchestration status in terminal."""
    mode = "Recon" if is_recon_mode(state) else "Engage"
    platform = resolve_platform_name(state).capitalize()
    action = resolve_dashboard_action(state, mode)

    try:
        locked_targets = count_locked_target_communities()
    except Exception:
        locked_targets = 0

    logger.info(
        "Cycle %d/%d │ Mode: %s │ Platform: %s │ Action: %s │ Locked Targets: %d",
        cycle_value,
        hard_max,
        mode,
        platform,
        action,
        locked_targets,
    )


def is_recon_mode(state: AgentState) -> bool:
    """Check if the current high-level command is a reconnaissance action."""
    if state.high_level_command is None:
        return False
    action_type = state.high_level_command.action_type.lower()
    return action_type in {"reconnaissance", "recon", "discovery", "community_recon"}


def resolve_platform_name(state: AgentState) -> str:
    """Infer target platform from campaign context or current URL."""
    if state.campaign.target_platforms:
        return state.campaign.target_platforms[0]

    url = state.current_url.lower()
    if "reddit.com" in url:
        return "reddit"
    if "github.com" in url:
        return "github"
    if "x.com" in url or "twitter.com" in url:
        return "x"
    return "web"


def resolve_dashboard_action(state: AgentState, mode: str) -> str:
    """Determine the current action label for dashboard display."""
    if state.worker_action_queue:
        if state.worker_action_queue[0].action == "manual_intervention_required":
            return "Manual Pause"
        return state.worker_action_queue[0].action.replace("_", " ").title()

    if mode == "Recon":
        return "Searching"

    if state.high_level_command is not None:
        return state.high_level_command.action_type.replace("_", " ").title()

    return "Idle"


def pause_for_manual_intervention(state: AgentState, cycle_value: int, hard_max: int) -> None:
    """Pause graph execution so a human can complete login/captcha manually."""
    platform = resolve_platform_name(state).upper()
    logger.warning("=" * 86)
    logger.warning(
        "HITL PAUSE │ Cycle %d/%d │ Platform: %s │ Action: Manual Intervention Required",
        cycle_value,
        hard_max,
        platform,
    )
    logger.warning("Reason: Login wall, Gmail auth, captcha, or MFA requires manual completion.")
    logger.warning("Please complete the required manual browser actions, then press ENTER to resume.")
    logger.warning("=" * 86)
    try:
        input("[HITL] Press ENTER when manual intervention is complete: ")
    except EOFError:
        logger.info("No interactive stdin detected; continuing without blocking.")
