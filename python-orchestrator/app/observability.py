"""LangSmith/LangChain observability helpers for orchestration runs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from app import config
from app.logger import get_logger

if TYPE_CHECKING:
    from app.browser_promoter.state import AgentState

logger = get_logger(__name__)
_LANGSMITH_CONFIGURED = False


def _disable_tracing_env() -> None:
    """Hard-disable tracing env flags for this process."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ.pop("LANGCHAIN_API_KEY", None)
    os.environ.pop("LANGSMITH_API_KEY", None)


def configure_langsmith() -> None:
    """Apply LangSmith environment configuration once per process."""
    global _LANGSMITH_CONFIGURED
    if _LANGSMITH_CONFIGURED:
        return

    if config.LANGCHAIN_TRACING_V2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if config.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
    if config.LANGCHAIN_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT
    if config.LANGCHAIN_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT

    if not config.LANGCHAIN_TRACING_V2:
        _disable_tracing_env()
        logger.info("LangSmith tracing disabled (LANGCHAIN_TRACING_V2=false).")
        _LANGSMITH_CONFIGURED = True
        return

    if not config.LANGCHAIN_API_KEY:
        _disable_tracing_env()
        logger.warning("LangSmith tracing enabled but LANGCHAIN_API_KEY is missing.")
        _LANGSMITH_CONFIGURED = True
        return

    # Validate auth once to avoid repeated 401 spam during long LangGraph loops.
    try:
        from langsmith import Client

        client_kwargs: dict[str, Any] = {
            "api_key": config.LANGCHAIN_API_KEY,
        }
        if config.LANGCHAIN_ENDPOINT:
            client_kwargs["api_url"] = config.LANGCHAIN_ENDPOINT
        client = Client(**client_kwargs)
        next(client.list_projects(limit=1), None)
    except Exception as exc:
        logger.warning(
            "LangSmith authentication failed (%s). Tracing disabled for this process.",
            exc,
        )
        _disable_tracing_env()
        _LANGSMITH_CONFIGURED = True
        return

    logger.info("LangSmith tracing enabled for project '%s'.", config.LANGCHAIN_PROJECT)
    _LANGSMITH_CONFIGURED = True


def build_run_config(
    *,
    state: AgentState | None,
    run_name: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a LangChain/LangGraph run config with consistent tracing metadata."""
    merged_tags = ["agent-first-ide", run_name]
    if tags:
        merged_tags.extend(tags)

    merged_metadata: dict[str, Any] = {
        "component": run_name,
        "autonomous_continuation": state.autonomous_continuation if state else None,
    }
    if state is not None:
        merged_metadata.update(
            {
                "campaign_id": state.campaign.campaign_id,
                "campaign_name": state.campaign.campaign_name,
                "thread_id": state.thread_id,
                "cycle_count": state.cycle_count,
                "dry_run_mode": state.dry_run_mode,
                "target_platforms": list(state.campaign.target_platforms),
            }
        )

    if metadata:
        merged_metadata.update(metadata)

    config_payload: dict[str, Any] = {
        "run_name": run_name,
        "tags": merged_tags,
        "metadata": merged_metadata,
    }

    if state is not None:
        config_payload["configurable"] = {
            "thread_id": state.thread_id,
        }
    config_payload.setdefault("configurable", {})["recursion_limit"] = 1000

    return config_payload


def build_llm_config(
    *,
    run_name: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create lightweight LLM invocation config for traced sub-operations."""
    merged_tags = ["llm", run_name]
    if tags:
        merged_tags.extend(tags)

    payload: dict[str, Any] = {
        "run_name": run_name,
        "tags": merged_tags,
    }
    if metadata:
        payload["metadata"] = metadata
    return payload
