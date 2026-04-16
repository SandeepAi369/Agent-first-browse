from __future__ import annotations

import functools
import json
import os
from typing import Any, TypeVar
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.browser_promoter.database import lock_target_community
from app.browser_promoter.state import AgentState, HighLevelCommand
from app.logger import get_logger

logger = get_logger(__name__)


class ContextAnalysisOutput(BaseModel):
    """Platform-aware context analysis for the Supervisor meeting."""

    platform_context: str = Field(min_length=1)
    pain_points: list[str] = Field(default_factory=list)
    audience_intent: str = ""
    stealth_notes: str = ""

    model_config = ConfigDict(extra="forbid")


class StrategyPlanOutput(BaseModel):
    """Strategic decision for next high-level action."""

    action_type: str = Field(min_length=1)
    target_description: str = Field(min_length=1)
    behavior_plan: str = Field(min_length=1)
    base_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reconnaissance_query: str = ""
    community_niche: str = ""
    candidate_community_url: str = ""

    model_config = ConfigDict(extra="forbid")


class CopyDraftOutput(BaseModel):
    """Optional human-like text draft for text-producing actions."""

    draft_text: str | None = None

    model_config = ConfigDict(extra="forbid")


class RiskStealthOutput(BaseModel):
    """Risk review with anti-bot adjustments and confidence scoring."""

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    stealth_adjustments: list[str] = Field(default_factory=list)
    risk_summary: str = ""

    model_config = ConfigDict(extra="forbid")


class SupervisorInternalState(BaseModel):
    """
    Ephemeral state for Supervisor subgraph collaboration.

    This state is private to the subgraph and never stored in main AgentState.
    """

    campaign_name: str = ""
    objective: str = ""
    target_platforms: list[str] = Field(default_factory=list)
    current_url: str = ""
    current_scene_summary: str = ""
    worker_last_confused: bool = False
    worker_last_confusion_reason: str = ""

    context_analysis: ContextAnalysisOutput | None = None
    strategy_plan: StrategyPlanOutput | None = None
    copy_draft: CopyDraftOutput | None = None
    risk_assessment: RiskStealthOutput | None = None
    final_command: HighLevelCommand | None = None

    model_config = ConfigDict(extra="forbid")


TModel = TypeVar("TModel", bound=BaseModel)

_SUPERVISOR_LLM: ChatOpenAI | None = None


def _get_supervisor_llm() -> ChatOpenAI | None:
    """Create one shared big-model client for Supervisor sub-agents."""
    global _SUPERVISOR_LLM
    if _SUPERVISOR_LLM is not None:
        return _SUPERVISOR_LLM

    api_key = os.getenv("SUPERVISOR_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    _SUPERVISOR_LLM = ChatOpenAI(
        model=os.getenv("SUPERVISOR_BIG_MODEL", "gpt-4o"),
        api_key=api_key,
        temperature=0.2,
        timeout=40,
    )
    return _SUPERVISOR_LLM


async def _invoke_structured(
    *,
    schema: type[TModel],
    system_prompt: str,
    payload: dict[str, Any],
    fallback: TModel,
) -> TModel:
    """Run a structured LLM chain with resilient fallback behavior."""
    llm = _get_supervisor_llm()
    if llm is None:
        return fallback

    chain = llm.with_structured_output(schema)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=True)),
    ]

    try:
        raw = await chain.ainvoke(messages)
        if isinstance(raw, BaseModel):
            return schema.model_validate(raw.model_dump(mode="json"))
        return schema.model_validate(raw)
    except Exception:
        return fallback


def _infer_platform_hint(state: SupervisorInternalState) -> str:
    url = state.current_url.lower()
    if "x.com" in url or "twitter.com" in url:
        return "x"
    if "reddit.com" in url:
        return "reddit"
    if "github.com" in url:
        return "github"
    if state.target_platforms:
        return state.target_platforms[0]
    return "web"


async def context_analyzer_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Sub-agent A: Analyze platform context and likely audience pain points."""
    platform = _infer_platform_hint(state)
    fallback = ContextAnalysisOutput(
        platform_context=f"Platform={platform}; objective={state.objective[:160]}",
        pain_points=["time", "trust", "signal-to-noise"],
        audience_intent="Find practical, credible, and fast value.",
        stealth_notes="Blend with native posting rhythm and avoid repetitive phrasing.",
    )

    result = await _invoke_structured(
        schema=ContextAnalysisOutput,
        system_prompt=(
            "You are Context Analyzer. Extract platform context and pain points for organic promotion. "
            "Be concise, tactical, and human. Avoid corporate buzzwords."
        ),
        payload={
            "campaign_name": state.campaign_name,
            "objective": state.objective,
            "platform_hint": platform,
            "current_url": state.current_url,
            "scene_summary": state.current_scene_summary,
            "worker_confusion": {
                "flag": state.worker_last_confused,
                "reason": state.worker_last_confusion_reason,
            },
        },
        fallback=fallback,
    )
    return {"context_analysis": result}


async def strategy_planner_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Sub-agent B: Decide next action type and behavior mimicry plan."""
    context = state.context_analysis or ContextAnalysisOutput(
        platform_context="Unknown context",
        pain_points=[],
        audience_intent="",
        stealth_notes="",
    )
    fallback = StrategyPlanOutput(
        action_type="reconnaissance" if _should_default_to_recon(state) else "engage",
        target_description="Most relevant organic discussion thread on current page",
        behavior_plan=(
            "Search for high-signal communities, observe discussion quality, and avoid posting until a "
            "strong fit is confirmed."
            if _should_default_to_recon(state)
            else "Observe nearby human phrasing, mirror pacing, and take one minimal action before any text submission."
        ),
        base_confidence=0.55,
        reconnaissance_query="best developer communities reddit github open source discussions",
        community_niche="software-engineering",
        candidate_community_url=state.current_url,
    )

    result = await _invoke_structured(
        schema=StrategyPlanOutput,
        system_prompt=(
            "You are Strategy Planner. Choose one next action_type and behavior plan. "
            "Supported action types include reconnaissance and engage. "
            "When discovery is needed, choose reconnaissance and provide a concrete recon query plus "
            "candidate_community_url if visible. Prioritize stealth and organic behavior."
        ),
        payload={
            "objective": state.objective,
            "platform_context": context.platform_context,
            "pain_points": context.pain_points,
            "audience_intent": context.audience_intent,
            "stealth_notes": context.stealth_notes,
            "current_scene_summary": state.current_scene_summary,
            "current_url": state.current_url,
            "target_platforms": state.target_platforms,
        },
        fallback=fallback,
    )
    return {"strategy_plan": result}


async def copywriter_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Sub-agent C: Draft short human-like copy when text is needed."""
    strategy = state.strategy_plan or StrategyPlanOutput(
        action_type="engage",
        target_description="Relevant target element",
        behavior_plan="Act naturally.",
        base_confidence=0.5,
    )

    fallback_text: str | None = None
    if strategy.action_type.lower() in {"type", "comment", "reply", "post", "message", "engage"}:
        fallback_text = "Interesting point. I tested a lightweight approach and it helped quickly."

    fallback = CopyDraftOutput(draft_text=fallback_text)

    result = await _invoke_structured(
        schema=CopyDraftOutput,
        system_prompt=(
            "You are Copywriter. Draft one short, natural line in a human tone. "
            "No hype, no slogans, no corporate buzzwords."
        ),
        payload={
            "objective": state.objective,
            "action_type": strategy.action_type,
            "target_description": strategy.target_description,
            "behavior_plan": strategy.behavior_plan,
            "scene_summary": state.current_scene_summary,
        },
        fallback=fallback,
    )
    return {"copy_draft": result}


async def risk_stealth_assessor_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Sub-agent D: Score anti-bot risk and propose stealth adjustments."""
    strategy = state.strategy_plan or StrategyPlanOutput(
        action_type="engage",
        target_description="Relevant target element",
        behavior_plan="Act naturally.",
        base_confidence=0.5,
    )

    fallback = RiskStealthOutput(
        confidence=max(0.35, min(0.85, strategy.base_confidence)),
        stealth_adjustments=[
            "Randomize think-time between actions (1.2s-4.5s).",
            "Avoid repetitive action loops on identical UI elements.",
            "Prefer selector-based interaction over pixel clicks when available.",
        ],
        risk_summary="Moderate risk. Keep cadence natural and sparse.",
    )

    result = await _invoke_structured(
        schema=RiskStealthOutput,
        system_prompt=(
            "You are Risk & Stealth Assessor. Evaluate anti-bot risk and return concrete adjustments. "
            "Favor conservative, human-like behavior."
        ),
        payload={
            "action_type": strategy.action_type,
            "target_description": strategy.target_description,
            "behavior_plan": strategy.behavior_plan,
            "worker_confused": state.worker_last_confused,
            "worker_confusion_reason": state.worker_last_confusion_reason,
            "scene_summary": state.current_scene_summary,
        },
        fallback=fallback,
    )
    return {"risk_assessment": result}


async def final_merge_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Merge all sub-agent outputs into the final HighLevelCommand."""
    strategy = state.strategy_plan or StrategyPlanOutput(
        action_type="engage",
        target_description="Relevant target element",
        behavior_plan="Act naturally with minimal interaction.",
        base_confidence=0.45,
    )
    copy = state.copy_draft or CopyDraftOutput(draft_text=None)
    risk = state.risk_assessment or RiskStealthOutput(
        confidence=0.45,
        stealth_adjustments=["Slow down and reduce action frequency."],
        risk_summary="Fallback risk profile.",
    )

    merged_confidence = max(0.0, min(1.0, (strategy.base_confidence + risk.confidence) / 2.0))
    behavior_plan = _compose_behavior_plan(strategy=strategy, state=state)

    command = HighLevelCommand(
        action_type=strategy.action_type,
        target_description=strategy.target_description,
        draft_text=copy.draft_text,
        behavior_plan=behavior_plan,
        confidence=merged_confidence,
        stealth_adjustments=risk.stealth_adjustments,
    )

    _lock_target_if_reconnaissance(state=state, strategy=strategy)
    return {"final_command": command}


def _should_default_to_recon(state: SupervisorInternalState) -> bool:
    if not state.current_url.strip():
        return True
    if state.worker_last_confused:
        return True
    return False


def _is_recon_action(action_type: str) -> bool:
    lowered = action_type.strip().lower()
    return lowered in {"reconnaissance", "recon", "discovery", "community_recon"}


def _compose_behavior_plan(strategy: StrategyPlanOutput, state: SupervisorInternalState) -> str:
    """Ensure reconnaissance actions carry explicit search-and-read instructions."""
    if not _is_recon_action(strategy.action_type):
        return strategy.behavior_plan

    recon_query = strategy.reconnaissance_query.strip() or (
        "site:reddit.com programming communities OR site:github.com topics developers"
    )
    platform_hint = _infer_platform_hint(state)
    base = strategy.behavior_plan.strip()

    recon_suffix = (
        f" Run reconnaissance using query: {recon_query}. "
        f"Search on Google and {platform_hint} where applicable, read on-screen signals, "
        "and identify high-trust communities before engagement."
    )
    if recon_suffix.strip() in base:
        return base
    return (base + " " + recon_suffix).strip()


def _lock_target_if_reconnaissance(state: SupervisorInternalState, strategy: StrategyPlanOutput) -> None:
    """Persist discovered communities during reconnaissance for future posting workflows."""
    if not _is_recon_action(strategy.action_type):
        return

    candidate_url = strategy.candidate_community_url.strip() or state.current_url.strip()
    if not candidate_url or not _looks_like_valid_http_url(candidate_url):
        return

    platform = _platform_from_url(candidate_url) or _infer_platform_hint(state)
    niche = strategy.community_niche.strip() or _infer_niche(state)

    try:
        lock_target_community(
            platform=platform,
            url=candidate_url,
            niche=niche,
        )
    except Exception:
        # Persistence failures should not break Supervisor planning loop.
        return


def _looks_like_valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _platform_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "reddit.com" in host:
        return "reddit"
    if "github.com" in host:
        return "github"
    if "x.com" in host or "twitter.com" in host:
        return "x"
    return "web"


def _infer_niche(state: SupervisorInternalState) -> str:
    summary = state.current_scene_summary.lower()
    if "ai" in summary:
        return "ai"
    if "security" in summary:
        return "security"
    if "devops" in summary:
        return "devops"
    if "python" in summary:
        return "python"
    return "software-engineering"


def build_supervisor_subgraph():
    """Compile the hierarchical Supervisor meeting as a LangGraph subgraph."""
    graph = StateGraph(SupervisorInternalState)

    graph.add_node("context_analyzer", context_analyzer_node)
    graph.add_node("strategy_planner", strategy_planner_node)
    graph.add_node("copywriter", copywriter_node)
    graph.add_node("risk_stealth_assessor", risk_stealth_assessor_node)
    graph.add_node("final_merge", final_merge_node)

    graph.add_edge(START, "context_analyzer")
    graph.add_edge("context_analyzer", "strategy_planner")

    graph.add_edge("strategy_planner", "copywriter")
    graph.add_edge("strategy_planner", "risk_stealth_assessor")

    graph.add_edge("copywriter", "final_merge")
    graph.add_edge("risk_stealth_assessor", "final_merge")

    graph.add_edge("final_merge", END)

    return graph.compile()


@functools.lru_cache(maxsize=1)
def _get_supervisor_subgraph():
    """Lazily compile and cache the Supervisor subgraph singleton."""
    logger.info("Compiling Supervisor subgraph (one-time)")
    return build_supervisor_subgraph()


def _fallback_high_level_command(main_state: AgentState) -> HighLevelCommand:
    return HighLevelCommand(
        action_type="engage",
        target_description=main_state.current_url or "Best-matching active thread on page",
        draft_text=None,
        behavior_plan=(
            "Take one low-risk interaction step, then re-evaluate with fresh screenshot before posting."
        ),
        confidence=0.4,
        stealth_adjustments=[
            "Keep interaction cadence irregular and sparse.",
            "Avoid repetitive text patterns across attempts.",
        ],
    )


async def run_supervisor_subgraph(main_state: AgentState) -> HighLevelCommand:
    """Run the compiled Supervisor subgraph and return final structured command."""
    internal_state = SupervisorInternalState(
        campaign_name=main_state.campaign.campaign_name,
        objective=main_state.campaign.objective,
        target_platforms=list(main_state.campaign.target_platforms),
        current_url=main_state.current_url,
        current_scene_summary=main_state.current_scene_summary,
        worker_last_confused=main_state.worker_last_confused,
        worker_last_confusion_reason=main_state.worker_last_confusion_reason,
    )

    graph = _get_supervisor_subgraph()
    result = await graph.ainvoke(internal_state.model_dump(mode="json"))
    parsed = SupervisorInternalState.model_validate(result)

    if parsed.final_command is None:
        return _fallback_high_level_command(main_state)
    return parsed.final_command
