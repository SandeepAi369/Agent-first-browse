from __future__ import annotations

import functools
import json
from typing import Any, TypeVar
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.call_pacing import AsyncGapLimiter
from app.browser_promoter.database import lock_target_community
from app.browser_promoter.github_intelligence import GitHubIntelligence, RepoProfile
from app.browser_promoter.marketing_engine import (
    MarketingEngine,
    PromotionPlan,
    build_portfolio_context,
)
from app.browser_promoter.state import AgentState, HighLevelCommand
from app import config
from app.observability import build_llm_config
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


class StrategyChallengeOutput(BaseModel):
    """Critical discussion output that challenges and improves strategy choices."""

    revised_action_type: str = ""
    revised_target_description: str = ""
    revised_behavior_plan: str = ""
    confidence_delta: float = Field(default=0.0, ge=-0.35, le=0.35)
    discussion_points: list[str] = Field(default_factory=list)

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
    ephemeral: dict[str, Any] = Field(default_factory=dict)

    # v3.0 Marketing Intelligence
    github_username: str = ""
    promotion_style: str = "organic"
    portfolio_context: str = ""  # Injected by github_intelligence_node
    promotion_plans_json: str = "[]"  # Serialized PromotionPlan summaries

    context_analysis: ContextAnalysisOutput | None = None
    strategy_plan: StrategyPlanOutput | None = None
    strategy_challenge: StrategyChallengeOutput | None = None
    copy_draft: CopyDraftOutput | None = None
    risk_assessment: RiskStealthOutput | None = None
    final_command: HighLevelCommand | None = None

    model_config = ConfigDict(extra="forbid")


TModel = TypeVar("TModel", bound=BaseModel)

_SUPERVISOR_LLM_CLIENTS: list[Any] | None = None
_SUPERVISOR_LLM_CURSOR: int = 0
_SUPERVISOR_LLM_GAP = AsyncGapLimiter(config.SUPERVISOR_LLM_MIN_GAP_SECONDS)


def _is_google_model_name(model_name: str) -> bool:
    lowered = model_name.strip().lower()
    return "gemini" in lowered or "gemma" in lowered


def _get_supervisor_llm_clients() -> list[Any]:
    """Create shared big-model clients with multi-provider failover.

    Architecture:
      - Client 0: NVIDIA GLM 5.1 (primary) via integrate.api.nvidia.com
      - Client 1+: Groq GPT-OSS-120B (backup) via api.groq.com
    Each client carries its own base_url and model name so failover
    seamlessly switches providers.
    """
    global _SUPERVISOR_LLM_CLIENTS
    if _SUPERVISOR_LLM_CLIENTS is not None:
        return _SUPERVISOR_LLM_CLIENTS

    clients: list[Any] = []
    primary_model = config.SUPERVISOR_LLM_MODEL.strip()
    fallback_model = config.SUPERVISOR_FALLBACK_MODEL.strip() or "openai/gpt-oss-120b"
    primary_base_url = config.SUPERVISOR_PRIMARY_BASE_URL.strip() or None
    fallback_base_url = config.SUPERVISOR_FALLBACK_BASE_URL.strip() or config.OPENAI_BASE_URL.strip() or None

    # ── Primary: NVIDIA NIM (or first key) ──
    primary_key = config.SUPERVISOR_MODEL_API_KEY.strip()
    if primary_key:
        clients.append(
            ChatOpenAI(
                model=primary_model,
                api_key=primary_key,
                base_url=primary_base_url,
                temperature=0.2,
                timeout=40,
            )
        )
        logger.info("Supervisor primary: %s via %s", primary_model, primary_base_url or "default")

    # ── Fallback: Groq keys (use fallback model + fallback base_url) ──
    fallback_keys = config.SUPERVISOR_MODEL_API_KEY_FALLBACKS
    for fk in fallback_keys:
        fk = fk.strip()
        if not fk or fk == primary_key:
            continue
        clients.append(
            ChatOpenAI(
                model=fallback_model,
                api_key=fk,
                base_url=fallback_base_url,
                temperature=0.2,
                timeout=40,
            )
        )
    if fallback_keys:
        logger.info("Supervisor fallback: %s via %s (%d keys)", fallback_model, fallback_base_url or "default", len(fallback_keys))

    if not clients:
        logger.warning("Supervisor model is not configured: no NVIDIA or Groq API keys found.")

    _SUPERVISOR_LLM_CLIENTS = clients
    return _SUPERVISOR_LLM_CLIENTS


def _ordered_supervisor_llms(clients: list[Any]) -> list[tuple[int, Any]]:
    if not clients:
        return []

    total = len(clients)
    start = _SUPERVISOR_LLM_CURSOR % total
    ordered: list[tuple[int, Any]] = []
    for offset in range(total):
        idx = (start + offset) % total
        ordered.append((idx, clients[idx]))
    return ordered


async def _invoke_structured(
    *,
    schema: type[TModel],
    system_prompt: str,
    payload: dict[str, Any],
    fallback: TModel,
    run_name: str,
    trace_metadata: dict[str, Any] | None = None,
) -> TModel:
    """Run a structured LLM chain with resilient fallback behavior."""
    clients = _get_supervisor_llm_clients()
    if not clients:
        return fallback

    global _SUPERVISOR_LLM_CURSOR

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=True)),
    ]
    invoke_config = build_llm_config(
        run_name=run_name,
        tags=["supervisor", schema.__name__],
        metadata=trace_metadata,
    )

    wait_seconds = await _SUPERVISOR_LLM_GAP.wait_turn()
    if wait_seconds > 0.0:
        logger.info("Supervisor LLM pacing wait applied: %.2fs (%s)", wait_seconds, run_name)

    total_clients = len(clients)
    last_exc: Exception | None = None
    for idx, llm in _ordered_supervisor_llms(clients):
        try:
            chain = llm.with_structured_output(schema)
            raw = await chain.ainvoke(messages, config=invoke_config)
            _SUPERVISOR_LLM_CURSOR = idx
            if isinstance(raw, BaseModel):
                return schema.model_validate(raw.model_dump(mode="json"))
            return schema.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "Structured supervisor invoke failed (%s) on key %d/%d: %s",
                run_name,
                idx + 1,
                total_clients,
                exc,
            )
            try:
                fallback_messages = [
                    SystemMessage(content=system_prompt + f"\n\nCRITICAL INSTRUCTION: You MUST format your response as a pure JSON object string conforming exactly to this schema:\n{json.dumps(schema.model_json_schema())}"),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=True)),
                ]
                raw = await llm.ainvoke(fallback_messages, config=invoke_config)
                raw_text = _extract_response_text(raw)
                parsed = _extract_json_payload(raw_text)
                _SUPERVISOR_LLM_CURSOR = idx
                return schema.model_validate(parsed)
            except Exception as raw_exc:
                last_exc = raw_exc
                logger.warning(
                    "Supervisor raw-JSON fallback failed (%s) on key %d/%d: %s",
                    run_name,
                    idx + 1,
                    total_clients,
                    raw_exc,
                )

    logger.error("SUPERVISOR LLM FAILED ALL KEYS (%s): %s", run_name, last_exc)
    return fallback


def _extract_response_text(raw_response: Any) -> str:
    content = getattr(raw_response, "content", raw_response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    return str(content)


def _extract_json_payload(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if "```json" in candidate:
        candidate = candidate.split("```json", maxsplit=1)[1].split("```", maxsplit=1)[0]
    elif "```" in candidate:
        candidate = candidate.split("```", maxsplit=1)[1].split("```", maxsplit=1)[0]

    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(candidate[start : end + 1])


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

    payload_data = {
        "campaign_name": state.campaign_name,
        "objective": state.objective,
        "platform_hint": platform,
        "current_url": state.current_url,
        "scene_summary": state.current_scene_summary,
        "worker_confusion": {
            "flag": state.worker_last_confused,
            "reason": state.worker_last_confusion_reason,
        },
    }
    
    if state.ephemeral.get("kick_prompt"):
        payload_data["SYSTEM_OVERRIDE"] = state.ephemeral["kick_prompt"]

    result = await _invoke_structured(
        schema=ContextAnalysisOutput,
        system_prompt=(
            "You are Context Analyzer. Extract platform context and pain points for organic promotion. "
            "Be concise, tactical, and human. Avoid corporate buzzwords."
        ),
        payload=payload_data,
        fallback=fallback,
        run_name="supervisor.context_analyzer",
        trace_metadata={
            "platform_hint": platform,
            "current_url": state.current_url,
        },
    )
    return {"context_analysis": result}


async def github_intelligence_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Sub-agent A2: Fetch GitHub repo intelligence and build portfolio context.

    Queries the GitHub API (or cache) for the user's public repositories,
    analyzes READMEs, and injects structured product knowledge into the
    supervisor pipeline so downstream nodes have real data to work with.
    """
    username = state.github_username or config.GITHUB_USERNAME
    if not username:
        logger.info("No GitHub username configured — skipping intelligence fetch")
        return {"portfolio_context": "", "promotion_plans_json": "[]"}

    try:
        intelligence = GitHubIntelligence(username=username)
        profiles = await intelligence.get_repo_profiles()

        if not profiles:
            logger.warning("GitHub Intelligence returned 0 profiles for %s", username)
            return {"portfolio_context": "No promotable repos found.", "promotion_plans_json": "[]"}

        # Build portfolio context for LLM injection
        portfolio_ctx = build_portfolio_context(profiles)

        # Generate promotion plans
        engine = MarketingEngine()
        plans = await engine.generate_promotion_plans(profiles, max_plans=5)
        plans_summary = [
            {
                "repo": p.repo.name,
                "platform": p.platform,
                "channel": p.channel,
                "tactic": p.tactic,
                "angle": p.content_angle,
                "search_query": p.search_query,
                "safety": p.safety_score,
                "blocked": p.blocked_reason,
                "draft": p.draft_message[:200],
            }
            for p in plans if not p.is_blocked()
        ]

        logger.info(
            "GitHub Intelligence: %d repos, %d actionable plans for %s",
            len(profiles), len(plans_summary), username,
        )

        return {
            "portfolio_context": portfolio_ctx,
            "promotion_plans_json": json.dumps(plans_summary, ensure_ascii=True),
        }

    except Exception as exc:
        logger.error("GitHub Intelligence failed: %s", exc)
        return {"portfolio_context": "", "promotion_plans_json": "[]"}


async def strategy_planner_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Sub-agent B: Decide next action type and behavior mimicry plan.

    Now enhanced with GitHub portfolio context and promotion plans from
    the github_intelligence_node.
    """
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

    # Build enhanced payload with portfolio + promotion intelligence
    payload: dict[str, Any] = {
        "objective": state.objective,
        "platform_context": context.platform_context,
        "pain_points": context.pain_points,
        "audience_intent": context.audience_intent,
        "stealth_notes": context.stealth_notes,
        "current_scene_summary": state.current_scene_summary,
        "current_url": state.current_url,
        "target_platforms": state.target_platforms,
    }

    if state.ephemeral.get("kick_prompt"):
        payload["SYSTEM_OVERRIDE"] = state.ephemeral["kick_prompt"]

    # Inject portfolio context if available
    if state.portfolio_context:
        payload["github_portfolio"] = state.portfolio_context[:2000]
    if state.promotion_plans_json and state.promotion_plans_json != "[]":
        payload["promotion_plans"] = state.promotion_plans_json[:1500]

    result = await _invoke_structured(
        schema=StrategyPlanOutput,
        system_prompt=(
            "You are Strategy Planner. Choose one next action_type and behavior plan. "
            "Supported action types include reconnaissance and engage. "
            "When discovery is needed, choose reconnaissance and provide a concrete recon query plus "
            "candidate_community_url if visible. Prioritize stealth and organic behavior. "
            "If github_portfolio and promotion_plans are provided, use them to inform your "
            "strategy — pick the best repo and channel for the current context. "
            "CRITICAL: You are a multi-platform autonomous agent. If Reddit yields no results or you get stuck, "
            "you MUST PIVOT to searching other platforms like Facebook Groups, HackerNews, GitHub Discussions, "
            "or Dev.to using the web search tools. "
            "You MUST return a valid JSON object matching the requested schema exactly."
        ),
        payload=payload,
        fallback=fallback,
        run_name="supervisor.strategy_planner",
        trace_metadata={
            "current_url": state.current_url,
            "worker_last_confused": state.worker_last_confused,
            "has_portfolio": bool(state.portfolio_context),
        },
    )
    return {"strategy_plan": result}


async def strategy_challenger_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Sub-agent B2: Challenge strategy and propose safer, higher-signal alternatives."""
    strategy = state.strategy_plan or StrategyPlanOutput(
        action_type="reconnaissance",
        target_description="Most relevant active thread",
        behavior_plan="Collect context before engagement.",
        base_confidence=0.5,
    )

    fallback = StrategyChallengeOutput(
        revised_action_type=strategy.action_type,
        revised_target_description=strategy.target_description,
        revised_behavior_plan=strategy.behavior_plan,
        confidence_delta=0.0,
        discussion_points=[
            "Prefer one reversible action before any irreversible submission.",
            "If signals conflict, gather one more observation cycle.",
        ],
    )

    result = await _invoke_structured(
        schema=StrategyChallengeOutput,
        system_prompt=(
            "You are Strategy Challenger. Critique the proposed strategy and improve it. "
            "You may keep the same plan if it is already strong. "
            "Return concrete discussion_points and confidence_delta adjustments."
        ),
        payload={
            "objective": state.objective,
            "current_url": state.current_url,
            "scene_summary": state.current_scene_summary,
            "strategy": strategy.model_dump(mode="json"),
            "worker_confusion": {
                "flag": state.worker_last_confused,
                "reason": state.worker_last_confusion_reason,
            },
        },
        fallback=fallback,
        run_name="supervisor.strategy_challenger",
        trace_metadata={
            "current_url": state.current_url,
            "action_type": strategy.action_type,
        },
    )
    return {"strategy_challenge": result}


async def copywriter_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Sub-agent C: Draft short human-like copy when text is needed.

    Enhanced in v3.0 with real product knowledge from GitHub Intelligence.
    Uses portfolio context and promotion hooks to generate authentic,
    experience-based messages instead of generic marketing copy.
    """
    strategy = _resolve_effective_strategy(state)

    fallback_text: str | None = None
    if strategy.action_type.lower() in {"type", "comment", "reply", "post", "message", "engage"}:
        fallback_text = "Interesting point. I tested a lightweight approach and it helped quickly."

    fallback = CopyDraftOutput(draft_text=fallback_text)

    # Build enhanced payload with real product data
    copy_payload: dict[str, Any] = {
        "objective": state.objective,
        "action_type": strategy.action_type,
        "target_description": strategy.target_description,
        "behavior_plan": strategy.behavior_plan,
        "scene_summary": state.current_scene_summary,
    }

    # Inject portfolio context for authentic copy generation
    if state.portfolio_context:
        copy_payload["product_knowledge"] = state.portfolio_context[:1500]
    if state.promotion_plans_json and state.promotion_plans_json != "[]":
        copy_payload["promotion_context"] = state.promotion_plans_json[:800]

    # Determine tone based on promotion style
    style = state.promotion_style or "organic"
    tone_guide = {
        "organic": "Write like a fellow developer sharing a personal experience. Never sell.",
        "direct": "Write a concise project showcase. Be genuine and invite feedback.",
        "educational": "Write as someone sharing a technical insight. The project is a supporting reference, not the focus.",
    }.get(style, "Write naturally like a developer.")

    result = await _invoke_structured(
        schema=CopyDraftOutput,
        system_prompt=(
            f"You are Copywriter for a developer marketing campaign. {tone_guide} "
            "Draft one short, natural paragraph in a genuine human tone. "
            "No hype, no slogans, no corporate buzzwords. "
            "If product_knowledge is provided, reference specific features and your "
            "actual experience building or using the tool. "
            "CRITICAL: The text must sound like something a real developer would write "
            "in a forum post — casual, helpful, with a personal touch."
        ),
        payload=copy_payload,
        fallback=fallback,
        run_name="supervisor.copywriter",
        trace_metadata={
            "action_type": strategy.action_type,
            "current_url": state.current_url,
            "promotion_style": style,
            "has_product_knowledge": bool(state.portfolio_context),
        },
    )
    return {"copy_draft": result}


async def risk_stealth_assessor_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Sub-agent D: Score anti-bot risk and propose stealth adjustments."""
    strategy = _resolve_effective_strategy(state)

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
        run_name="supervisor.risk_stealth_assessor",
        trace_metadata={
            "action_type": strategy.action_type,
            "worker_last_confused": state.worker_last_confused,
        },
    )
    return {"risk_assessment": result}


async def final_merge_node(state: SupervisorInternalState) -> dict[str, Any]:
    """Merge all sub-agent outputs into the final HighLevelCommand."""
    strategy = _resolve_effective_strategy(state)
    challenge = state.strategy_challenge or StrategyChallengeOutput()
    copy = state.copy_draft or CopyDraftOutput(draft_text=None)
    risk = state.risk_assessment or RiskStealthOutput(
        confidence=0.45,
        stealth_adjustments=["Slow down and reduce action frequency."],
        risk_summary="Fallback risk profile.",
    )

    merged_confidence = max(
        0.0,
        min(1.0, ((strategy.base_confidence + risk.confidence) / 2.0) + challenge.confidence_delta),
    )
    behavior_plan = _compose_behavior_plan(strategy=strategy, state=state)
    if challenge.discussion_points:
        behavior_plan = (
            behavior_plan
            + " Discussion notes: "
            + " | ".join(point for point in challenge.discussion_points[:3] if point.strip())
        )

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


def _resolve_effective_strategy(state: SupervisorInternalState) -> StrategyPlanOutput:
    """Merge planner proposal with challenger revisions for stronger decision quality."""
    base = state.strategy_plan or StrategyPlanOutput(
        action_type="engage",
        target_description="Relevant target element",
        behavior_plan="Act naturally with minimal interaction.",
        base_confidence=0.45,
    )
    challenge = state.strategy_challenge
    if challenge is None:
        return base

    return base.model_copy(
        update={
            "action_type": challenge.revised_action_type.strip() or base.action_type,
            "target_description": challenge.revised_target_description.strip() or base.target_description,
            "behavior_plan": challenge.revised_behavior_plan.strip() or base.behavior_plan,
        }
    )


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
    """Compile the hierarchical Supervisor meeting as a LangGraph subgraph.

    v3.0 Pipeline:
      START → context_analyzer → github_intelligence → strategy_planner
            → strategy_challenger → [copywriter, risk_assessor] → final_merge → END
    """
    graph = StateGraph(SupervisorInternalState)

    graph.add_node("context_analyzer", context_analyzer_node)
    graph.add_node("github_intelligence", github_intelligence_node)
    graph.add_node("strategy_planner", strategy_planner_node)
    graph.add_node("strategy_challenger", strategy_challenger_node)
    graph.add_node("copywriter", copywriter_node)
    graph.add_node("risk_stealth_assessor", risk_stealth_assessor_node)
    graph.add_node("final_merge", final_merge_node)

    # v3.0: GitHub Intelligence runs right after context analysis
    graph.add_edge(START, "context_analyzer")
    graph.add_edge("context_analyzer", "github_intelligence")
    graph.add_edge("github_intelligence", "strategy_planner")
    graph.add_edge("strategy_planner", "strategy_challenger")

    graph.add_edge("strategy_challenger", "copywriter")
    graph.add_edge("strategy_challenger", "risk_stealth_assessor")

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
        ephemeral=dict(main_state.ephemeral),
        # v3.0 Marketing Intelligence
        github_username=main_state.campaign.github_username,
        promotion_style=main_state.campaign.promotion_style,
    )

    graph = _get_supervisor_subgraph()
    result = await graph.ainvoke(
        internal_state.model_dump(mode="json"),
        config=build_llm_config(
            run_name="supervisor.subgraph",
            tags=[
                "supervisor",
                main_state.campaign.campaign_id,
            ],
            metadata={
                "thread_id": main_state.thread_id,
                "cycle_count": main_state.cycle_count,
                "current_url": main_state.current_url,
            },
        ),
    )
    parsed = SupervisorInternalState.model_validate(result)

    if parsed.final_command is None:
        return _fallback_high_level_command(main_state)
    return parsed.final_command
