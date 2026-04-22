"""v3.0 Marketing Engine import and integration test."""
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "python-orchestrator"))

from app import config
from app.browser_promoter.database import initialize_persistence_database
from app.browser_promoter.github_intelligence import GitHubIntelligence, RepoProfile
from app.browser_promoter.marketing_engine import (
    MarketingEngine,
    PromotionPlan,
    build_portfolio_context,
)
from app.browser_promoter.supervisor_subgraph import (
    SupervisorInternalState,
    build_supervisor_subgraph,
    github_intelligence_node,
)
from app.browser_promoter.state import AgentState, CampaignContext, BrowserConfig, BrowserMode
from app.browser_promoter import (
    GitHubIntelligence as GI_export,
    MarketingEngine as ME_export,
    PromotionPlan as PP_export,
    RepoProfile as RP_export,
)


async def main():
    print("=" * 60)
    print("  MARKETING ENGINE v3.0 — INTEGRATION TEST")
    print("=" * 60)
    print()

    # 1. Database initialization (creates new marketing tables)
    print("[1/7] Initializing database with marketing tables...")
    db_path = initialize_persistence_database()
    print(f"  OK — {db_path}")
    print()

    # 2. Config validation
    print("[2/7] Config validation...")
    print(f"  GITHUB_USERNAME:     {config.GITHUB_USERNAME}")
    print(f"  PROMOTION_STYLE:     {config.PROMOTION_STYLE}")
    print(f"  MAX_PER_PLATFORM:    {config.MAX_PROMOTIONS_PER_PLATFORM}")
    print(f"  COOLDOWN_DAYS:       {config.PROMOTION_COOLDOWN_DAYS}")
    print(f"  PLATFORMS:           {config.PROMOTION_PLATFORMS}")
    print()

    # 3. GitHub Intelligence — fetch live repos
    print("[3/7] GitHub Intelligence — fetching repos...")
    intel = GitHubIntelligence(username=config.GITHUB_USERNAME)
    profiles = await intel.get_repo_profiles()
    print(f"  Found {len(profiles)} promotable repos:")
    for p in profiles:
        features = f" ({', '.join(p.key_features[:2])})" if p.key_features else ""
        print(f"    ⭐{p.stars} {p.name} [{p.language}]{features}")
    print()

    # 4. Marketing Engine — generate promotion plans
    print("[4/7] Marketing Engine — generating promotion plans...")
    engine = MarketingEngine()
    plans = await engine.generate_promotion_plans(profiles, max_plans=8)
    print(f"  Generated {len(plans)} plans:")
    for plan in plans:
        blocked = " ⛔ BLOCKED" if plan.is_blocked() else ""
        print(f"    [{plan.safety_score:.1f}] {plan.repo.name} → {plan.platform}/{plan.channel} ({plan.tactic}){blocked}")
        if not plan.is_blocked():
            print(f"         Draft: {plan.draft_message[:80]}...")
    print()

    # 5. Portfolio context for LLM
    print("[5/7] Portfolio context (for LLM injection)...")
    ctx = build_portfolio_context(profiles)
    lines = ctx.split("\n")
    print(f"  {len(lines)} lines, {len(ctx)} chars")
    for line in lines[:6]:
        print(f"    {line}")
    print("    ...")
    print()

    # 6. Supervisor subgraph compilation
    print("[6/7] Compiling v3.0 Supervisor subgraph...")
    graph = build_supervisor_subgraph()
    print(f"  OK — Nodes: {list(graph.get_graph().nodes.keys())}")
    print()

    # 7. CampaignContext with new fields
    print("[7/7] CampaignContext with marketing fields...")
    campaign = CampaignContext(
        campaign_id="test-v3",
        campaign_name="Marketing Engine Test",
        objective="Promote SearchWala on Reddit organically",
        target_platforms=["reddit"],
        session_id="test-session",
        github_username="SandeepAi369",
        promotion_style="organic",
    )
    print(f"  campaign.github_username: {campaign.github_username}")
    print(f"  campaign.promotion_style: {campaign.promotion_style}")
    print()

    # Package exports
    assert GI_export is GitHubIntelligence
    assert ME_export is MarketingEngine
    assert PP_export is PromotionPlan
    assert RP_export is RepoProfile
    print("  Package exports: ✓")

    print()
    print("=" * 60)
    print("  ALL v3.0 TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
