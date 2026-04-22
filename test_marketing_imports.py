"""v3.0 import-only test (Windows-safe, no SQLite over \\\\wsl$)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "python-orchestrator"))

# Import everything — this validates Python syntax + module structure
from app.browser_promoter.github_intelligence import (
    GitHubIntelligence, RepoProfile,
    _extract_features_from_readme, _infer_target_audience, _generate_promotion_hooks,
)
from app.browser_promoter.marketing_engine import (
    MarketingEngine, PromotionPlan, build_portfolio_context,
)
from app.browser_promoter.supervisor_subgraph import (
    SupervisorInternalState, github_intelligence_node,
)
from app.browser_promoter.state import CampaignContext
from app import config

print("=== MARKETING ENGINE v3.0 — IMPORT TEST ===")
print()

# 1. Config
print(f"  GITHUB_USERNAME:     {config.GITHUB_USERNAME}")
print(f"  PROMOTION_STYLE:     {config.PROMOTION_STYLE}")
print(f"  PLATFORMS:           {config.PROMOTION_PLATFORMS}")

# 2. RepoProfile construction
profile = RepoProfile(
    name="SearchWala",
    full_name="SandeepAi369/SearchWala",
    url="https://github.com/SandeepAi369/SearchWala",
    description="Lightning-fast meta-search engine aggregator built in Rust",
    language="Rust",
    stars=4,
    forks=1,
    topics=["search-engine", "meta-search", "rust"],
    homepage="https://sandeepai369.github.io/SearchWala/",
    readme_summary="# SearchWala\n\n## Features\n- 90+ search engines\n- BM25 ranking\n- Concurrent fetching",
)
profile.key_features = _extract_features_from_readme(profile.readme_summary)
profile.target_audience = _infer_target_audience(profile)
profile.promotion_hooks = _generate_promotion_hooks(profile)

print(f"\n  RepoProfile: {profile.name}")
print(f"  Features: {profile.key_features}")
print(f"  Audience: {profile.target_audience}")
print(f"  Hooks: {profile.promotion_hooks[:2]}")

# 3. Portfolio context
ctx = build_portfolio_context([profile])
print(f"\n  Portfolio context: {len(ctx)} chars")

# 4. CampaignContext with new fields
campaign = CampaignContext(
    campaign_id="test", campaign_name="Test",
    objective="Promote SearchWala", target_platforms=["reddit"],
    session_id="test",
    github_username="SandeepAi369",
    promotion_style="organic",
)
print(f"  CampaignContext.github_username: {campaign.github_username}")
print(f"  CampaignContext.promotion_style: {campaign.promotion_style}")

# 5. SupervisorInternalState with new fields
state = SupervisorInternalState(
    github_username="SandeepAi369",
    promotion_style="organic",
    portfolio_context=ctx,
)
print(f"  SupervisorInternalState.portfolio_context: {len(state.portfolio_context)} chars")

print("\n=== ALL IMPORTS PASSED ===")
