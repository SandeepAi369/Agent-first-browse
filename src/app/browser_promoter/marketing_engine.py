"""Marketing Strategy Engine — Autonomous promotion planning.

Decides WHERE and HOW to promote repositories across platforms.
Implements safety controls: cooldown tracking, rate limiting,
and anti-spam content validation.

Strategies:
  - question_answering: Find questions where the project is the answer
  - show_and_tell: Post in "Share Your Project" threads
  - relevant_comment: Add value to existing discussions with natural mention
  - community_building: Star related projects, follow relevant devs
  - content_seeding: Reference articles/docs in technical threads

Usage:
    engine = MarketingEngine()
    plans = await engine.generate_promotion_plans(profiles, platform="reddit")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app import config
from app.browser_promoter.database import (
    check_promotion_cooldown,
    count_platform_promotions_today,
    record_promotion,
)
from app.browser_promoter.github_intelligence import RepoProfile
from app.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Platform Channel Maps — where to promote each type of project
# ═══════════════════════════════════════════════════════════════════════════════

# Maps (language/topic) → best subreddits/communities
_REDDIT_CHANNELS: dict[str, list[str]] = {
    "rust": ["r/rust", "r/programming", "r/selfhosted", "r/opensource"],
    "python": ["r/Python", "r/learnpython", "r/programming", "r/selfhosted"],
    "typescript": ["r/webdev", "r/reactjs", "r/nextjs", "r/programming"],
    "javascript": ["r/webdev", "r/javascript", "r/programming"],
    "search": ["r/selfhosted", "r/degoogle", "r/privacy", "r/opensource"],
    "ai": ["r/MachineLearning", "r/artificial", "r/LocalLLaMA", "r/programming"],
    "bot": ["r/Telegram", "r/chatbots", "r/Python"],
    "security": ["r/netsec", "r/hacking", "r/cybersecurity"],
    "general": ["r/programming", "r/opensource", "r/sideproject", "r/coolgithubprojects"],
}

_GITHUB_CHANNELS: dict[str, list[str]] = {
    "rust": ["rust-lang/rust discussions", "trending/rust"],
    "python": ["python/cpython discussions", "trending/python"],
    "search": ["searxng/searxng discussions", "awesome-selfhosted"],
    "ai": ["langchain-ai discussions", "trending/ai"],
    "general": ["github/explore", "trending"],
}

_HN_CHANNELS: dict[str, list[str]] = {
    "general": ["Show HN", "Ask HN: Share Your Side Project"],
}


@dataclass
class PromotionPlan:
    """A single promotion action plan ready for execution."""

    repo: RepoProfile
    platform: str               # "reddit" | "github" | "hackernews"
    channel: str                # "r/rust" | "Show HN" | specific URL
    tactic: str                 # "question_answering" | "show_and_tell" | etc.
    content_angle: str          # "performance comparison" | "personal experience"
    draft_message: str          # Generated human-like message
    search_query: str = ""      # Query to find relevant discussions
    safety_score: float = 0.5   # 0.0-1.0 anti-spam risk assessment
    blocked_reason: str = ""    # If blocked by cooldown/rate limit

    def is_blocked(self) -> bool:
        return bool(self.blocked_reason)

    def to_objective_string(self) -> str:
        """Convert to a campaign objective the supervisor can execute."""
        lines = [
            f"PROMOTE {self.repo.name} on {self.platform} ({self.channel})",
            f"TACTIC: {self.tactic}",
            f"ANGLE: {self.content_angle}",
        ]
        if self.search_query:
            lines.append(f"SEARCH: {self.search_query}")
        lines.append(f"DRAFT: {self.draft_message}")
        lines.append(
            f"SAFETY: Score={self.safety_score:.1f} — "
            f"Use natural tone, add genuine value, don't just drop a link."
        )
        return "\n".join(lines)


class MarketingEngine:
    """Generates and validates promotion plans for repositories."""

    def __init__(self) -> None:
        self.max_per_platform = config.MAX_PROMOTIONS_PER_PLATFORM
        self.cooldown_days = config.PROMOTION_COOLDOWN_DAYS
        self.style = config.PROMOTION_STYLE
        self.allowed_platforms = config.PROMOTION_PLATFORMS

    async def generate_promotion_plans(
        self,
        profiles: list[RepoProfile],
        *,
        platform: str = "",
        max_plans: int = 5,
    ) -> list[PromotionPlan]:
        """Generate promotion plans for the given repos.

        Args:
            profiles: Repo profiles from GitHubIntelligence
            platform: Target a specific platform (empty = all allowed)
            max_plans: Maximum number of plans to generate
        """
        platforms = [platform] if platform else self.allowed_platforms
        all_plans: list[PromotionPlan] = []

        for profile in profiles:
            for plat in platforms:
                if plat not in self.allowed_platforms:
                    continue

                plans = self._generate_plans_for_repo(profile, plat)
                for plan in plans:
                    plan = self._apply_safety_checks(plan)
                    all_plans.append(plan)

        # Sort by safety score (highest first) and limit
        all_plans.sort(key=lambda p: (not p.is_blocked(), p.safety_score), reverse=True)
        result = all_plans[:max_plans]

        logger.info(
            "Marketing Engine: generated %d plans (%d blocked) from %d repos",
            len(result),
            sum(1 for p in result if p.is_blocked()),
            len(profiles),
        )
        return result

    def _generate_plans_for_repo(
        self, profile: RepoProfile, platform: str,
    ) -> list[PromotionPlan]:
        """Generate platform-specific promotion plans for one repo."""
        plans: list[PromotionPlan] = []
        channels = self._get_channels(profile, platform)

        for channel in channels[:2]:  # Max 2 channels per platform per repo
            tactic, angle, query = self._select_tactic(profile, platform, channel)
            draft = self._generate_draft(profile, tactic, angle, platform)

            plans.append(PromotionPlan(
                repo=profile,
                platform=platform,
                channel=channel,
                tactic=tactic,
                content_angle=angle,
                draft_message=draft,
                search_query=query,
                safety_score=self._calculate_safety_score(profile, tactic, platform),
            ))

        return plans

    def _get_channels(self, profile: RepoProfile, platform: str) -> list[str]:
        """Determine best channels for this repo on the given platform."""
        lang = profile.language.lower()
        topics = [t.lower() for t in profile.topics]
        desc = profile.description.lower()

        # Determine category
        category = "general"
        if "search" in desc or any("search" in t for t in topics):
            category = "search"
        elif "bot" in desc or "telegram" in desc:
            category = "bot"
        elif "ai" in desc or "intelligence" in desc or any("ai" in t for t in topics):
            category = "ai"
        elif "security" in desc or "cyber" in desc:
            category = "security"
        elif lang in ("rust", "python", "typescript", "javascript"):
            category = lang

        if platform == "reddit":
            channels = _REDDIT_CHANNELS.get(category, _REDDIT_CHANNELS["general"])
        elif platform == "github":
            channels = _GITHUB_CHANNELS.get(category, _GITHUB_CHANNELS["general"])
        elif platform == "hackernews":
            channels = _HN_CHANNELS.get("general", ["Show HN"])
        else:
            channels = ["general"]

        return channels

    def _select_tactic(
        self, profile: RepoProfile, platform: str, channel: str,
    ) -> tuple[str, str, str]:
        """Select the best tactic, content angle, and search query."""
        name = profile.name
        lang = profile.language
        desc = profile.description

        if self.style == "educational":
            return (
                "content_seeding",
                "technical_insight",
                f"site:reddit.com {lang} tutorial OR guide OR how to",
            )

        if self.style == "direct":
            return (
                "show_and_tell",
                "project_showcase",
                f"site:reddit.com show project OR share your project OR side project",
            )

        # Organic style (default) — varies by platform
        if platform == "hackernews":
            return (
                "show_and_tell",
                "personal_experience",
                "",
            )

        if platform == "github":
            return (
                "relevant_comment",
                "technical_comparison",
                f"{name} OR {desc[:40]}",
            )

        # Reddit: question answering is highest signal
        query_keywords = _build_search_keywords(profile)
        return (
            "question_answering",
            "personal_experience",
            f"site:reddit.com {channel} {query_keywords}",
        )

    def _generate_draft(
        self,
        profile: RepoProfile,
        tactic: str,
        angle: str,
        platform: str,
    ) -> str:
        """Generate a human-like draft message for the promotion.

        This generates a TEMPLATE — the Supervisor's copywriter sub-agent
        will refine it further based on the actual discussion context.
        """
        name = profile.name
        url = profile.url
        desc = profile.description
        hooks = profile.promotion_hooks

        if tactic == "question_answering":
            hook = hooks[0] if hooks else f"I built {name} to solve this"
            return (
                f"I ran into the same issue. Ended up building something for it — "
                f"{desc.lower()[:100]}. "
                f"It's open source if you want to try: {url}"
            )

        if tactic == "show_and_tell":
            features = ", ".join(profile.key_features[:3]) if profile.key_features else desc[:100]
            return (
                f"Been working on {name} — {desc.lower()[:80]}. "
                f"Key things: {features}. "
                f"Would love feedback: {url}"
            )

        if tactic == "relevant_comment":
            hook = hooks[1] if len(hooks) > 1 else f"I've been using {name} for this"
            return (
                f"Interesting thread. {hook.capitalize()}. "
                f"Source: {url}"
            )

        if tactic == "content_seeding":
            return (
                f"Wrote about this topic recently. "
                f"The approach I took: {desc.lower()[:100]}. "
                f"Full write-up and code: {url}"
            )

        return f"Check out {name}: {url} — {desc[:100]}"

    def _calculate_safety_score(
        self, profile: RepoProfile, tactic: str, platform: str,
    ) -> float:
        """Calculate anti-spam safety score (0.0 = risky, 1.0 = safe)."""
        score = 0.5  # baseline

        # Question answering is safest (adds genuine value)
        if tactic == "question_answering":
            score += 0.2
        elif tactic == "relevant_comment":
            score += 0.1
        elif tactic == "show_and_tell":
            score += 0.0  # neutral
        elif tactic == "content_seeding":
            score += 0.15

        # More stars = more credible
        if profile.stars >= 5:
            score += 0.1
        if profile.stars >= 20:
            score += 0.1

        # Has live demo = more useful to community
        if profile.homepage:
            score += 0.05

        # Has substantial description
        if len(profile.description) > 40:
            score += 0.05

        return min(1.0, max(0.0, score))

    def _apply_safety_checks(self, plan: PromotionPlan) -> PromotionPlan:
        """Apply cooldown and rate limit safety checks."""
        # Check cooldown
        if check_promotion_cooldown(
            repo_name=plan.repo.name,
            platform=plan.platform,
            channel=plan.channel,
            cooldown_days=self.cooldown_days,
        ):
            plan.blocked_reason = (
                f"Cooldown active: {plan.repo.name} was promoted in "
                f"{plan.channel} within last {self.cooldown_days} days"
            )
            return plan

        # Check platform rate limit
        today_count = count_platform_promotions_today(platform=plan.platform)
        if today_count >= self.max_per_platform:
            plan.blocked_reason = (
                f"Rate limit: {today_count}/{self.max_per_platform} "
                f"promotions already done on {plan.platform} today"
            )
            return plan

        return plan

    def record_execution(self, plan: PromotionPlan, status: str = "executed") -> int:
        """Record a promotion execution for future cooldown tracking."""
        return record_promotion(
            repo_name=plan.repo.name,
            platform=plan.platform,
            channel=plan.channel,
            tactic=plan.tactic,
            content_angle=plan.content_angle,
            draft_message=plan.draft_message,
            status=status,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _build_search_keywords(profile: RepoProfile) -> str:
    """Build search keywords from repo metadata for finding relevant discussions."""
    keywords: list[str] = []

    # From topics
    for topic in profile.topics[:3]:
        clean = topic.replace("-", " ").replace("_", " ")
        keywords.append(clean)

    # From language
    if profile.language:
        keywords.append(profile.language.lower())

    # From description
    desc_words = profile.description.lower().split()
    stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is", "it"}
    for word in desc_words[:10]:
        clean = word.strip(".,!?()[]{}\"'")
        if len(clean) > 3 and clean not in stop_words:
            keywords.append(clean)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)

    return " OR ".join(unique[:5])


def build_portfolio_context(profiles: list[RepoProfile]) -> str:
    """Build a compact portfolio context string for LLM injection.

    This is injected into the Supervisor's strategy planner so it has
    real product knowledge to work with.
    """
    if not profiles:
        return "No GitHub repos available for promotion."

    lines = [
        f"=== GITHUB PORTFOLIO ({len(profiles)} repos) ===",
        "",
    ]
    for i, profile in enumerate(profiles, 1):
        lines.append(f"[{i}] {profile.to_context_string()}")
        lines.append("")

    return "\n".join(lines)
