"""GitHub Community Engagement — Value-First, Link-Second strategy.

Implements organic developer community engagement by:
1. Navigating to target repositories
2. Reading their READMEs to understand the project
3. Drafting contextual, value-driven issues or discussion comments
4. Dropping the SearchWala link ONLY after providing genuine value

STRICT RULES:
- First comment must be 100% value, zero links
- Link is only placed in the 2nd or 3rd interaction
- All interactions use GhostCursor (Bézier + human typing)
- "Value First, Link Second" strategy is enforced programmatically
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from app.logger import get_logger

logger = get_logger("github_engagement")


# ═══════════════════════════════════════════════════════════════════════════════
#  Engagement Strategy Templates
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EngagementTarget:
    """A GitHub repository to engage with."""
    owner: str
    repo: str
    url: str
    niche: str = ""           # e.g., "search-engine", "rust-tools", "web-scraping"
    readme_summary: str = ""  # Filled after reading the repo
    engagement_type: str = "discussion"  # "issue", "discussion", "pr_comment"


class EngagementDraft(BaseModel):
    """LLM-generated engagement content."""
    title: str = Field(description="Title of the issue/discussion (if applicable)")
    body: str = Field(description="The comment/issue body")
    is_value_only: bool = Field(
        default=True,
        description="True if this is a pure value comment with no self-promotion"
    )
    relevance_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How relevant our project is to this repo"
    )


# Curated list of target repository niches that SearchWala is relevant to
TARGET_NICHES = [
    {
        "niche": "meta-search",
        "repos": [
            "searxng/searxng",
            "benbusby/whoogle-search",
            "prabhatsharma/zinc",
        ],
        "value_angle": "search engine architecture, ranking algorithms, engine integration",
    },
    {
        "niche": "rust-web",
        "repos": [
            "nickel-org/nickel.rs",
            "SergioBenitez/Rocket",
            "tokio-rs/axum",
        ],
        "value_angle": "Rust async web service patterns, Tokio runtime optimization",
    },
    {
        "niche": "web-scraping",
        "repos": [
            "nicehash/Excavator",
            "nicedoc/crawlee",
            "AetheraelSolutions/aetherscraper",
        ],
        "value_angle": "browser fingerprint rotation, anti-detection, concurrent scraping",
    },
    {
        "niche": "llm-tools",
        "repos": [
            "langchain-ai/langchain",
            "run-llama/llama_index",
        ],
        "value_angle": "LLM-powered search synthesis, RAG pipeline optimization",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Value-First Engagement Generator
# ═══════════════════════════════════════════════════════════════════════════════

class GitHubEngagement:
    """Generates value-first engagement content for target repositories.
    
    Usage:
        eng = GitHubEngagement()
        targets = eng.select_targets(max_targets=3)
        for target in targets:
            draft = await eng.draft_engagement(target, failover_chain, invoke_fn)
    """

    # Strict link injection rules
    LINK_INJECTION_RULES = {
        "first_comment": "NEVER include any links to your own projects.",
        "second_comment": "You MAY casually mention your project name if contextually relevant.",
        "third_comment": "You MAY include a GitHub link IF the discussion naturally leads to it.",
    }

    def __init__(self, project_url: str = "https://github.com/SandeepAi369/SearchWala"):
        self.project_url = project_url
        self._engagement_count: dict[str, int] = {}  # repo → interaction count

    def select_targets(self, max_targets: int = 3) -> list[EngagementTarget]:
        """Select target repositories to engage with."""
        targets = []
        for niche_data in TARGET_NICHES:
            for repo_path in niche_data["repos"]:
                parts = repo_path.split("/")
                if len(parts) == 2:
                    targets.append(EngagementTarget(
                        owner=parts[0],
                        repo=parts[1],
                        url=f"https://github.com/{repo_path}",
                        niche=niche_data["niche"],
                    ))

        # Shuffle and limit
        random.shuffle(targets)
        selected = targets[:max_targets]
        logger.info("Selected %d engagement targets: %s",
                     len(selected), [t.url for t in selected])
        return selected

    def get_interaction_stage(self, repo_key: str) -> str:
        """Get the current interaction stage for a repo (controls link injection)."""
        count = self._engagement_count.get(repo_key, 0)
        if count == 0:
            return "first_comment"
        elif count == 1:
            return "second_comment"
        else:
            return "third_comment"

    def record_interaction(self, repo_key: str):
        """Record that we interacted with a repository."""
        self._engagement_count[repo_key] = self._engagement_count.get(repo_key, 0) + 1

    async def draft_engagement(
        self,
        target: EngagementTarget,
        failover_chain: list,
        invoke_fn,
        readme_content: str = "",
    ) -> EngagementDraft:
        """Generate a value-first engagement draft for a target repository.
        
        Args:
            target: The repository to engage with
            failover_chain: LLM failover chain
            invoke_fn: The _invoke_with_failover function
            readme_content: Optional README content (scraped by the agent)
        """
        repo_key = f"{target.owner}/{target.repo}"
        stage = self.get_interaction_stage(repo_key)
        link_rule = self.LINK_INJECTION_RULES[stage]

        # Find the niche data for the value angle
        value_angle = ""
        for niche_data in TARGET_NICHES:
            if f"{target.owner}/{target.repo}" in niche_data["repos"]:
                value_angle = niche_data["value_angle"]
                break

        system_prompt = (
            "You are a genuine open-source developer engaging with the community.\n"
            "You are writing a comment or discussion post on a GitHub repository.\n\n"
            "ABSOLUTE RULES:\n"
            f"- LINK INJECTION RULE: {link_rule}\n"
            "- Sound like a real developer, not a marketer\n"
            "- Be specific and technical, not generic\n"
            "- Ask genuine questions if you are curious\n"
            "- Share real technical insights from your experience\n"
            "- Keep it concise (150-300 words max)\n"
            "- Use first person ('I', 'my', 'I have been building')\n"
            "- If mentioning your project, explain HOW it relates, not just THAT it exists\n\n"
            f"YOUR BACKGROUND: You are building SearchWala, a Rust-based meta-search engine.\n"
            f"YOUR EXPERTISE: {value_angle}\n"
        )

        readme_context = ""
        if readme_content:
            # Truncate README to save tokens
            readme_context = f"\nTARGET REPO README (first 1500 chars):\n{readme_content[:1500]}\n"

        user_prompt = (
            f"TARGET REPOSITORY: {target.owner}/{target.repo}\n"
            f"REPO URL: {target.url}\n"
            f"NICHE: {target.niche}\n"
            f"INTERACTION STAGE: {stage}\n"
            f"{readme_context}\n\n"
            "Draft a valuable, authentic engagement post for this repository's Discussions section.\n"
            "Remember: VALUE FIRST, never spam."
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        try:
            draft, model = await invoke_fn(failover_chain, messages, EngagementDraft)
            logger.info("GitHub engagement draft for %s (stage=%s, model=%s)",
                        repo_key, stage, model)

            # Enforce link injection rules programmatically
            if stage == "first_comment" and self.project_url in draft.body:
                logger.warning("BLOCKED: Link detected in first_comment stage — removing")
                draft.body = draft.body.replace(self.project_url, "[my project]")
                draft.body = draft.body.replace("SearchWala", "my search engine project")
                draft.is_value_only = True

            self.record_interaction(repo_key)
            return draft

        except Exception as e:
            logger.error("GitHub engagement draft failed: %s", e)
            # Return a safe fallback
            return EngagementDraft(
                title="",
                body="",
                is_value_only=True,
                relevance_score=0.0,
            )
