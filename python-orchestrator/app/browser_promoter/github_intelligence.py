"""GitHub Intelligence Module — Autonomous repo analysis for marketing.

Fetches public repositories via GitHub API, parses READMEs, and builds
structured RepoProfile objects with promotion hooks. Results are cached
in SQLite for 24 hours to avoid redundant API calls.

Usage:
    intelligence = GitHubIntelligence(username="SandeepAi369", token="ghp_...")
    profiles = await intelligence.get_repo_profiles()
    for profile in profiles:
        print(profile.name, profile.promotion_hooks)
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from app import config
from app.browser_promoter.database import (
    get_cached_repos,
    upsert_repo_cache,
)
from app.logger import get_logger

logger = get_logger(__name__)

# Repos that are utility/profile repos — skip promotion by default
_SKIP_REPOS = frozenset({
    "sandeepai369",     # GitHub profile README
    "uptime-keeper",    # Utility cron job
})


@dataclass
class RepoProfile:
    """Structured intelligence about a single repository for promotion."""

    name: str
    full_name: str
    url: str
    description: str = ""
    language: str = ""
    stars: int = 0
    forks: int = 0
    topics: list[str] = field(default_factory=list)
    homepage: str = ""
    readme_summary: str = ""
    key_features: list[str] = field(default_factory=list)
    target_audience: str = ""
    promotion_hooks: list[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Serialize to a compact string for LLM context injection."""
        lines = [
            f"REPO: {self.name} ({self.language}) — ⭐{self.stars}",
            f"  URL: {self.url}",
        ]
        if self.homepage:
            lines.append(f"  LIVE: {self.homepage}")
        if self.description:
            lines.append(f"  DESC: {self.description}")
        if self.key_features:
            lines.append(f"  FEATURES: {', '.join(self.key_features[:5])}")
        if self.target_audience:
            lines.append(f"  AUDIENCE: {self.target_audience}")
        if self.promotion_hooks:
            lines.append(f"  HOOKS: {' | '.join(self.promotion_hooks[:3])}")
        if self.topics:
            lines.append(f"  TOPICS: {', '.join(self.topics[:8])}")
        return "\n".join(lines)


class GitHubIntelligence:
    """Fetches and analyzes GitHub repos for promotion intelligence."""

    def __init__(
        self,
        username: str = "",
        token: str = "",
        cache_ttl_hours: int = 0,
    ) -> None:
        self.username = username or config.GITHUB_USERNAME
        self.token = token or config.GITHUB_TOKEN
        self.cache_ttl_hours = cache_ttl_hours or config.GITHUB_REPO_CACHE_TTL_HOURS
        self._headers = self._build_headers()

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": f"AgentFirstIDE/{self.username}",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_repo_profiles(
        self,
        *,
        filter_repos: list[str] | None = None,
    ) -> list[RepoProfile]:
        """Return promotion-ready repo profiles, using cache when fresh.

        Args:
            filter_repos: If provided, only return profiles for these repo names.
                          Empty list = return all promotable repos.
        """
        # 1. Try cache first
        cached = get_cached_repos(self.username, max_age_hours=self.cache_ttl_hours)
        if cached:
            logger.info(
                "GitHub intelligence cache hit: %d repos for %s",
                len(cached), self.username,
            )
            profiles = [self._row_to_profile(row) for row in cached]
            return self._apply_filter(profiles, filter_repos)

        # 2. Fetch fresh from GitHub API
        logger.info("Fetching fresh repo data from GitHub API for %s", self.username)
        profiles = await self._fetch_and_analyze_repos()

        # 3. Cache results
        for profile in profiles:
            try:
                upsert_repo_cache(
                    username=self.username,
                    repo_name=profile.name,
                    full_name=profile.full_name,
                    url=profile.url,
                    description=profile.description,
                    language=profile.language,
                    stars=profile.stars,
                    forks=profile.forks,
                    topics_json=json.dumps(profile.topics),
                    readme_summary=profile.readme_summary,
                    key_features_json=json.dumps(profile.key_features),
                    target_audience=profile.target_audience,
                    promotion_hooks_json=json.dumps(profile.promotion_hooks),
                    homepage=profile.homepage,
                )
            except Exception as exc:
                logger.warning("Cache write failed for %s: %s", profile.name, exc)

        return self._apply_filter(profiles, filter_repos)

    async def _fetch_and_analyze_repos(self) -> list[RepoProfile]:
        """Fetch all public repos and build profiles with README analysis."""
        raw_repos = await self._fetch_repos()
        profiles: list[RepoProfile] = []

        for repo_data in raw_repos:
            name = repo_data.get("name", "")
            if name.lower() in _SKIP_REPOS:
                logger.info("Skipping utility/profile repo: %s", name)
                continue

            profile = self._parse_repo_data(repo_data)

            # Fetch README for deeper analysis
            readme_text = await self._fetch_readme(profile.full_name)
            if readme_text:
                profile.readme_summary = readme_text[:800]
                profile.key_features = _extract_features_from_readme(readme_text)
                profile.target_audience = _infer_target_audience(profile)
                profile.promotion_hooks = _generate_promotion_hooks(profile)

            profiles.append(profile)

        logger.info(
            "GitHub Intelligence: analyzed %d promotable repos for %s",
            len(profiles), self.username,
        )
        return profiles

    async def _fetch_repos(self) -> list[dict[str, Any]]:
        """Fetch public repos from GitHub API."""
        url = f"https://api.github.com/users/{self.username}/repos"
        params = {"type": "public", "sort": "updated", "per_page": "30"}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=self._headers, params=params)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            logger.error("GitHub API fetch failed: %s", exc)
            return []

    async def _fetch_readme(self, full_name: str) -> str:
        """Fetch and decode README content for a repo."""
        url = f"https://api.github.com/repos/{full_name}/readme"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=self._headers)
                if response.status_code != 200:
                    return ""
                data = response.json()
                content = data.get("content", "")
                encoding = data.get("encoding", "")
                if encoding == "base64" and content:
                    return base64.b64decode(content).decode("utf-8", errors="replace")
                return content
        except Exception as exc:
            logger.warning("README fetch failed for %s: %s", full_name, exc)
            return ""

    def _parse_repo_data(self, data: dict[str, Any]) -> RepoProfile:
        """Parse GitHub API repo JSON into a RepoProfile."""
        return RepoProfile(
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            url=data.get("html_url", ""),
            description=data.get("description", "") or "",
            language=data.get("language", "") or "",
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            topics=data.get("topics", []),
            homepage=data.get("homepage", "") or "",
        )

    @staticmethod
    def _row_to_profile(row: dict) -> RepoProfile:
        """Convert a SQLite cache row back to a RepoProfile."""
        return RepoProfile(
            name=row["repo_name"],
            full_name=row["full_name"],
            url=row["url"],
            description=row.get("description", ""),
            language=row.get("language", ""),
            stars=row.get("stars", 0),
            forks=row.get("forks", 0),
            topics=json.loads(row.get("topics_json", "[]")),
            homepage=row.get("homepage", ""),
            readme_summary=row.get("readme_summary", ""),
            key_features=json.loads(row.get("key_features_json", "[]")),
            target_audience=row.get("target_audience", ""),
            promotion_hooks=json.loads(row.get("promotion_hooks_json", "[]")),
        )

    @staticmethod
    def _apply_filter(
        profiles: list[RepoProfile],
        filter_repos: list[str] | None,
    ) -> list[RepoProfile]:
        """Filter profiles by repo name if a filter list is provided."""
        if not filter_repos:
            return profiles
        filter_set = {name.lower() for name in filter_repos}
        return [p for p in profiles if p.name.lower() in filter_set]


# ═══════════════════════════════════════════════════════════════════════════════
#  Static analysis helpers (no LLM needed — fast heuristic extraction)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_features_from_readme(readme_text: str) -> list[str]:
    """Extract key features from README using heuristic patterns.

    Looks for bullet points under common feature headings like
    "Features", "Highlights", "What it does", etc.
    """
    features: list[str] = []
    lines = readme_text.split("\n")
    in_feature_section = False

    feature_headers = {
        "features", "highlights", "what it does", "key features",
        "capabilities", "why", "what makes it special",
    }

    for line in lines:
        stripped = line.strip()

        # Detect feature section headers
        if stripped.startswith("#"):
            header_text = stripped.lstrip("#").strip().lower()
            in_feature_section = any(h in header_text for h in feature_headers)
            continue

        # Collect bullet points in feature sections
        if in_feature_section and (stripped.startswith("- ") or stripped.startswith("* ")):
            feature_text = stripped.lstrip("-* ").strip()
            if 5 < len(feature_text) < 200:
                features.append(feature_text)
                if len(features) >= 8:
                    break

        # Stop if we hit another section header
        if in_feature_section and stripped.startswith("#"):
            break

    # Fallback: extract from description or first meaningful lines
    if not features:
        for line in lines[:20]:
            stripped = line.strip()
            if (stripped.startswith("- ") or stripped.startswith("* ")) and 10 < len(stripped) < 200:
                features.append(stripped.lstrip("-* ").strip())
                if len(features) >= 5:
                    break

    return features


def _infer_target_audience(profile: RepoProfile) -> str:
    """Infer who would benefit from this project based on metadata."""
    lang = profile.language.lower()
    desc = profile.description.lower()
    topics = [t.lower() for t in profile.topics]
    all_signals = f"{desc} {' '.join(topics)}"

    audiences = []

    # Language-based audience
    lang_audiences = {
        "rust": "Rust developers and systems programmers",
        "typescript": "Full-stack developers and frontend engineers",
        "python": "Python developers and data/AI engineers",
        "javascript": "Web developers",
        "go": "Backend engineers and DevOps teams",
    }
    if lang in lang_audiences:
        audiences.append(lang_audiences[lang])

    # Topic-based audience
    if any(kw in all_signals for kw in ("search", "aggregator", "meta-search")):
        audiences.append("developers needing fast search APIs")
    if any(kw in all_signals for kw in ("ai", "ml", "intelligence", "neural")):
        audiences.append("AI/ML practitioners")
    if any(kw in all_signals for kw in ("bot", "telegram", "discord", "chat")):
        audiences.append("chatbot builders and community managers")
    if any(kw in all_signals for kw in ("security", "cyber", "pentest", "hacking")):
        audiences.append("cybersecurity professionals")
    if any(kw in all_signals for kw in ("studio", "editor", "ide", "tool")):
        audiences.append("developer tool users")

    return "; ".join(audiences[:3]) if audiences else "developers and tech enthusiasts"


def _generate_promotion_hooks(profile: RepoProfile) -> list[str]:
    """Generate natural talking points for promoting this repo.

    These hooks are designed to sound like genuine developer experiences,
    not marketing copy.
    """
    hooks: list[str] = []
    name = profile.name
    lang = profile.language

    # Performance hook (for Rust/Go/C++)
    if lang.lower() in ("rust", "go", "c++", "c"):
        hooks.append(
            f"Built {name} in {lang} for raw performance — "
            f"it handles edge cases that crash slower alternatives"
        )

    # Open source hook
    hooks.append(
        f"Open-sourced {name} after using it internally — "
        f"figured others might find it useful too"
    )

    # Feature-specific hooks from README
    for feature in profile.key_features[:2]:
        hooks.append(
            f"One thing I like about {name}: {feature.lower()}"
        )

    # Stars social proof (if any)
    if profile.stars >= 3:
        hooks.append(
            f"{name} just passed {profile.stars} stars — "
            f"small wins but it's growing organically"
        )

    # Live demo hook
    if profile.homepage:
        hooks.append(
            f"You can try {name} live at {profile.homepage} — "
            f"no setup needed"
        )

    # Problem-solution hook from description
    if profile.description and len(profile.description) > 20:
        hooks.append(
            f"I built {name} because {profile.description.lower()}"
        )

    return hooks[:5]
