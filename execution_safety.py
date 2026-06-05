"""Execution-layer safety guard (arXiv 2511.19477 / ST-WebAgentBench 2410.06703).

The CODE enforces safety constraints the LLM cannot override:
- Domain allowlist: only navigate to pre-approved domains
- CoVe pre-done: verify completion signals before declaring done

Research Sources:
  - Building Browser Agents (arXiv 2511.19477): deterministic safety guards
  - ST-WebAgentBench (arXiv 2410.06703): safe task execution benchmarks
  - CoVe (arXiv 2309.11495): chain-of-verification before high-impact actions
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger("execution_safety")


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain Allowlist — code-level enforcement the LLM cannot override
# ═══════════════════════════════════════════════════════════════════════════════

ALLOWED_DOMAINS: set[str] = {
    # Social media
    "x.com", "twitter.com",
    "reddit.com", "old.reddit.com",
    "facebook.com", "www.facebook.com",
    "instagram.com", "www.instagram.com",
    "linkedin.com", "www.linkedin.com",
    # Publishing platforms
    "dev.to",
    "medium.com",
    "hashnode.com",
    "substack.com",
    "wordpress.com",
    "blogger.com",
    # Common utilities
    "google.com", "www.google.com",
    "github.com",
    "youtube.com", "www.youtube.com",
    "accounts.google.com",
}


def is_domain_allowed(url: str) -> bool:
    """Check if the URL's domain is in the allowlist.

    Matches exact domain OR is a subdomain of an allowed domain.
    Returns True for empty/relative URLs (internal navigation).
    """
    if not url or not url.startswith(("http://", "https://")):
        return True  # Relative URLs are fine
    try:
        host = urlparse(url).hostname
        if not host:
            return True
        # Check exact match or subdomain match
        for allowed in ALLOWED_DOMAINS:
            if host == allowed or host.endswith("." + allowed):
                return True
        return False
    except Exception:
        return False


def add_allowed_domain(domain: str) -> None:
    """Add a domain to the allowlist at runtime."""
    ALLOWED_DOMAINS.add(domain.lower().strip())
    logger.info("🛡️ Domain added to allowlist: %s", domain)


# ═══════════════════════════════════════════════════════════════════════════════
#  CoVe Pre-Done Check (Chain-of-Verification, arXiv 2309.11495)
#  Verify completion signals BEFORE declaring done — kills premature done.
# ═══════════════════════════════════════════════════════════════════════════════

async def cove_pre_done_check(page, plan, working_mem) -> tuple[bool, str]:
    """Chain-of-Verification before declaring 'done' (CoVe 2309.11495).

    Checks three signals:
    1. Plan completion: Are all plan steps done?
    2. Progress threshold: Has sufficient progress been made?
    3. Page state: Is the current page consistent with completion?

    Returns:
        (True, reason) if done is valid
        (False, reason) if done should be blocked
    """
    reasons_to_block: list[str] = []

    # Check 1: Plan completion
    if not plan.is_complete:
        remaining = [s for s in plan.steps if s["status"] not in ("done", "failed")]
        if len(remaining) > 1:
            reasons_to_block.append(
                f"Plan incomplete: {len(remaining)} steps remaining "
                f"(next: '{remaining[0]['desc'][:40]}')"
            )

    # Check 2: Progress threshold — at least 40% must be done
    if plan.progress_pct < 40:
        reasons_to_block.append(
            f"Only {plan.progress_pct}% progress — too early to declare done"
        )

    # Check 3: Page state — check if we're on a plausible completion page
    try:
        url = page.url
        # If still on a login page, definitely not done
        if any(kw in url.lower() for kw in ["login", "signin", "auth", "signup"]):
            reasons_to_block.append(
                f"Still on auth page ({url[:50]})"
            )
    except Exception:
        pass

    if reasons_to_block:
        return False, "; ".join(reasons_to_block)
    return True, f"All checks passed (progress={plan.progress_pct}%)"
