"""Execution-layer safety guard (arXiv 2511.19477 / ST-WebAgentBench 2410.06703).

The CODE enforces safety constraints the LLM cannot override:
- Domain policy: block dangerous domains, auto-allow from objective
- CoVe pre-done: verify completion signals before declaring done
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("execution_safety")

# Default folder the agent may upload from (override: AGENT_UPLOAD_DIRS, os.pathsep-separated).
DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"

# The user's objective — the one input a page cannot inject into. Files named in
# it verbatim are user-approved uploads. Set by auto_allow_from_objective().
_OBJECTIVE_TEXT: str = ""


# Dangerous domains that should NEVER be automated
BLOCKED_DOMAINS: set[str] = {
    # Banking & financial
    "chase.com", "bankofamerica.com", "wellsfargo.com",
    "citibank.com", "capitalone.com",
    # Payment processors
    "paypal.com", "venmo.com", "stripe.com",
    # Government
    "irs.gov", "ssa.gov",
}

# Domains always allowed (search engines, etc.)
ALWAYS_ALLOWED: set[str] = {
    "google.com", "www.google.com", "accounts.google.com",
    "bing.com", "www.bing.com",
    "duckduckgo.com",
}

# Dynamic allowlist — populated at runtime from objective URLs
_RUNTIME_ALLOWED: set[str] = set()


def auto_allow_from_objective(objective: str) -> None:
    """Extract URLs from the objective and add their domains to the runtime allowlist.

    Also records the objective so is_upload_allowed() can honour files it names.
    """
    global _OBJECTIVE_TEXT
    _OBJECTIVE_TEXT = objective or ""
    urls = re.findall(r'https?://([^/\s,)\"\'>]+)', objective)
    for url_host in urls:
        domain = url_host.lower().strip().removeprefix("www.")
        if domain and domain not in BLOCKED_DOMAINS:
            _RUNTIME_ALLOWED.add(domain)
            _RUNTIME_ALLOWED.add("www." + domain)
            # Also allow subdomains
            parts = domain.split(".")
            if len(parts) >= 2:
                base = ".".join(parts[-2:])
                _RUNTIME_ALLOWED.add(base)
                _RUNTIME_ALLOWED.add("www." + base)
            logger.info("🛡️ Auto-allowed domain from objective: %s", domain)


def is_domain_allowed(url: str) -> bool:
    """Check if the URL is safe to navigate to.

    Policy: only http(s) URLs (minus the blocked domains) and about:blank.
    Every other scheme — file:, javascript:, data:, chrome:, view-source:, … —
    is refused: a page-injected instruction could otherwise steer the agent to
    file:///…/.env or ~/.ssh and pull local secrets into the DOM it sends to LLMs.
    """
    url = (url or "").strip()
    if url.lower() == "about:blank":
        return True
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            logger.warning("🛡️ BLOCKED non-http(s) URL: %s", url[:80])
            return False
        host = parsed.hostname
        if not host:
            return False
        host = host.lower()
        # Check blocked list
        for blocked in BLOCKED_DOMAINS:
            if host == blocked or host.endswith("." + blocked):
                logger.warning("🛡️ BLOCKED dangerous domain: %s", host)
                return False
        # Everything else is allowed
        return True
    except Exception:
        return False


def add_allowed_domain(domain: str) -> None:
    """Add a domain to the runtime allowlist."""
    _RUNTIME_ALLOWED.add(domain.lower().strip())
    logger.info("🛡️ Domain added to allowlist: %s", domain)


def _named_in_objective(candidates) -> bool:
    """True if a candidate path appears in the objective as a whole path token
    (not merely as a substring of another word or path)."""
    if not _OBJECTIVE_TEXT:
        return False
    for c in candidates:
        if c and len(c) >= 3 and re.search(
                r"(?<![\w./~-])" + re.escape(c) + r"(?![\w/-])", _OBJECTIVE_TEXT):
            return True
    return False


def upload_roots() -> list[Path]:
    """Folders the agent may upload from on its own initiative."""
    raw = os.getenv("AGENT_UPLOAD_DIRS", "").strip()
    dirs = [Path(d).expanduser() for d in raw.split(os.pathsep) if d.strip()] if raw else [DEFAULT_UPLOAD_DIR]
    return [d.resolve() for d in dirs]


def is_upload_allowed(file_path: str) -> tuple[bool, str]:
    """Decide whether the agent may hand `file_path` to a page's file input.

    The model picks the path, and page content can steer the model, so an
    unchecked upload would let any site exfiltrate local files (~/.ssh, .env…).
    Allowed only if the file exists and either:
      • the user's objective names it verbatim (user-approved), or
      • it lives inside an upload root and no path component is hidden (dot-dir/file).
    Returns (allowed, resolved_path_or_reason).
    """
    raw = (file_path or "").strip()
    if not raw:
        return False, "no file path given"
    p = Path(raw).expanduser()
    try:
        real = p.resolve(strict=True)  # follows symlinks — the check is on the real target
    except (OSError, RuntimeError):
        return False, f"file not found: {raw[:80]}"
    if not real.is_file():
        return False, f"not a regular file: {raw[:80]}"

    if _named_in_objective((raw, str(p), str(real))):
        return True, str(real)

    for root in upload_roots():
        if real.is_relative_to(root):
            if any(part.startswith(".") for part in real.relative_to(root).parts):
                return False, "hidden files are never uploaded automatically"
            return True, str(real)

    logger.warning("🛡️ BLOCKED upload outside upload folders: %s", raw[:80])
    return False, ("file is not in an upload folder (AGENT_UPLOAD_DIRS, default ./uploads) "
                   "and is not named in the task")


def _extract_critical_actions(objective: str) -> list[str]:
    """Extract action-intent keywords from the objective."""
    obj_lower = objective.lower()
    keywords = []
    if "add to cart" in obj_lower or "add to bag" in obj_lower:
        keywords.extend(["add to cart", "add to bag", "buy now"])
    elif "post" in obj_lower and "reddit" in obj_lower:
        keywords.extend(["post", "submit"])
    return keywords

def _has_performed_critical_action(working_mem, keywords: list[str]) -> bool:
    """Scan action history for a click or type action matching critical keywords."""
    for step in working_mem.episodic:
        if isinstance(step, str):
            step_desc = step.lower()
            if any(kw in step_desc for kw in keywords):
                return True
        elif isinstance(step, dict):
            action = step.get("action", {})
            if isinstance(action, str):
                # action is a plain string like "click on Create Post"
                if any(kw in action.lower() for kw in keywords):
                    return True
            elif isinstance(action, dict):
                if action.get("type") in ("click", "type"):
                    action_desc = str(action).lower()
                    if any(kw in action_desc for kw in keywords):
                        return True
            # Also check other top-level dict keys for keyword matches
            step_text = str(step).lower()
            if any(kw in step_text for kw in keywords):
                return True
    return False


# CoVe Pre-Done Check
async def cove_pre_done_check(page, plan, working_mem, objective: str = "") -> tuple[bool, str]:
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

    # Check 2: Progress threshold
    if plan.progress_pct < 40:
        reasons_to_block.append(
            f"Only {plan.progress_pct}% progress — too early to declare done"
        )

    # Check 3: Page state
    try:
        url = page.url
        if any(kw in url.lower() for kw in ["login", "signin", "auth", "signup"]):
            reasons_to_block.append(
                f"Still on auth page ({url[:50]})"
            )
    except Exception:
        pass

    # Check 4: Action-trail verification
    if objective:
        critical_keywords = _extract_critical_actions(objective)
        if critical_keywords and not _has_performed_critical_action(working_mem, critical_keywords):
            reasons_to_block.append(
                f"No critical action matching {critical_keywords} found in action history"
            )

    if reasons_to_block:
        return False, "; ".join(reasons_to_block)
    return True, f"All checks passed (progress={plan.progress_pct}%)"
