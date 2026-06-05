"""Platform detection via proper URL parsing (not substring matching).

Fixes V-06: 'twittercomics.com' no longer matches 'twitter.com'.
"""

from urllib.parse import urlparse

KNOWN_PLATFORMS = {
    "dev.to", "medium.com", "x.com", "twitter.com",
    "hashnode.com", "github.com", "news.ycombinator.com",
    "reddit.com", "linkedin.com", "issuetracker.google.com",
}

CHAR_LIMITS = {
    "x.com": 280, "twitter.com": 280,
    "reddit.com": 40000, "dev.to": 100000,
    "medium.com": 100000, "hashnode.com": 100000,
}


def detect_platform(url: str) -> str | None:
    """Detect platform from URL using proper host matching."""
    try:
        host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return None
    # Check exact match
    if host in KNOWN_PLATFORMS:
        return host
    # Handle subdomains (e.g., old.reddit.com)
    parts = host.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in KNOWN_PLATFORMS:
            return candidate
    return None


def get_char_limit(url: str) -> int:
    """Get platform-specific character limit."""
    platform = detect_platform(url)
    return CHAR_LIMITS.get(platform, 100000)
