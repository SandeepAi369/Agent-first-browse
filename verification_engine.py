"""VerificationEngine V13 — Multi-signal key-node success detection.

Replaces the blunt hash(url + json[:500]) check with a weighted
multi-signal verdict inspired by Mind2Web-Live key-node evaluation
and Skyvern's Validator pattern.

Fixes V-07 (stale truncated hash), V-11 (CriticV12 unused).

Signals (ranked by reliability):
  1. Key-node predicate (platform-specific success indicator)
  2. URL change + URL matches expected pattern
  3. AX snapshot fingerprint delta
  4. Confirmation detection (toast/alert/banner)
  5. Target element disappeared
"""

from __future__ import annotations
import hashlib
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("verification")

# ── Success URL patterns per platform ──
SUCCESS_PATTERNS = {
    "x.com":        re.compile(r"x\.com/\w+/status/\d+"),
    "twitter.com":  re.compile(r"twitter\.com/\w+/status/\d+"),
    "reddit.com":   re.compile(r"reddit\.com/r/\w+/comments/"),
    "dev.to":       re.compile(r"dev\.to/\w+/[\w-]+"),
    "medium.com":   re.compile(r"medium\.com/.*[a-f0-9]{8,}"),
}

# ── Confirmation keywords (toast/banner detection) ──
CONFIRMATION_KEYWORDS = [
    "posted", "published", "submitted", "created", "success",
    "your post", "your article", "saved", "sent",
]


@dataclass
class PageSnapshot:
    """Minimal snapshot for before/after comparison."""
    url: str
    ax_fingerprint: str  # hash of normalized AX tree
    element_count: int
    text_preview: str  # first 500 chars of visible text


@dataclass
class Verdict:
    """The verification engine's judgment."""
    success: bool
    confidence: float
    signals: dict[str, bool]
    reason: str


def fingerprint_ax_tree(elements: list[dict]) -> str:
    """Create a deterministic fingerprint of the AX tree structure."""
    parts = []
    for el in sorted(elements, key=lambda e: (e.get("x", 0), e.get("y", 0))):
        parts.append(f"{el.get('kind', '')}/{el.get('text', '')[:30]}")
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


def take_snapshot(url: str, elements: list[dict], text: str = "") -> PageSnapshot:
    """Capture a page snapshot for before/after comparison."""
    return PageSnapshot(
        url=url,
        ax_fingerprint=fingerprint_ax_tree(elements),
        element_count=len(elements),
        text_preview=text[:500],
    )


def detect_confirmation(after: PageSnapshot) -> bool:
    """Check if the page shows a success/confirmation signal."""
    text_lower = after.text_preview.lower()
    return any(kw in text_lower for kw in CONFIRMATION_KEYWORDS)


def verify_outcome(
    before: PageSnapshot,
    after: PageSnapshot,
    platform: str | None = None,
) -> Verdict:
    """Multi-signal verification with weighted voting.

    Combines URL change, success URL pattern, AX fingerprint delta,
    confirmation detection, and element count change.
    """
    signals = {
        "url_changed": before.url != after.url,
        "url_matches_success": False,
        "snapshot_delta": before.ax_fingerprint != after.ax_fingerprint,
        "confirmation_detected": detect_confirmation(after),
        "element_count_changed": abs(after.element_count - before.element_count) >= 3,
    }

    # Check platform-specific success URL
    if platform and platform in SUCCESS_PATTERNS:
        signals["url_matches_success"] = bool(
            SUCCESS_PATTERNS[platform].search(after.url)
        )

    # Weighted vote
    weights = {
        "url_matches_success": 3.0,
        "confirmation_detected": 2.5,
        "url_changed": 2.0,
        "snapshot_delta": 1.5,
        "element_count_changed": 1.0,
    }

    score = sum(weights[k] for k, v in signals.items() if v)
    max_score = sum(weights.values())
    confidence = score / max_score

    THRESHOLD = 0.35  # At least url_changed + one more signal
    success = confidence >= THRESHOLD

    reason_parts = [k for k, v in signals.items() if v]
    icon = "\u2705" if success else "\u274c"
    reason = f"{icon} Signals: {', '.join(reason_parts) or 'none'} ({confidence:.0%})"

    logger.info(reason)
    return Verdict(success=success, confidence=confidence, signals=signals, reason=reason)
