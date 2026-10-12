"""
content_store.py — Intelligent Article Content Store
═════════════════════════════════════════════════════
Separates long-form content from the LLM objective to prevent
hallucination, truncation, and infinite typing loops.

Architecture:
  1. DETECT — Scans the objective for embedded article content
  2. EXTRACT — Splits into title, body, tags, and target URL
  3. SHORTEN — Returns a compact objective for the LLM (< 200 chars)
  4. SERVE — Provides full content to the executor on demand
  5. CLEANUP — Auto-purges after successful post or on next launch

The store persists to disk (JSON) so it survives crashes. On each
new run, stale entries older than 24 hours are automatically purged.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("content_store")

# ─────────────────────────────────────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArticlePayload:
    """Structured article content extracted from a user objective."""
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    target_url: str = ""
    target_platform: str = ""
    created_at: float = 0.0
    posted: bool = False

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    @property
    def total_chars(self) -> int:
        return len(self.title) + len(self.body)

    @property
    def body_preview(self) -> str:
        return self.body[:120].replace("\n", " ") + ("..." if len(self.body) > 120 else "")

    def short_objective(self) -> str:
        """Generate a compact LLM-friendly objective (no article body)."""
        parts = []
        if self.target_url:
            parts.append(f"Go to {self.target_url}")
        elif self.target_platform:
            parts.append(f"Go to {self.target_platform}")
        parts.append("and publish a new post")
        parts.append(f"with title: \"{self.title}\"")
        parts.append(f"The full article content ({len(self.body)} chars) is pre-loaded in the content store.")
        parts.append("Steps: 1) Navigate to the editor")
        parts.append("2) You MUST click the TITLE field to trigger title injection (even if it looks filled)")
        parts.append("3) You MUST click the BODY/CONTENT field to trigger article auto-injection (even if it looks filled with a draft)")
        parts.append("4) Click Publish/Submit")
        return ". ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Detection Engine — Identifies articles in raw objectives
# ─────────────────────────────────────────────────────────────────────────────

# Platform patterns
_PLATFORM_PATTERNS = {
    "dev.to":       re.compile(r"dev\.to", re.IGNORECASE),
    "medium.com":   re.compile(r"medium\.com", re.IGNORECASE),
    "hashnode.com": re.compile(r"hashnode\.com", re.IGNORECASE),
    "x.com":        re.compile(r"(?:x\.com|twitter\.com)", re.IGNORECASE),
    "reddit.com":   re.compile(r"reddit\.com", re.IGNORECASE),
}

# URL extraction
_URL_PATTERN = re.compile(r"https?://[^\s,)\"']+", re.IGNORECASE)

# Title detection heuristics
_TITLE_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:Title|Heading|Subject)\s*[:\-—]\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\*{1,2}(.{10,120}?)\*{1,2}\s*(?:\n|$)"),  # **Bold title**
    re.compile(r"(?:^|\n)\s*#{1,3}\s+(.{10,120})\s*(?:\n|$)"),         # # Markdown heading
]

# Content threshold — objectives shorter than this are NOT articles
_ARTICLE_THRESHOLD = 400


def detect_article(objective: str) -> Optional[ArticlePayload]:
    """Analyze an objective string and extract article content if present.
    
    Returns None if the objective is a simple task (tweet, click, navigate).
    Returns an ArticlePayload if it contains long-form content.
    """
    # ── Quick reject: too short to be an article ──
    stripped = objective.strip()
    if len(stripped) < _ARTICLE_THRESHOLD:
        return None

    # ── Detect target platform/URL ──
    target_url = ""
    target_platform = ""

    urls = _URL_PATTERN.findall(stripped)
    for url in urls:
        for platform, pattern in _PLATFORM_PATTERNS.items():
            if pattern.search(url):
                target_url = url
                target_platform = platform
                break
        if target_url:
            break

    # If no URL found, check for platform mention in text
    if not target_platform:
        for platform, pattern in _PLATFORM_PATTERNS.items():
            if pattern.search(stripped):
                target_platform = platform
                break

    # ── Extract title ──
    title = ""
    body_start_idx = 0

    # Try explicit title patterns first
    for pat in _TITLE_PATTERNS:
        m = pat.search(stripped)
        if m:
            title = m.group(1).strip().strip("*#").strip()
            body_start_idx = m.end()
            break

    # Fallback: first line that looks like a title (10-150 chars, no periods)
    if not title:
        lines = stripped.split("\n")
        for i, line in enumerate(lines):
            clean = line.strip().strip("*#").strip()
            # Skip lines that are URLs, instructions, or too short
            if len(clean) < 10 or len(clean) > 150:
                continue
            if clean.startswith("http") or clean.startswith("Go to"):
                continue
            if "." not in clean or clean.count(".") <= 1:
                title = clean
                body_start_idx = stripped.index(line) + len(line)
                break

    if not title:
        # Last resort: use first substantial line
        for line in stripped.split("\n"):
            clean = line.strip()
            if len(clean) > 15 and not clean.startswith("http"):
                title = clean[:120]
                body_start_idx = stripped.index(line) + len(line)
                break

    # ── Extract body ──
    # Remove the instruction prefix (everything before the actual article content)
    body_raw = stripped[body_start_idx:].strip()

    # Remove trailing instruction lines (like "Dev.to give this target...")
    trailing_instruction_patterns = [
        re.compile(r"\n.*?(?:give this|target to|don't any|just go|publish in).*$", re.IGNORECASE | re.DOTALL),
        re.compile(r"\n.*?(?:dev\.to|medium|hashnode)\s+(?:give|target|publish).*$", re.IGNORECASE | re.DOTALL),
    ]
    for pat in trailing_instruction_patterns:
        body_raw = pat.sub("", body_raw).strip()

    # Remove GitHub link line if it's the very last thing (we'll keep it in the body)
    # Actually keep it — it's part of the article

    if len(body_raw) < 100:
        # Body too short after extraction — not actually an article
        return None

    # ── Extract tags ──
    tags = []
    tag_match = re.search(r"(?:Tags?|Categories)\s*[:\-]\s*(.+?)(?:\n|$)", stripped, re.IGNORECASE)
    if tag_match:
        tags = [t.strip().strip("#") for t in re.split(r"[,\s]+", tag_match.group(1)) if t.strip()]

    # Also extract hashtags from the body
    hashtags = re.findall(r"#(\w{3,30})", body_raw)
    tags.extend(h for h in hashtags if h not in tags)

    payload = ArticlePayload(
        title=title,
        body=body_raw,
        tags=tags[:5],  # Cap at 5 tags
        target_url=target_url,
        target_platform=target_platform,
    )

    log.info(
        "Article detected: title='%s' body=%d chars platform=%s tags=%s",
        title[:60], len(body_raw), target_platform or "?", tags[:3],
    )
    return payload


# ─────────────────────────────────────────────────────────────────────────────
#  Persistent Store — JSON-backed with auto-cleanup
# ─────────────────────────────────────────────────────────────────────────────

class ContentStore:
    """Manages article content persistence with automatic lifecycle cleanup."""

    _STORE_FILE = "content_store.json"
    _STALE_HOURS = 24  # Auto-purge entries older than this

    def __init__(self, base_dir: str | Path = "."):
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / self._STORE_FILE
        self._current: Optional[ArticlePayload] = None
        self._auto_purge_stale()

    # ── Public API ──

    def load_from_objective(self, objective: str) -> tuple[str, bool]:
        """Detect article in objective. Returns (possibly_shortened_objective, has_article).
        
        If an article is detected, stores it and returns a short objective.
        If no article, returns the original objective unchanged.
        """
        payload = detect_article(objective)
        if not payload:
            return objective, False

        self._current = payload
        self._persist()
        short = payload.short_objective()
        log.info(
            "Content store loaded: %d chars → short objective (%d chars)",
            payload.total_chars, len(short),
        )
        return short, True

    @property
    def has_article(self) -> bool:
        return self._current is not None

    @property
    def article(self) -> Optional[ArticlePayload]:
        return self._current

    def get_title(self) -> str:
        return self._current.title if self._current else ""

    def get_body(self) -> str:
        return self._current.body if self._current else ""

    def get_tags(self) -> list[str]:
        return self._current.tags if self._current else []

    def mark_posted(self) -> None:
        """Mark the current article as posted and clean up."""
        if self._current:
            self._current.posted = True
            log.info("Content store: article marked as posted, cleaning up")
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove the store file after successful post."""
        try:
            if self._path.exists():
                self._path.unlink()
            self._current = None
            log.info("Content store: cleaned up successfully")
        except Exception as e:
            log.warning("Content store cleanup failed: %s", e)

    # ── Persistence ──

    def _persist(self) -> None:
        """Save current payload to disk (atomic: temp → fsync → os.replace)."""
        if not self._current:
            return
        try:
            import tempfile, os
            data = asdict(self._current)
            content = json.dumps(data, indent=2, ensure_ascii=False)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, str(self._path))
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            log.warning("Content store persist failed: %s", e)

    def _load_from_disk(self) -> Optional[ArticlePayload]:
        """Load a previously saved payload from disk."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return ArticlePayload(**data)
        except Exception as e:
            log.warning("Content store load failed: %s", e)
            return None

    def _auto_purge_stale(self) -> None:
        """Remove store entries older than _STALE_HOURS."""
        existing = self._load_from_disk()
        if not existing:
            return

        age_hours = (time.time() - existing.created_at) / 3600
        if age_hours > self._STALE_HOURS:
            log.info(
                "Content store: purging stale entry (%.1f hours old): '%s'",
                age_hours, existing.title[:50],
            )
            self._cleanup()
        elif existing.posted:
            log.info("Content store: purging already-posted entry: '%s'", existing.title[:50])
            self._cleanup()
        else:
            # Restore the in-progress article
            self._current = existing
            log.info(
                "Content store: restored in-progress article: '%s' (%d chars)",
                existing.title[:50], existing.total_chars,
            )
