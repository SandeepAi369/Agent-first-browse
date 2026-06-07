"""Episodic Task Memory — Persistent record of all task execution actions.

Prevents duplicate work by recording every published post and providing
deduplication checks before new tasks execute.

Uses the existing SQLite schema from database.py (task_history table)
but is a standalone module that advanced_agent.py can import directly.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.logger import get_logger

logger = get_logger("campaign_memory")

# Use the same persistence directory as advanced_agent.py
PERSISTENCE_ROOT = Path(__file__).parent / "persistence"
DB_PATH = PERSISTENCE_ROOT / "agent_persistence.db"


def _ensure_db() -> sqlite3.Connection:
    """Open (and initialize if needed) the task memory database."""
    PERSISTENCE_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.execute("PRAGMA busy_timeout=5000;")
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        conn.execute("PRAGMA journal_mode=DELETE;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # Create the posts table (idempotent)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS published_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            url TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL,
            content_preview TEXT NOT NULL DEFAULT '',
            published_at TEXT NOT NULL,
            agent_model TEXT NOT NULL DEFAULT '',
            steps_taken INTEGER NOT NULL DEFAULT 0,
            UNIQUE(platform, content_hash)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_published_posts_platform
        ON published_posts(platform)
    """)

    # Timing tracking table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS platform_cooldowns (
            platform TEXT PRIMARY KEY,
            last_post_at TEXT NOT NULL,
            cooldown_hours INTEGER NOT NULL DEFAULT 168
        )
    """)
    conn.commit()
    return conn


class CampaignMemory:
    """Lightweight episodic task execution history tracker.

    Auto-reconnects on database errors — no single failure kills the memory.
    """

    def __init__(self, db_path: Path | None = None):
        if db_path:
            global DB_PATH
            DB_PATH = db_path
        self._conn: sqlite3.Connection | None = None
        self._connect()

    def _connect(self) -> None:
        """Establish (or re-establish) the database connection."""
        try:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = _ensure_db()
        except Exception as e:
            logger.error("Task memory DB connect failed: %s", e)
            self._conn = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get a live connection, auto-reconnecting if needed."""
        if self._conn is None:
            self._connect()
        if self._conn is not None:
            try:
                self._conn.execute("SELECT 1")
                return self._conn
            except (sqlite3.OperationalError, sqlite3.ProgrammingError):
                logger.warning("Task memory connection dead, reconnecting...")
                self._connect()
        if self._conn is None:
            raise sqlite3.OperationalError("Cannot connect to task memory DB")
        return self._conn

    def record_post(
        self,
        platform: str,
        title: str,
        content: str,
        url: str = "",
        agent_model: str = "",
        steps_taken: int = 0,
    ) -> bool:
        """Record a completed task action. Returns True if recorded, False if duplicate."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()

        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO published_posts
                   (platform, url, title, content_hash, content_preview, published_at, agent_model, steps_taken)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    platform.lower().strip(),
                    url,
                    title,
                    content_hash,
                    content[:200],
                    now,
                    agent_model,
                    steps_taken,
                ),
            )
            # Update cooldown
            conn.execute(
                """INSERT OR REPLACE INTO platform_cooldowns (platform, last_post_at, cooldown_hours)
                   VALUES (?, ?, 168)""",
                (platform.lower().strip(), now),
            )
            conn.commit()
            logger.info(
                "📝 Recorded post on %s: '%s' (hash=%s)",
                platform, title[:60], content_hash,
            )
            return True
        except sqlite3.IntegrityError:
            logger.warning(
                "⚠️  Duplicate post detected on %s with hash %s — skipping record",
                platform, content_hash,
            )
            return False

    def has_posted_on(self, platform: str) -> bool:
        """Check if we have ever posted on this platform."""
        cursor = self._get_conn().execute(
            "SELECT COUNT(*) FROM published_posts WHERE platform = ?",
            (platform.lower().strip(),),
        )
        count = cursor.fetchone()[0]
        return count > 0

    def is_on_cooldown(self, platform: str, cooldown_hours: int = 168) -> tuple[bool, str]:
        """Cooldown check stub — always returns not on cooldown.

        Kept for backwards compatibility.
        Returns: (is_cooling_down, reason_message)
        """
        return False, ""

    def get_campaign_log(self, limit: int = 20) -> list[dict[str, Any]]:  # name kept for compat
        """Return the most recent task actions for context injection."""
        cursor = self._get_conn().execute(
            """SELECT platform, url, title, content_preview, published_at, agent_model, steps_taken
               FROM published_posts
               ORDER BY published_at DESC
               LIMIT ?""",
            (limit,),
        )
        columns = ["platform", "url", "title", "content_preview", "published_at", "agent_model", "steps_taken"]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_post_count(self, platform: str | None = None) -> int:
        """Get total number of posts, optionally filtered by platform."""
        if platform:
            cursor = self._get_conn().execute(
                "SELECT COUNT(*) FROM published_posts WHERE platform = ?",
                (platform.lower().strip(),),
            )
        else:
            cursor = self._get_conn().execute("SELECT COUNT(*) FROM published_posts")
        return cursor.fetchone()[0]

    def get_summary(self) -> str:
        """Return a human-readable summary of the task history."""
        total = self.get_post_count()
        if total == 0:
            return "No posts published yet."

        cursor = self._get_conn().execute(
            "SELECT DISTINCT platform FROM published_posts ORDER BY platform"
        )
        platforms = [row[0] for row in cursor.fetchall()]

        lines = [f"Task History: {total} post(s) across {len(platforms)} platform(s)"]
        for p in platforms:
            count = self.get_post_count(p)
            lines.append(f"  • {p}: {count} post(s)")

        return "\n".join(lines)

    def close(self):
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass
