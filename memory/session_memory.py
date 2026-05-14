"""SESSION MEMORY V12 — SQLite-Backed Contextual Retrieval.

V11 Problem: The EventLog.to_context() dumps the last 15 raw events
(~2000-4000 tokens) into every CEO replan prompt. After 8 steps of
clicking the same button, the CEO gets 8 identical events — zero
useful signal, maximum token waste.

V12 Solution: Extract atomic facts from each action result and store
them in SQLite. The CEO retrieves ONLY facts relevant to the current
URL/domain, compressed to ~200-400 tokens maximum.

Inspired by Mem0's extract-consolidate-retrieve cycle.

Schema:
  facts     — Atomic observations (action, target, outcome, lesson)
  sessions  — Session metadata (objective, start time, status)

Usage:
    memory = SessionMemory("persistence/memory.db")
    memory.start_session("Post a paragraph on Reddit r/test")

    # After each action:
    memory.record(
        url="https://reddit.com/r/test/submit",
        action="click",
        target="Post button",
        outcome="Page did not change",
        learned="Post button may need text content first",
    )

    # When building CEO prompt:
    context = memory.get_context_for_llm("https://reddit.com/r/test/submit")
    # Returns: "- click on 'Post button' → Page did not change (Post button may need text content first)"
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("memory.session_memory")


class SessionMemory:
    """SQLite-backed session memory for compact CEO context.

    Instead of passing raw event logs to the LLM, we:
    1. Extract atomic facts from each action result
    2. Store them with URL + element context
    3. Retrieve only relevant facts for the current page
    4. Format as ultra-compact text for the CEO prompt

    Thread-safe via SQLite's built-in locking.
    """

    def __init__(self, db_path: str | Path = "persistence/memory.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._session_id: str = ""
        self._objective: str = ""

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                objective TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT DEFAULT 'active',
                total_actions INTEGER DEFAULT 0,
                total_progress INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                url TEXT NOT NULL,
                domain TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT '',
                learned TEXT NOT NULL DEFAULT '',
                is_progress INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS page_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                url TEXT NOT NULL,
                domain TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                a11y_hash INTEGER DEFAULT 0,
                element_count INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);
            CREATE INDEX IF NOT EXISTS idx_facts_domain ON facts(domain);
            CREATE INDEX IF NOT EXISTS idx_facts_url ON facts(url);
            CREATE INDEX IF NOT EXISTS idx_visits_session ON page_visits(session_id);
        """)
        self._conn.commit()

    # ── Session Lifecycle ─────────────────────────────────────────────────

    def start_session(self, objective: str) -> str:
        """Start a new session. Returns session ID."""
        self._session_id = str(uuid.uuid4())[:8]
        self._objective = objective
        self._conn.execute(
            "INSERT INTO sessions (id, objective, started_at) VALUES (?, ?, ?)",
            (self._session_id, objective[:500], datetime.now().isoformat()),
        )
        self._conn.commit()
        logger.info("Session started: %s — %s", self._session_id, objective[:80])
        return self._session_id

    def end_session(self, status: str = "completed") -> None:
        """End the current session."""
        if not self._session_id:
            return
        self._conn.execute(
            "UPDATE sessions SET ended_at = ?, status = ? WHERE id = ?",
            (datetime.now().isoformat(), status, self._session_id),
        )
        self._conn.commit()
        logger.info("Session ended: %s (%s)", self._session_id, status)

    @property
    def session_id(self) -> str:
        return self._session_id

    # ── Recording Facts ───────────────────────────────────────────────────

    def record(
        self,
        url: str,
        action: str,
        target: str = "",
        outcome: str = "",
        learned: str = "",
        is_progress: bool = False,
    ) -> None:
        """Record an atomic fact about what happened.

        Args:
            url: Current page URL
            action: Action type (click, type, goto, scroll, etc.)
            target: Element or target description (e.g., "Post button")
            outcome: What happened (e.g., "Compose modal opened")
            learned: Lesson for future reference (e.g., "Click title first")
            is_progress: Whether this action made real progress
        """
        if not self._session_id:
            return

        domain = _extract_domain(url)

        self._conn.execute(
            """INSERT INTO facts
               (session_id, timestamp, url, domain, action, target, outcome, learned, is_progress)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._session_id,
                datetime.now().isoformat(),
                url[:500],
                domain,
                action[:50],
                target[:100],
                outcome[:200],
                learned[:200],
                1 if is_progress else 0,
            ),
        )

        # Update session counters
        progress_inc = 1 if is_progress else 0
        self._conn.execute(
            """UPDATE sessions
               SET total_actions = total_actions + 1,
                   total_progress = total_progress + ?
               WHERE id = ?""",
            (progress_inc, self._session_id),
        )
        self._conn.commit()

    def record_visit(
        self,
        url: str,
        a11y_hash: int = 0,
        element_count: int = 0,
    ) -> None:
        """Record a page visit for navigation tracking."""
        if not self._session_id:
            return

        domain = _extract_domain(url)
        self._conn.execute(
            """INSERT INTO page_visits
               (session_id, url, domain, timestamp, a11y_hash, element_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (self._session_id, url[:500], domain, datetime.now().isoformat(), a11y_hash, element_count),
        )
        self._conn.commit()

    # ── Retrieval ─────────────────────────────────────────────────────────

    def recall_for_url(self, url: str, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve facts relevant to a URL (exact URL match first, then domain)."""
        if not self._session_id:
            return []

        domain = _extract_domain(url)

        # Priority 1: Exact URL match from current session
        rows = self._conn.execute(
            """SELECT action, target, outcome, learned, is_progress
               FROM facts
               WHERE session_id = ? AND url = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (self._session_id, url[:500], limit),
        ).fetchall()

        # Priority 2: Domain match if not enough exact matches
        if len(rows) < limit:
            remaining = limit - len(rows)
            domain_rows = self._conn.execute(
                """SELECT action, target, outcome, learned, is_progress
                   FROM facts
                   WHERE session_id = ? AND domain = ? AND url != ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (self._session_id, domain, url[:500], remaining),
            ).fetchall()
            rows.extend(domain_rows)

        return [
            {
                "action": r[0],
                "target": r[1],
                "outcome": r[2],
                "learned": r[3],
                "is_progress": bool(r[4]),
            }
            for r in rows
        ]

    def recall_failures(self, url: str, limit: int = 3) -> list[dict[str, Any]]:
        """Retrieve ONLY failed actions for a URL — most useful for replanning."""
        if not self._session_id:
            return []

        domain = _extract_domain(url)
        rows = self._conn.execute(
            """SELECT action, target, outcome, learned
               FROM facts
               WHERE session_id = ? AND domain = ? AND is_progress = 0
               ORDER BY timestamp DESC LIMIT ?""",
            (self._session_id, domain, limit),
        ).fetchall()

        return [
            {"action": r[0], "target": r[1], "outcome": r[2], "learned": r[3]}
            for r in rows
        ]

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary stats for the current session."""
        if not self._session_id:
            return {}

        row = self._conn.execute(
            "SELECT objective, total_actions, total_progress, status FROM sessions WHERE id = ?",
            (self._session_id,),
        ).fetchone()

        if not row:
            return {}

        return {
            "session_id": self._session_id,
            "objective": row[0],
            "total_actions": row[1],
            "total_progress": row[2],
            "status": row[3],
            "progress_rate": row[2] / row[1] if row[1] > 0 else 0.0,
        }

    # ── LLM Context Generation ────────────────────────────────────────────

    def get_context_for_llm(self, current_url: str, max_chars: int = 800) -> str:
        """Build a compact memory context string for the CEO prompt.

        This replaces the old EventLog.to_context() which dumped 15 raw events.

        Returns something like:
            Memory for reddit.com:
            - click on 'Post button' → no effect (try filling text first)
            - type in 'Title field' → title inserted ✓
            - goto reddit.com/r/test/submit → navigated ✓

        Typically 200-400 chars vs the old 2000-4000 chars.
        """
        facts = self.recall_for_url(current_url, limit=6)

        if not facts:
            return "(no prior experience with this page)"

        domain = _extract_domain(current_url)
        lines: list[str] = [f"Memory for {domain}:"]
        total_chars = len(lines[0])

        for f in facts:
            icon = "✓" if f["is_progress"] else "✗"
            line = f"- {f['action']} on '{f['target']}' → {f['outcome']} {icon}"
            if f.get("learned"):
                line += f" ({f['learned']})"

            if total_chars + len(line) > max_chars:
                lines.append("... (earlier facts omitted)")
                break

            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    def get_failure_context_for_replan(self, current_url: str, max_chars: int = 500) -> str:
        """Build a failure-focused context for CEO replanning.

        Only includes failed actions — maximum signal for strategy change.
        """
        failures = self.recall_failures(current_url, limit=4)

        if not failures:
            return "(no prior failures on this page)"

        lines: list[str] = ["Failed approaches:"]
        for f in failures:
            line = f"- {f['action']} on '{f['target']}' → {f['outcome']}"
            if f.get("learned"):
                line += f" | hint: {f['learned']}"
            lines.append(line)

        result = "\n".join(lines)
        return result[:max_chars]

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_domain(url: str) -> str:
    """Extract the domain from a URL for grouping."""
    try:
        parsed = urlparse(url)
        return parsed.netloc or "unknown"
    except Exception:
        return "unknown"
