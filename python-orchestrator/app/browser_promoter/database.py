from __future__ import annotations

import sqlite3
from pathlib import Path


def get_persistence_dir() -> Path:
    """Return persistence directory path, creating it when missing."""
    workspace_root = Path(__file__).resolve().parents[3]
    persistence_dir = workspace_root / "persistence"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    return persistence_dir


def get_agent_database_path() -> Path:
    """Return local sqlite database path for campaign and vector memory data."""
    return get_persistence_dir() / "agent_persistence.db"


def initialize_persistence_database(database_path: Path | None = None) -> Path:
    """
    Initialize sqlite persistence schema for campaign history and vector memory.

    Returns the resolved sqlite file path.
    """
    db_path = database_path or get_agent_database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA foreign_keys=ON;")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS campaign_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                target_url TEXT NOT NULL,
                action_type TEXT NOT NULL,
                outcome_score REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Standard sqlite fallback schema for vector memory records.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                target_url TEXT NOT NULL,
                action_type TEXT NOT NULL,
                content_text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                success_score REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_campaign_history_thread_platform
            ON campaign_history(thread_id, platform)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_vector_memory_thread_platform
            ON vector_memory(thread_id, platform)
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS target_communities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                url TEXT NOT NULL,
                niche TEXT NOT NULL,
                locked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(platform, url)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_target_communities_platform_niche
            ON target_communities(platform, niche)
            """
        )

        connection.commit()

    return db_path


def lock_target_community(
    *,
    platform: str,
    url: str,
    niche: str,
    database_path: Path | None = None,
) -> int:
    """Upsert a locked community target and return the resulting row id when available."""
    db_path = initialize_persistence_database(database_path)

    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO target_communities(platform, url, niche)
            VALUES(?, ?, ?)
            ON CONFLICT(platform, url)
            DO UPDATE SET
                niche=excluded.niche,
                locked_at=CURRENT_TIMESTAMP
            """,
            (platform.strip(), url.strip(), niche.strip() or "general"),
        )
        connection.commit()
        return int(cursor.lastrowid or 0)


def count_locked_target_communities(database_path: Path | None = None) -> int:
    """Return total number of locked target communities."""
    db_path = initialize_persistence_database(database_path)

    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute("SELECT COUNT(*) FROM target_communities")
        row = cursor.fetchone()
        return int(row[0] if row else 0)
