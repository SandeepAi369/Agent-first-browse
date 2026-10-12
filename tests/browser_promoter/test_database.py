"""Tests for SQLite persistence layer."""

from __future__ import annotations

from pathlib import Path

from app.browser_promoter.database import (
    count_locked_target_communities,
    initialize_persistence_database,
    lock_target_community,
)


class TestInitializePersistenceDatabase:
    """Validate schema creation and idempotency."""

    def test_creates_database_file(self, tmp_db_path: Path) -> None:
        result_path = initialize_persistence_database(tmp_db_path)
        assert result_path == tmp_db_path
        assert tmp_db_path.exists()

    def test_idempotent_initialization(self, tmp_db_path: Path) -> None:
        """Calling initialize twice should not raise or corrupt data."""
        initialize_persistence_database(tmp_db_path)
        initialize_persistence_database(tmp_db_path)
        assert tmp_db_path.exists()

    def test_creates_required_tables(self, tmp_db_path: Path) -> None:
        import sqlite3

        initialize_persistence_database(tmp_db_path)
        with sqlite3.connect(tmp_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row[0] for row in cursor.fetchall()}

        assert "campaign_history" in tables
        assert "vector_memory" in tables
        assert "target_communities" in tables


class TestLockTargetCommunity:
    """Validate community locking insert and upsert behavior."""

    def test_insert_new_community(self, tmp_db_path: Path) -> None:
        initialize_persistence_database(tmp_db_path)
        row_id = lock_target_community(
            platform="reddit",
            url="https://reddit.com/r/programming",
            niche="programming",
            database_path=tmp_db_path,
        )
        assert row_id > 0

    def test_upsert_existing_community(self, tmp_db_path: Path) -> None:
        initialize_persistence_database(tmp_db_path)
        lock_target_community(
            platform="reddit",
            url="https://reddit.com/r/python",
            niche="python",
            database_path=tmp_db_path,
        )
        # Upsert same platform+url with different niche
        lock_target_community(
            platform="reddit",
            url="https://reddit.com/r/python",
            niche="python-advanced",
            database_path=tmp_db_path,
        )

        import sqlite3

        with sqlite3.connect(tmp_db_path) as conn:
            cursor = conn.execute(
                "SELECT niche FROM target_communities WHERE url = ?",
                ("https://reddit.com/r/python",),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "python-advanced"

    def test_multiple_communities(self, tmp_db_path: Path) -> None:
        initialize_persistence_database(tmp_db_path)
        lock_target_community(
            platform="reddit",
            url="https://reddit.com/r/programming",
            niche="programming",
            database_path=tmp_db_path,
        )
        lock_target_community(
            platform="github",
            url="https://github.com/topics/ai",
            niche="ai",
            database_path=tmp_db_path,
        )

        count = count_locked_target_communities(database_path=tmp_db_path)
        assert count == 2


class TestCountLockedTargetCommunities:
    """Validate community counting."""

    def test_empty_database(self, tmp_db_path: Path) -> None:
        initialize_persistence_database(tmp_db_path)
        count = count_locked_target_communities(database_path=tmp_db_path)
        assert count == 0

    def test_after_inserts(self, tmp_db_path: Path) -> None:
        initialize_persistence_database(tmp_db_path)
        for i in range(5):
            lock_target_community(
                platform="reddit",
                url=f"https://reddit.com/r/test{i}",
                niche="test",
                database_path=tmp_db_path,
            )
        count = count_locked_target_communities(database_path=tmp_db_path)
        assert count == 5
