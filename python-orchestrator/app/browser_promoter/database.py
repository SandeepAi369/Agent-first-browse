from __future__ import annotations

import functools
import sqlite3
import time
from pathlib import Path
from typing import Any

from app.logger import get_logger

logger = get_logger("database")


def get_persistence_dir() -> Path:
    """Return persistence directory path, creating it when missing."""
    workspace_root = Path(__file__).resolve().parents[3]
    persistence_dir = workspace_root / "persistence"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    return persistence_dir


def get_agent_database_path() -> Path:
    """Return local sqlite database path for campaign and vector memory data."""
    return get_persistence_dir() / "agent_persistence.db"


# ═══════════════════════════════════════════════════════════════════════════════
#  Database Pool — Singleton connection manager with auto-retry
# ═══════════════════════════════════════════════════════════════════════════════

class DatabasePool:
    """Thread-safe singleton database connection manager.

    - Lazy-initializes a single connection per lifetime
    - Auto-retries on 'database is locked' errors (3x with exponential backoff)
    - Schema initialization runs ONCE, not on every query
    - Auto-reconnects if the connection dies
    """

    _instance: DatabasePool | None = None
    _initialized_paths: set[str] = set()

    def __init__(self) -> None:
        self._connections: dict[str, sqlite3.Connection] = {}

    @classmethod
    def get(cls) -> DatabasePool:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_connection(self, db_path: Path | None = None) -> sqlite3.Connection:
        """Get or create a cached connection for the given path."""
        path = db_path or get_agent_database_path()
        key = str(path)

        # Check if existing connection is still alive
        if key in self._connections:
            try:
                self._connections[key].execute("SELECT 1")
                return self._connections[key]
            except (sqlite3.OperationalError, sqlite3.ProgrammingError):
                logger.warning("DB connection dead, reconnecting: %s", key)
                try:
                    self._connections[key].close()
                except Exception:
                    pass
                del self._connections[key]

        # Create new connection
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=10.0)
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            conn.execute("PRAGMA journal_mode=DELETE;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        self._connections[key] = conn

        # Initialize schema once per path
        if key not in self._initialized_paths:
            _create_all_tables(conn)
            conn.commit()
            self._initialized_paths.add(key)
            logger.info("Database initialized: %s", key)

        return conn


def retry_on_locked(func):
    """Decorator: retry up to 3x on 'database is locked' with exponential backoff."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_err = None
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    last_err = e
                    wait = 0.5 * (2 ** attempt)
                    logger.warning(
                        "DB locked (attempt %d/3), retrying in %.1fs: %s",
                        attempt + 1, wait, e,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise last_err  # type: ignore[misc]
    return wrapper


def _create_all_tables(connection: sqlite3.Connection) -> None:
    """Create ALL persistence tables (called once by DatabasePool)."""
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
    _create_marketing_tables(connection)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS system_access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_name TEXT NOT NULL,
            operation TEXT NOT NULL,
            target_table TEXT NOT NULL,
            query_context TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS completed_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            final_state_json TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def initialize_persistence_database(database_path: Path | None = None) -> Path:
    """Initialize persistence schema. Now routes through DatabasePool for caching.

    Backward-compatible — still returns the resolved db path.
    """
    db_path = database_path or get_agent_database_path()
    DatabasePool.get().get_connection(db_path)
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


def provision_dynamic_table(
    table_name: str,
    schema_json: dict[str, str],
    database_path: Path | None = None,
) -> None:
    """
    Dynamically provision a new SQLite table based on a schema mapping.
    schema_json maps column_name -> sqlite_data_type (e.g., {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT'})
    """
    db_path = initialize_persistence_database(database_path)
    
    # Safe prefixing to prevent accidental core table overwrite
    safe_table_name = table_name if table_name.startswith("dynamic_") else f"dynamic_{table_name}"
    
    columns = [f"{col} {dtype}" for col, dtype in schema_json.items()]
    columns_str = ", ".join(columns)
    
    create_stmt = f"CREATE TABLE IF NOT EXISTS {safe_table_name} ({columns_str})"
    
    with sqlite3.connect(db_path) as conn:
        conn.execute(create_stmt)
        conn.commit()


def log_data_access(
    node_name: str,
    operation: str,
    target_table: str,
    query_context: str,
    database_path: Path | None = None,
) -> None:
    """Log DB access events to system_access_logs."""
    db_path = initialize_persistence_database(database_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO system_access_logs(node_name, operation, target_table, query_context)
            VALUES(?, ?, ?, ?)
            """,
            (node_name, operation, target_table, query_context)
        )
        conn.commit()
# ═══════════════════════════════════════════════════════════════════════════════
#  Marketing Engine Tables & Helpers (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

def _create_marketing_tables(connection: sqlite3.Connection) -> None:
    """Create v3.0 marketing engine tables if they don't exist."""

    # Cached GitHub repo intelligence
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS github_repo_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            repo_name TEXT NOT NULL,
            full_name TEXT NOT NULL,
            url TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT '',
            stars INTEGER NOT NULL DEFAULT 0,
            forks INTEGER NOT NULL DEFAULT 0,
            topics_json TEXT NOT NULL DEFAULT '[]',
            readme_summary TEXT NOT NULL DEFAULT '',
            key_features_json TEXT NOT NULL DEFAULT '[]',
            target_audience TEXT NOT NULL DEFAULT '',
            promotion_hooks_json TEXT NOT NULL DEFAULT '[]',
            homepage TEXT NOT NULL DEFAULT '',
            cached_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, repo_name)
        )
        """
    )

    # Promotion history — tracks what was promoted where (for cooldown)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS promotion_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_name TEXT NOT NULL,
            platform TEXT NOT NULL,
            channel TEXT NOT NULL,
            tactic TEXT NOT NULL,
            content_angle TEXT NOT NULL DEFAULT '',
            draft_message TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'planned',
            promoted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_promotion_history_repo_platform
        ON promotion_history(repo_name, platform, promoted_at)
        """
    )

    # Promotion results — engagement tracking
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS promotion_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            promotion_id INTEGER NOT NULL,
            url TEXT NOT NULL DEFAULT '',
            upvotes INTEGER NOT NULL DEFAULT 0,
            comments INTEGER NOT NULL DEFAULT 0,
            views INTEGER NOT NULL DEFAULT 0,
            checked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(promotion_id) REFERENCES promotion_history(id)
        )
        """
    )


def upsert_repo_cache(
    *,
    username: str,
    repo_name: str,
    full_name: str,
    url: str,
    description: str = "",
    language: str = "",
    stars: int = 0,
    forks: int = 0,
    topics_json: str = "[]",
    readme_summary: str = "",
    key_features_json: str = "[]",
    target_audience: str = "",
    promotion_hooks_json: str = "[]",
    homepage: str = "",
    database_path: Path | None = None,
) -> int:
    """Insert or update a cached repo profile."""
    db_path = initialize_persistence_database(database_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO github_repo_cache(
                username, repo_name, full_name, url, description, language,
                stars, forks, topics_json, readme_summary, key_features_json,
                target_audience, promotion_hooks_json, homepage, cached_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(username, repo_name)
            DO UPDATE SET
                full_name=excluded.full_name, url=excluded.url,
                description=excluded.description, language=excluded.language,
                stars=excluded.stars, forks=excluded.forks,
                topics_json=excluded.topics_json, readme_summary=excluded.readme_summary,
                key_features_json=excluded.key_features_json,
                target_audience=excluded.target_audience,
                promotion_hooks_json=excluded.promotion_hooks_json,
                homepage=excluded.homepage, cached_at=CURRENT_TIMESTAMP
            """,
            (username, repo_name, full_name, url, description, language,
             stars, forks, topics_json, readme_summary, key_features_json,
             target_audience, promotion_hooks_json, homepage),
        )
        conn.commit()
        return int(cursor.lastrowid or 0)


def get_cached_repos(
    username: str,
    *,
    max_age_hours: int = 24,
    database_path: Path | None = None,
) -> list[dict]:
    """Return cached repo profiles that are still fresh."""
    db_path = initialize_persistence_database(database_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM github_repo_cache
            WHERE username = ?
              AND cached_at > datetime('now', ? || ' hours')
            ORDER BY stars DESC
            """,
            (username, f"-{max_age_hours}"),
        ).fetchall()
        return [dict(row) for row in rows]


def record_promotion(
    *,
    repo_name: str,
    platform: str,
    channel: str,
    tactic: str,
    content_angle: str = "",
    draft_message: str = "",
    status: str = "planned",
    database_path: Path | None = None,
) -> int:
    """Record a promotion action for cooldown tracking."""
    db_path = initialize_persistence_database(database_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO promotion_history(
                repo_name, platform, channel, tactic,
                content_angle, draft_message, status
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (repo_name, platform, channel, tactic,
             content_angle, draft_message, status),
        )
        conn.commit()
        return int(cursor.lastrowid or 0)


def check_promotion_cooldown(
    *,
    repo_name: str,
    platform: str,
    channel: str,
    cooldown_days: int = 7,
    database_path: Path | None = None,
) -> bool:
    """Return True if the repo was promoted in this channel within cooldown period."""
    db_path = initialize_persistence_database(database_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM promotion_history
            WHERE repo_name = ? AND platform = ? AND channel = ?
              AND promoted_at > datetime('now', ? || ' days')
              AND status != 'cancelled'
            """,
            (repo_name, platform, channel, f"-{cooldown_days}"),
        ).fetchone()
        return bool(row and row[0] > 0)


def count_platform_promotions_today(
    *,
    platform: str,
    database_path: Path | None = None,
) -> int:
    """Count how many promotions were done on a platform today."""
    db_path = initialize_persistence_database(database_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM promotion_history
            WHERE platform = ?
              AND promoted_at > datetime('now', '-1 day')
              AND status != 'cancelled'
            """,
            (platform,),
        ).fetchone()
        return int(row[0] if row else 0)

