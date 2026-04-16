"""Centralized environment configuration for Agent First IDE.

All environment variable reads are consolidated here to avoid scattered
os.getenv() calls across the codebase. Import from this module instead
of reading os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.is_file():
    load_dotenv(_ENV_PATH)


# ─── OpenAI / LLM Provider ───────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ─── Worker Vision-Language Model ─────────────────────────────────────────────
WORKER_VLM_API_KEY: str = os.getenv("WORKER_VLM_API_KEY", "") or OPENAI_API_KEY
WORKER_VLM_MODEL: str = os.getenv("WORKER_VLM_MODEL", "gpt-4.1-mini")

# ─── Supervisor LLM ──────────────────────────────────────────────────────────
SUPERVISOR_LLM_MODEL: str = os.getenv("SUPERVISOR_LLM_MODEL", "gpt-4.1-mini")

# ─── Browser Runtime ─────────────────────────────────────────────────────────
BROWSER_MODE: str = os.getenv("BROWSER_MODE", "ISOLATED_SERVER")
BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "true").lower() in ("true", "1", "yes")
LOCAL_CDP_ENDPOINT: str = os.getenv("LOCAL_CDP_ENDPOINT", "http://localhost:9222")
REMOTE_WEBSOCKET_URL: str = os.getenv("REMOTE_WEBSOCKET_URL", "")

# ─── Agent Safety ─────────────────────────────────────────────────────────────
DRY_RUN_MODE: bool = os.getenv("DRY_RUN_MODE", "true").lower() in ("true", "1", "yes")
MAX_CYCLES: int = int(os.getenv("MAX_CYCLES", "15"))
WORKER_CONFIDENCE_THRESHOLD: float = float(os.getenv("WORKER_CONFIDENCE_THRESHOLD", "0.4"))

# ─── Persistence ──────────────────────────────────────────────────────────────
PERSISTENCE_DIR: Path = _PROJECT_ROOT / "persistence"
