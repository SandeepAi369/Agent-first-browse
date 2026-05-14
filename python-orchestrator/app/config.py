"""Centralized environment configuration for Agent First IDE.

All environment variable reads are consolidated here to avoid scattered
os.getenv() calls across the codebase. Import from this module instead
of reading os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

# Load .env from project root if present.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.is_file():
    load_dotenv(_ENV_PATH)


def _split_env_list(raw: str) -> list[str]:
    """Split comma/semicolon-delimited env values into trimmed non-empty items."""
    if not raw.strip():
        return []
    normalized = raw.replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _merge_unique_keys(*groups: str | Iterable[str]) -> list[str]:
    """Merge key groups while preserving order and removing duplicates."""
    merged: list[str] = []
    for group in groups:
        items: list[str]
        if isinstance(group, str):
            items = [group]
        else:
            items = list(group)

        for item in items:
            candidate = item.strip()
            if candidate and candidate not in merged:
                merged.append(candidate)
    return merged


# ─── OpenAI / LLM Provider ───────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "") or os.getenv("OPENAI_API_BASE", "")
OPENAI_API_KEY_FALLBACKS: list[str] = _split_env_list(os.getenv("OPENAI_API_KEY_FALLBACKS", ""))
OPENAI_API_KEYS: list[str] = _merge_unique_keys(OPENAI_API_KEY, OPENAI_API_KEY_FALLBACKS)

# ─── Google LLM Provider (Gemma / Gemini) ───────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
GOOGLE_API_KEY_FALLBACKS: list[str] = _split_env_list(os.getenv("GOOGLE_API_KEY_FALLBACKS", ""))
GEMINI_API_KEY_FALLBACKS: list[str] = _split_env_list(os.getenv("GEMINI_API_KEY_FALLBACKS", ""))
GOOGLE_API_KEYS: list[str] = _merge_unique_keys(
    GOOGLE_API_KEY,
    GOOGLE_API_KEY_FALLBACKS,
    GEMINI_API_KEY_FALLBACKS,
)

# ─── Vision Model (Gemini / Gemma) ──────────────────────────────────────────
VISION_GOOGLE_API_KEY: str = (
    os.getenv("VISION_GOOGLE_API_KEY", "")
    or os.getenv("VISION_GEMINI_API_KEY", "")
    or os.getenv("VISION_API_KEY", "")
)
VISION_GOOGLE_API_KEY_FALLBACKS: list[str] = _split_env_list(
    os.getenv("VISION_GOOGLE_API_KEY_FALLBACKS", "")
)
VISION_GEMINI_API_KEY_FALLBACKS: list[str] = _split_env_list(
    os.getenv("VISION_GEMINI_API_KEY_FALLBACKS", "")
)
VISION_GOOGLE_API_KEYS: list[str] = _merge_unique_keys(
    VISION_GOOGLE_API_KEY,
    VISION_GOOGLE_API_KEY_FALLBACKS,
    VISION_GEMINI_API_KEY_FALLBACKS,
)
VISION_MODEL: str = os.getenv("VISION_MODEL", "gemma-4-32b-it")

# ─── Worker Vision-Language Model ─────────────────────────────────────────────
WORKER_VLM_API_KEY: str = os.getenv("WORKER_VLM_API_KEY", "") or OPENAI_API_KEY
WORKER_VLM_API_KEY_FALLBACKS: list[str] = _split_env_list(os.getenv("WORKER_VLM_API_KEY_FALLBACKS", ""))
WORKER_VLM_API_KEYS: list[str] = _merge_unique_keys(
    WORKER_VLM_API_KEY,
    WORKER_VLM_API_KEY_FALLBACKS,
    OPENAI_API_KEYS,
)
WORKER_VLM_MODEL: str = os.getenv("WORKER_VLM_MODEL", "gemma-4-27b-it")
WORKER_VLM_FALLBACK_MODEL: str = os.getenv("WORKER_VLM_FALLBACK_MODEL", "gemma-3-27b-it")
# How many image/vision calls each API key handles before rotating to the next
WORKER_KEY_ROTATION_BATCH: int = int(os.getenv("WORKER_KEY_ROTATION_BATCH", "2"))

# ─── NVIDIA NIM ───────────────────────────────────────────────────────────────
NVIDIA_NIM_API_KEY: str = os.getenv("NVIDIA_NIM_API_KEY", "")
NVIDIA_NIM_BASE_URL: str = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_NIM_MODEL: str = os.getenv("NVIDIA_NIM_MODEL", "nvidia/glm-5.1")

# ─── Supervisor LLM ──────────────────────────────────────────────────────────
SUPERVISOR_LLM_MODEL: str = os.getenv("SUPERVISOR_LLM_MODEL", "nvidia/glm-5.1")
SUPERVISOR_FALLBACK_MODEL: str = os.getenv("SUPERVISOR_FALLBACK_MODEL", "openai/gpt-oss-120b")
SUPERVISOR_MODEL_API_KEY: str = os.getenv("SUPERVISOR_MODEL_API_KEY", "")
SUPERVISOR_MODEL_API_KEY_FALLBACKS: list[str] = _split_env_list(
    os.getenv("SUPERVISOR_MODEL_API_KEY_FALLBACKS", "")
)
SUPERVISOR_MODEL_API_KEYS: list[str] = _merge_unique_keys(
    SUPERVISOR_MODEL_API_KEY,
    SUPERVISOR_MODEL_API_KEY_FALLBACKS,
    OPENAI_API_KEYS,
)
# Primary endpoint (NVIDIA NIM) vs fallback endpoint (Groq)
SUPERVISOR_PRIMARY_BASE_URL: str = os.getenv("SUPERVISOR_PRIMARY_BASE_URL", "https://integrate.api.nvidia.com/v1")
SUPERVISOR_FALLBACK_BASE_URL: str = os.getenv("SUPERVISOR_FALLBACK_BASE_URL", "https://api.groq.com/openai/v1")

# ─── Browser Runtime ─────────────────────────────────────────────────────────
BROWSER_MODE: str = os.getenv("BROWSER_MODE", "PERSISTENT_CONTEXT")
BROWSER_STRATEGY: str = os.getenv("BROWSER_STRATEGY", "playwright").strip().lower()
BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "false").lower() in ("true", "1", "yes")
FORCE_VISIBLE_LOCAL_BROWSER: bool = os.getenv("FORCE_VISIBLE_LOCAL_BROWSER", "true").lower() in (
    "true",
    "1",
    "yes",
)

# ─── Human-Like Action Pacing ────────────────────────────────────────────────
HUMAN_ACTION_DELAY_MIN_SECONDS: float = float(os.getenv("HUMAN_ACTION_DELAY_MIN_SECONDS", "1.4"))
HUMAN_ACTION_DELAY_MAX_SECONDS: float = float(os.getenv("HUMAN_ACTION_DELAY_MAX_SECONDS", "3.2"))

# ─── Agent Safety ─────────────────────────────────────────────────────────────
DRY_RUN_MODE: bool = os.getenv("DRY_RUN_MODE", "false").lower() in ("true", "1", "yes")
AUTONOMOUS_CONTINUATION: bool = os.getenv("AUTONOMOUS_CONTINUATION", "true").lower() in (
    "true",
    "1",
    "yes",
)
MAX_CYCLES: int = int(os.getenv("MAX_CYCLES", "15"))
WORKER_CONFIDENCE_THRESHOLD: float = float(os.getenv("WORKER_CONFIDENCE_THRESHOLD", "0.4"))
PERSISTENT_THREAD_ID: str = os.getenv("PERSISTENT_THREAD_ID", "").strip()

# ─── LLM API Pacing (to reduce provider limit spikes) ───────────────────────
WORKER_LLM_MIN_GAP_SECONDS: float = float(os.getenv("WORKER_LLM_MIN_GAP_SECONDS", "2.0"))
SUPERVISOR_LLM_MIN_GAP_SECONDS: float = float(os.getenv("SUPERVISOR_LLM_MIN_GAP_SECONDS", "2.5"))

# ─── Zero-Token Executor Fallback Models ─────────────────────────────────────
DOM_SELECTOR_ENABLED: bool = os.getenv("DOM_SELECTOR_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)
DOM_SELECTOR_MODEL: str = os.getenv("DOM_SELECTOR_MODEL", "gemma-3-12b-it")
DOM_SELECTOR_FALLBACK_MODEL: str = os.getenv("DOM_SELECTOR_FALLBACK_MODEL", "gpt-4.1-mini")
DOM_SELECTOR_MAX_CANDIDATES: int = int(os.getenv("DOM_SELECTOR_MAX_CANDIDATES", "72"))
VISION_FALLBACK_ENABLED: bool = os.getenv("VISION_FALLBACK_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)

# ─── LangSmith / Tracing ─────────────────────────────────────────────────────
LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() in (
    "true",
    "1",
    "yes",
)
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "agent-first-ide")
LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "")

# ─── Input Driver ─────────────────────────────────────────────────────────────
# "playwright" = Playwright native API (recommended, no OS focus needed)
# "physical"   = Legacy WSL → PowerShell → user32.dll (deprecated)
INPUT_DRIVER: str = os.getenv("INPUT_DRIVER", "playwright").strip().lower()

# ─── Human Typing Delays (Playwright driver) ─────────────────────────────────
HUMAN_TYPING_DELAY_MIN_MS: int = int(os.getenv("HUMAN_TYPING_DELAY_MIN_MS", "28"))
HUMAN_TYPING_DELAY_MAX_MS: int = int(os.getenv("HUMAN_TYPING_DELAY_MAX_MS", "115"))

# ─── Physical Input Driver (DEPRECATED — WSL → Windows OS-level input) ───────
# Retained for backward compatibility. Set INPUT_DRIVER=physical to use.
PHYSICAL_INPUT_CHROME_HEIGHT_PX: int = int(os.getenv("PHYSICAL_INPUT_CHROME_HEIGHT_PX", "0"))
PHYSICAL_INPUT_DPI_MULTIPLIER: float = float(os.getenv("PHYSICAL_INPUT_DPI_MULTIPLIER", "1.0"))

# ─── GitHub Intelligence ──────────────────────────────────────────────────────
# Repo intelligence for autonomous marketing — reads public repos via API.
GITHUB_USERNAME: str = os.getenv("GITHUB_USERNAME", "SandeepAi369")
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO_CACHE_TTL_HOURS: int = int(os.getenv("GITHUB_REPO_CACHE_TTL_HOURS", "24"))

# ─── Marketing Engine ─────────────────────────────────────────────────────────
# Controls autonomous promotion behavior and safety limits.
# "organic"      = Answer questions naturally, mention project as one option
# "direct"       = "Show & Tell" style posts with project as focus
# "educational"  = Share technical insights, link tools naturally
PROMOTION_STYLE: str = os.getenv("PROMOTION_STYLE", "organic").strip().lower()
MAX_PROMOTIONS_PER_PLATFORM: int = int(os.getenv("MAX_PROMOTIONS_PER_PLATFORM", "2"))
PROMOTION_COOLDOWN_DAYS: int = int(os.getenv("PROMOTION_COOLDOWN_DAYS", "7"))
# Platforms the marketing engine is allowed to target
PROMOTION_PLATFORMS: list[str] = _split_env_list(
    os.getenv("PROMOTION_PLATFORMS", "reddit,github,hackernews")
)

# ─── Persistence ──────────────────────────────────────────────────────────────
PERSISTENCE_DIR: Path = _PROJECT_ROOT / "persistence"

