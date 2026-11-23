"""v2_engine — 2026-Standard Dual-Mode Browser Agent Engine for Agent-first-browse.

Coexists cleanly alongside the frozen legacy pipeline so operators can benchmark
and live-test `--mode legacy` vs `--mode v2` on identical URLs and objectives
without touching or risking the proven legacy execution path.

Mode selection (live-read from env, settable via CLI `--mode {legacy,v2}`):
  - AGENT_ENGINE_MODE=legacy (default for backward-compatible callers/tests)
  - AGENT_ENGINE_MODE=v2     (enables adaptive settle, occlusion filtering,
                              Tier-2 A11y escalation, compact/delta snapshots,
                              stale-ref semantic rebind, fixed __aid state
                              safety net, and LangGraph RetryPolicy)
"""

from __future__ import annotations

import os

ENGINE_MODE_LEGACY = "legacy"
ENGINE_MODE_V2 = "v2"
VALID_ENGINE_MODES = (ENGINE_MODE_LEGACY, ENGINE_MODE_V2)


def normalize_engine_mode(mode: str | None) -> str:
    """Normalize a mode string to 'legacy' or 'v2'."""
    if not mode:
        return ENGINE_MODE_LEGACY
    cleaned = str(mode).strip().lower()
    if cleaned in ("v2", "2", "new", "modern"):
        return ENGINE_MODE_V2
    if cleaned in ("legacy", "v1", "1", "old", "default"):
        return ENGINE_MODE_LEGACY
    raise ValueError(f"Unsupported engine mode '{mode}'. Expected one of: {VALID_ENGINE_MODES}")


def get_engine_mode() -> str:
    """Return the active engine mode from AGENT_ENGINE_MODE (default: legacy)."""
    raw = os.getenv("AGENT_ENGINE_MODE")
    if not raw:
        return ENGINE_MODE_LEGACY
    try:
        return normalize_engine_mode(raw)
    except ValueError:
        return ENGINE_MODE_LEGACY


def set_engine_mode(mode: str) -> str:
    """Validate and set AGENT_ENGINE_MODE in the current process environment."""
    normalized = normalize_engine_mode(mode)
    os.environ["AGENT_ENGINE_MODE"] = normalized
    return normalized


def is_v2_mode() -> bool:
    """True when AGENT_ENGINE_MODE is set to 'v2'."""
    return get_engine_mode() == ENGINE_MODE_V2


__all__ = [
    "ENGINE_MODE_LEGACY",
    "ENGINE_MODE_V2",
    "VALID_ENGINE_MODES",
    "get_engine_mode",
    "is_v2_mode",
    "normalize_engine_mode",
    "set_engine_mode",
]
