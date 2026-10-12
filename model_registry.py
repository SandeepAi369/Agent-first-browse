"""ModelRegistry — Model-Agnostic Provider Layer for Agent First IDE v17.0.

Enforces STRICT separation between Text and Vision API pipelines.
No graph node should ever construct an LLM client directly.
All model access goes through this registry.

Reads ALL API keys from .env — Groq, NVIDIA NIM, Cerebras, Google Gemini.

V17.0 — Resilient Reasoning Foundation:
  - MODEL-FIRST failover: the chain is sorted by (model quality tier,
    expected cost) so ALL instances of the best model — every API key,
    every provider hosting it — are exhausted before any weaker model.
    A 429 on Groq key 0 jumps to the SAME model on Groq key 1, then the
    same model on NVIDIA. Never to a lower-tier model while a same-model
    instance remains.
  - Startup probe: dead models (404/401) are pruned for the session;
    survivors get a measured latency seed.
  - EWMA health: per-instance latency and failure-probability estimates
    drive ordering (E[cost] = (1−p)·latency + p·timeout) and adaptive
    per-call timeouts (clamp(3·latency, 6s, 20s)).
  - Per-instance 429 cooldown (jittered) instead of provider-wide skip.
  - JSON-mode fallback when a provider rejects a structured-output schema.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import logging
from collections import deque
from typing import Any
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.is_file():
    load_dotenv(_ENV_PATH)

logger = logging.getLogger("model_registry")


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_keys(*env_vars: str) -> list[str]:
    """Collect all unique API keys from multiple env vars (comma-separated)."""
    keys: list[str] = []
    for var in env_vars:
        raw = os.getenv(var, "").strip()
        if not raw:
            continue
        for k in raw.replace(";", ",").split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Quality Tiers — V17.0 model-first failover
#
#  The failover chain is sorted by (tier, expected_cost): all instances of the
#  best model group at the front regardless of which provider/key hosts them.
#  Lower tier number = higher quality = tried first.
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL_TIERS: dict[str, int] = {
    # Tier 0 — proven excellent for tactical browser decisions
    "gpt-oss-120b": 0,
    # Tier 1 — strong general models (benchmark 2026-06: gpt-oss-20b 88%/1.0s/100%JSON,
    #   gemma-4-31b-it 100% via Gemini — both ideal cascade-consensus voters)
    "gpt-oss-20b": 1,
    "gemma-4-31b-it": 1,
    "gemma-4-32b-it": 1,
    # ── Vision-only models (NEVER used by the text chain — tiering them here is
    #    safe and does not touch text ordering). Without this, the proven-fast
    #    groq vision model fell to UNKNOWN tier and sorted BEHIND the slow
    #    gemini-vision (gemma-4-31b-it, tier 1), wasting ~20s/consult. ──
    "llama-4-scout-17b-16e-instruct": 0,        # groq-vision PRIMARY — proven, fast
    "llama-3.2-11b-vision-instruct": 1,         # NVIDIA — fast, lightweight (~0.6s)
    "llama-3.1-nemotron-nano-vl-8b-v1": 1,      # NVIDIA — fast nano VL (~0.7s)
    "llama-4-maverick-17b-128e-instruct": 1,    # NVIDIA — strong, 1M ctx native vision
    "llama-3.2-90b-vision-instruct": 2,         # NVIDIA — strong 90B fallback
}
UNKNOWN_MODEL_TIER = 3


def normalize_model_id(model_id: str) -> str:
    """'openai/gpt-oss-120b' → 'gpt-oss-120b'; 'google/gemma-4-31b-it' → 'gemma-4-31b-it'."""
    return model_id.rsplit("/", 1)[-1].strip().lower()


def _load_tier_overrides() -> dict[str, int]:
    """Parse MODEL_TIER_OVERRIDE='gpt-oss-120b=0,some-model=2' from .env."""
    overrides: dict[str, int] = {}
    raw = os.getenv("MODEL_TIER_OVERRIDE", "").strip()
    for part in raw.split(","):
        if "=" in part:
            mid, _, rank = part.partition("=")
            try:
                overrides[normalize_model_id(mid)] = int(rank)
            except ValueError:
                pass
    return overrides


MODEL_TIERS: dict[str, int] = {**DEFAULT_MODEL_TIERS, **_load_tier_overrides()}


def get_model_tier(instance_name: str) -> int:
    """Quality tier for an instance name like 'groq:openai/gpt-oss-120b:0'."""
    base = ProviderHealthTracker._base_model_name(instance_name)
    return MODEL_TIERS.get(normalize_model_id(base), UNKNOWN_MODEL_TIER)


# ═══════════════════════════════════════════════════════════════════════════════
#  V24 — Agentic Capability Gate + dual-mode (free / premium). See STRATEGY.md.
# ═══════════════════════════════════════════════════════════════════════════════

# Normalized base model names PROVEN (live probe, 2026-06) to do agentic
# structured output. Shipped default + SAFETY FLOOR for the capability gate
# (if gating would empty a pipeline, these alive models are restored).
AGENTIC_TEXT_ALLOWLIST: set[str] = {
    "gpt-oss-120b",                        # Groq + NVIDIA + Cerebras — gold standard
    "gpt-oss-20b",                         # NVIDIA — fast (1.1s), reliable
    "gemma-4-31b-it",                      # via Gemini (fails on NVIDIA → probe drops that combo)
    "gemma-4-32b-it",
    "llama-3.3-70b-instruct",              # NVIDIA — works, slower
    "llama-3.3-nemotron-super-49b-v1.5",   # NVIDIA — works, medium
}

# Role separation: the worker (action decisions) only uses models at or below
# this tier. Auxiliary calls (planner, PRM, judge) use the full chain.
WORKER_MAX_TIER = 1


def get_agent_mode() -> str:
    """'premium' | 'free'. AGENT_MODE=auto (default) → premium iff a PREMIUM_API_KEY
    is set; 'premium'/'free' force the respective path."""
    mode = os.getenv("AGENT_MODE", "auto").strip().lower()
    if mode == "premium":
        return "premium"
    if mode == "free":
        return "free"
    return "premium" if os.getenv("PREMIUM_API_KEY", "").strip() else "free"


def _premium_config() -> dict:
    """Premium single-key/model config — one paid key that does text + vision."""
    return {
        "keys": _collect_keys("PREMIUM_API_KEY", "PREMIUM_API_KEYS"),
        "model": os.getenv("PREMIUM_MODEL", "").strip(),
        "vision_model": os.getenv("PREMIUM_VISION_MODEL", "").strip(),
        "base_url": os.getenv("PREMIUM_BASE_URL", "https://api.openai.com/v1").strip(),
        "provider": os.getenv("PREMIUM_PROVIDER", "openai").strip().lower(),
        "timeout": int(os.getenv("PREMIUM_TIMEOUT", "60")),
    }


def _combo_of(mc: "ModelClient") -> tuple[str, str, str]:
    """(provider, normalized_base_model, pipeline) — the probe/gate identity key."""
    return (mc.provider,
            normalize_model_id(ProviderHealthTracker._base_model_name(mc.name)),
            mc.pipeline)


# ═══════════════════════════════════════════════════════════════════════════════
#  Provider Health Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class ProviderHealthTracker:
    """Per-model exponential backoff quarantine with V15.0/V15.1 improvements.

    V15.0 F6:
      - Quarantine cap at 16s (was uncapped up to 120s)
      - Cold-start reset: if model idle 5+ min, streak resets to 0
      - Schema blacklist: models that fail structured output permanently

    V15.1 Patch A+C:
      - Base-model-name blacklist: 'groq:openai/gpt-oss-120b:0' blacklist
        also blocks ':1' and ':2' (same underlying model, different API key)
      - Per-schema blacklist: 'ChecklistEvaluation' vs 'CandidateSet' tracked
        separately so a model that fails on one schema can still serve others
    """

    # EWMA smoothing factor for latency / failure-probability estimates
    EWMA_ALPHA = 0.3
    # Assumed latency (s) for an instance we have never measured
    DEFAULT_LATENCY = 8.0

    def __init__(self):
        self._health: dict[str, dict] = {}
        self._schema_blacklist: dict[str, set[str]] = {}  # "provider|base_model" → {schema_names}

    @staticmethod
    def _base_model_name(instance_name: str) -> str:
        """Extract base model from instance name.

        'groq:openai/gpt-oss-120b:0'  → 'openai/gpt-oss-120b'
        'nvidia-text:z-ai/glm-5.1:1'  → 'z-ai/glm-5.1'
        'gemini-text:gemma-4-31b-it:2' → 'gemma-4-31b-it'
        'some-model'                   → 'some-model' (fallback)
        """
        parts = instance_name.split(":")
        if len(parts) >= 3:
            # provider:model_path:key_index → extract model_path
            return ":".join(parts[1:-1])
        return instance_name

    @staticmethod
    def _blacklist_key(instance_name: str) -> str:
        """Blacklist key scoped to (provider, base_model).

        V17.0: schema support is a property of the provider+model combo, not
        the model name globally — gpt-oss may reject a schema on Groq while
        serving it fine on NVIDIA. Never bench a good model globally.
        """
        provider = instance_name.split(":", 1)[0]
        base = ProviderHealthTracker._base_model_name(instance_name)
        return f"{provider}|{base}"

    def _get(self, name: str) -> dict:
        if name not in self._health:
            self._health[name] = {
                "consecutive_failures": 0,
                "quarantine_until": 0.0,
                "total_calls": 0,
                "total_failures": 0,
                "last_call_time": 0.0,
                # V17.0 — EWMA health + cooldown
                "latency_ewma": None,   # seconds; None = never measured
                "p_fail": 0.0,          # smoothed failure probability
                "cooldown_until": 0.0,  # 429 cooldown deadline (monotonic)
            }
        return self._health[name]

    # ── V17.0: EWMA + cooldown API ─────────────────────────────────────────

    def observe_latency(self, name: str, seconds: float) -> None:
        """Fold an observed call latency into the EWMA estimate."""
        s = self._get(name)
        if s["latency_ewma"] is None:
            s["latency_ewma"] = seconds
        else:
            s["latency_ewma"] = (
                self.EWMA_ALPHA * seconds + (1 - self.EWMA_ALPHA) * s["latency_ewma"]
            )

    def seed_latency(self, name: str, seconds: float) -> None:
        """Seed the latency estimate (from the startup probe) without overriding live data."""
        s = self._get(name)
        if s["latency_ewma"] is None:
            s["latency_ewma"] = seconds

    def _update_p_fail(self, name: str, failed: bool) -> None:
        s = self._get(name)
        s["p_fail"] = (
            self.EWMA_ALPHA * (1.0 if failed else 0.0)
            + (1 - self.EWMA_ALPHA) * s["p_fail"]
        )

    def start_cooldown(self, name: str, seconds: float) -> None:
        """Put one instance (provider+key+model) on a rate-limit cooldown.

        Per-instance, NOT per-provider: Groq limits are per key/org, so a 429
        on key 0 says nothing about key 1 — the same model on the next key is
        tried immediately.
        """
        s = self._get(name)
        s["cooldown_until"] = time.monotonic() + seconds
        logger.info("⏸️ '%s' on 429 cooldown for %.0fs (same model continues on other keys)",
                    name, seconds)

    def in_cooldown(self, name: str) -> bool:
        return time.monotonic() < self._get(name)["cooldown_until"]

    def expected_cost(self, name: str, default_timeout: float = 20.0) -> float:
        """E[time-to-answer] = (1−p_fail)·latency + p_fail·timeout.

        Sorting instances ascending by this value minimizes expected
        time-to-first-success (greedy ordered-search optimality).
        """
        s = self._get(name)
        lat = s["latency_ewma"] if s["latency_ewma"] is not None else self.DEFAULT_LATENCY
        p = s["p_fail"]
        return (1.0 - p) * lat + p * default_timeout

    def adaptive_timeout(self, name: str, k: float = 3.0,
                         floor: float = 6.0, cap: float = 20.0) -> float:
        """Per-instance timeout: clamp(k·latency_ewma, floor, cap).

        Unknown models get the full cap; a chronically slow model is
        abandoned at the cap instead of a fixed 30s.
        """
        s = self._get(name)
        if s["latency_ewma"] is None:
            return cap
        return max(floor, min(cap, k * s["latency_ewma"]))

    def is_available(self, name: str, schema_name: str = "") -> bool:
        # V17.0: Check schema blacklist by (provider, base_model)
        if schema_name and self.is_schema_blacklisted(name, schema_name):
            return False

        s = self._get(name)
        # V15.0: Cold-start reset — if model idle 5+ min, treat as fresh
        if s["last_call_time"] > 0 and time.monotonic() - s["last_call_time"] > 300:
            s["consecutive_failures"] = 0
            s["quarantine_until"] = 0.0

        return time.monotonic() >= s["quarantine_until"]

    def record_success(self, name: str, latency: float | None = None) -> None:
        s = self._get(name)
        s["consecutive_failures"] = 0
        s["quarantine_until"] = 0.0
        s["total_calls"] += 1
        s["last_call_time"] = time.monotonic()
        self._update_p_fail(name, failed=False)
        if latency is not None:
            self.observe_latency(name, latency)

    def record_failure(self, name: str, is_rate_limit: bool = False,
                       error_msg: str = "", latency: float | None = None) -> None:
        s = self._get(name)
        s["consecutive_failures"] += 1
        s["total_calls"] += 1
        s["total_failures"] += 1
        s["last_call_time"] = time.monotonic()
        self._update_p_fail(name, failed=True)
        if latency is not None:
            # A timeout means "at least this slow" — fold it into the estimate
            self.observe_latency(name, latency)

        # V17.0: Schema blacklist keyed by (provider, base_model) — a model that
        # rejects a schema on Groq may serve it fine on NVIDIA.
        if error_msg and ("400" in error_msg or "Bad Request" in error_msg):
            if "additionalProperties" in error_msg or "schema" in error_msg.lower():
                bl_key = self._blacklist_key(name)
                if bl_key not in self._schema_blacklist:
                    self._schema_blacklist[bl_key] = set()

                # Extract specific schema name from error message
                # Error format: "response_format: 'ChecklistEvaluation': /properties/..."
                schema_match = re.search(r"response_format:\s*'(\w+)'", error_msg)
                if not schema_match:
                    # Fallback: look for CamelCase class names (Pydantic schema pattern)
                    schema_match = re.search(r"'([A-Z]\w{5,})'", error_msg)
                specific_schema = schema_match.group(1) if schema_match else None

                if specific_schema:
                    self._schema_blacklist[bl_key].add(specific_schema)
                    logger.warning(
                        "🚫 '%s' blacklisted for schema '%s' — will use JSON-mode fallback",
                        bl_key, specific_schema,
                    )
                else:
                    # Catch-all: blacklist for all structured output
                    self._schema_blacklist[bl_key].add("__structured_output__")
                    logger.warning(
                        "🚫 '%s' blacklisted for ALL structured output — JSON-mode fallback",
                        bl_key,
                    )

    def is_schema_blacklisted(self, name: str, schema_name: str = "__structured_output__") -> bool:
        """Check if a (provider, model) combo is blacklisted for a specific schema."""
        bl_key = self._blacklist_key(name)
        if bl_key not in self._schema_blacklist:
            return False
        bl = self._schema_blacklist[bl_key]
        return schema_name in bl or "__structured_output__" in bl

    def all_quarantined(self, names: list[str]) -> bool:
        return all(not self.is_available(n) for n in names)

    def shortest_quarantine_wait(self) -> float:
        """Seconds until the nearest quarantined model recovers."""
        now = time.monotonic()
        waits = [s["quarantine_until"] - now for s in self._health.values() if s["quarantine_until"] > now]
        return min(waits) if waits else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Circuit Breaker — 3-State Machine (Resilience4j-inspired)
#  CLOSED → OPEN → HALF_OPEN → CLOSED
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Three-state circuit breaker with sliding window and auto-recovery.

    - CLOSED: all requests pass; track outcomes in a sliding window.
      Trip to OPEN when failure RATE over last N >= threshold AND
      a minimum number of calls recorded (prevents 1/1 false trips).
    - OPEN: reject immediately. After wait_secs → HALF_OPEN.
    - HALF_OPEN: allow limited probes; all pass → CLOSED; any fail → OPEN.
    """

    def __init__(
        self,
        window_size: int = 50,
        min_calls: int = 10,
        failure_rate_threshold: float = 0.5,
        open_wait_secs: float = 30.0,
        half_open_probes: int = 3,
        # Legacy compat — ignored but accepted so existing code doesn't break
        max_requests: int = 80,
        max_failures: int = 15,
    ):
        self._window: deque[bool] = deque(maxlen=window_size)  # True = failure
        self._min_calls = min_calls
        self._failure_rate_threshold = failure_rate_threshold
        self._open_wait_secs = open_wait_secs
        self._max_probes = half_open_probes
        self._state = "closed"
        self._opened_at = 0.0
        self._probes_done = 0
        self._probe_failures = 0
        self._total_requests = 0
        self._total_failures = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    @property
    def tripped(self) -> bool:
        """Legacy compat: True if OPEN and not yet eligible for HALF_OPEN."""
        if self._state == "open":
            if time.monotonic() - self._opened_at >= self._open_wait_secs:
                return False  # Eligible for half-open transition
            return True
        return False

    async def allow(self) -> bool:
        """Check if a request should be allowed through."""
        async with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self._open_wait_secs:
                    self._state = "half_open"
                    self._probes_done = 1  # this request IS the first probe
                    self._probe_failures = 0
                    logger.info("⚡ Circuit breaker → HALF_OPEN (probing)")
                    return True
                return False
            # half_open
            if self._probes_done < self._max_probes:
                self._probes_done += 1
                return True
            return False

    async def record(self, is_failure: bool) -> None:
        """Record the outcome of a request."""
        async with self._lock:
            self._total_requests += 1
            if is_failure:
                self._total_failures += 1

            if self._state == "half_open":
                if is_failure:
                    self._probe_failures += 1
                    self._trip("Half-open probe failed")
                elif self._probes_done >= self._max_probes and self._probe_failures == 0:
                    self._reset()
                return

            self._window.append(is_failure)
            if len(self._window) >= self._min_calls:
                rate = sum(self._window) / len(self._window)
                if rate >= self._failure_rate_threshold:
                    self._trip(f"Failure rate {rate:.0%} >= {self._failure_rate_threshold:.0%}")

    # Legacy sync API (backward compat for callers that haven't migrated)
    def record_success(self) -> None:
        self._total_requests += 1
        self._window.append(False)

    def record_failure(self) -> None:
        self._total_requests += 1
        self._total_failures += 1
        self._window.append(True)
        if len(self._window) >= self._min_calls:
            rate = sum(self._window) / len(self._window)
            if rate >= self._failure_rate_threshold:
                self._trip(f"Failure rate {rate:.0%} (sync)")

    def _trip(self, reason: str) -> None:
        self._state = "open"
        self._opened_at = time.monotonic()
        logger.warning("🔌 Circuit breaker → OPEN: %s", reason)

    def _reset(self) -> None:
        self._state = "closed"
        self._window.clear()
        self._probes_done = 0
        logger.info("✅ Circuit breaker → CLOSED (recovered)")

    @property
    def reason(self) -> str:
        if self._state == "open":
            remaining = max(0, self._open_wait_secs - (time.monotonic() - self._opened_at))
            return f"OPEN (recovery in {remaining:.0f}s)"
        return ""

    def status_line(self) -> str:
        fails = sum(self._window)
        total = len(self._window)
        return f"[CB:{self._state} {fails}/{total} fails, {self._total_requests} total]"


# ═══════════════════════════════════════════════════════════════════════════════
#  ModelClient wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class ModelClient:
    __slots__ = ("name", "client", "provider", "pipeline")

    def __init__(self, name: str, client: Any, provider: str, pipeline: str):
        self.name = name
        self.client = client
        self.provider = provider
        self.pipeline = pipeline

    def __repr__(self) -> str:
        return f"ModelClient({self.name}, pipeline={self.pipeline})"


# ═══════════════════════════════════════════════════════════════════════════════
def _build_text_pipeline() -> list[ModelClient]:
    """TEXT-ONLY pipeline for Manager & Helper. NEVER receives images.

    V16.1 — Multi-model-per-key architecture:
      On NVIDIA NIM, ONE API key gives access to ALL models. So we register
      multiple models under the same key. The failover loop tries every model
      on each key before moving to the next provider.
    """
    from langchain_openai import ChatOpenAI

    clients: list[ModelClient] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    }

    # ── Groq — PRIMARY (all keys from .env) ──
    groq_keys = _collect_keys(
        "GROQ_API_KEY",
        "GROQ_API_KEYS",
        "OPENAI_API_KEY_FALLBACKS",
        "SUPERVISOR_MODEL_API_KEY_FALLBACKS",
    )
    groq_keys = [k for k in groq_keys if k.startswith("gsk_")]
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    for idx, key in enumerate(groq_keys):
        clients.append(ModelClient(
            name=f"groq:{model}:{idx}",
            client=ChatOpenAI(
                model=model, api_key=key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.0, timeout=30,
                default_headers=headers,
            ),
            provider="groq", pipeline="text",
        ))
    if groq_keys:
        logger.info("TEXT ── Groq (PRIMARY): %d keys loaded (%s)", len(groq_keys), model)

    # ── NVIDIA NIM — SECONDARY (multiple models, same key) ──
    # V17.0: Register the SAME top-tier model (gpt-oss-120b) here too, so the
    # model-first failover can continue with the identical model on NVIDIA when
    # all Groq keys are rate-limited. If the catalog ID is invalid on NVIDIA,
    # the startup probe prunes it automatically — zero per-step cost.
    # Override via NVIDIA_TEXT_MODELS="model1,model2" in .env.
    nvidia_keys = _collect_keys("NVIDIA_NIM_API_KEY", "NVIDIA_NIM_API_KEYS")
    base = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    # gpt-oss-20b added (benchmark 2026-06: fast 1.0s, 88% acc, 100% JSON) — gives the
    # dynamic cascade-consensus a fast, reliable, DISTINCT tertiary voter.
    _nvidia_models_raw = os.getenv(
        "NVIDIA_TEXT_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b,google/gemma-4-31b-it"
    )
    nvidia_text_models = [
        (m.strip(), 30) for m in _nvidia_models_raw.split(",") if m.strip()
    ]
    # To add more NVIDIA text models, set NVIDIA_TEXT_MODELS in .env. The startup
    # capability gate auto-drops any that 404 or can't do structured output, so
    # only proven-agentic models ever reach the chain (see STRATEGY.md).
    for key_idx, key in enumerate(nvidia_keys):
        for model_name, timeout_s in nvidia_text_models:
            clients.append(ModelClient(
                name=f"nvidia-text:{model_name}:{key_idx}",
                client=ChatOpenAI(
                    model=model_name, api_key=key,
                    base_url=base, temperature=0.0, timeout=timeout_s,
                    default_headers=headers,
                ),
                provider="nvidia", pipeline="text",
            ))
    if nvidia_keys:
        logger.info(
            "TEXT ── NVIDIA NIM (SECONDARY): %d keys × %d models = %d instances",
            len(nvidia_keys), len(nvidia_text_models),
            len(nvidia_keys) * len(nvidia_text_models),
        )

    # ── Cerebras — TERTIARY ──
    cerebras_keys = _collect_keys("CEREBRAS_API_KEY", "CEREBRAS_API_KEYS")
    if cerebras_keys:
        # Cerebras dropped qwen-3-235b-a22b (404) but serves gpt-oss-120b — the
        # same proven agentic model, very fast on Cerebras. Override: CEREBRAS_MODEL.
        cerebras_model = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
        cerebras_base = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
        for idx, key in enumerate(cerebras_keys):
            clients.append(ModelClient(
                name=f"cerebras:{cerebras_model}:{idx}",
                client=ChatOpenAI(
                    model=cerebras_model, api_key=key,
                    base_url=cerebras_base, temperature=0.0, timeout=20,
                    default_headers=headers,
                ),
                provider="cerebras", pipeline="text",
            ))
        logger.info("TEXT ── Cerebras (TERTIARY): %d keys loaded (%s)", len(cerebras_keys), cerebras_model)

    # ── Google Gemini — QUATERNARY TEXT FALLBACK ──
    gemini_text_keys = _collect_keys("GEMINI_API_KEY", "GEMINI_API_KEY_FALLBACKS")
    if gemini_text_keys:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            gemini_text_model = os.getenv("GEMINI_TEXT_MODEL", "gemma-4-31b-it")
            for idx, key in enumerate(gemini_text_keys):
                clients.append(ModelClient(
                    name=f"gemini-text:{gemini_text_model}:{idx}",
                    client=ChatGoogleGenerativeAI(
                        model=gemini_text_model,
                        google_api_key=key,
                        temperature=0.0,
                        timeout=30,
                    ),
                    provider="google", pipeline="text",
                ))
            logger.info("TEXT ── Google Gemini (QUATERNARY): %d keys loaded (%s)", len(gemini_text_keys), gemini_text_model)
        except ImportError:
            logger.warning("TEXT ── langchain-google-genai not installed, skipping Gemini text fallback")
        except Exception as e:
            logger.error("TEXT ── Gemini text fallback bootstrap failed: %s", e)

    return clients


# ═══════════════════════════════════════════════════════════════════════════════
#  VISION Pipeline — Google Gemini (proven) + NVIDIA NIM Vision (fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_vision_pipeline() -> list[ModelClient]:
    """VISION-ONLY pipeline for Executor. ALWAYS receives images."""
    clients: list[ModelClient] = []




    # ── Groq Vision — PRIMARY (Llama 4 Scout) ──
    groq_vision_key = os.getenv("GROQ_VISION_API_KEY", "").strip()
    if groq_vision_key:
        from langchain_openai import ChatOpenAI
        model = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        clients.append(ModelClient(
            name=f"groq-vision:{model}:0",
            client=ChatOpenAI(
                model=model, api_key=groq_vision_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.0, timeout=30,
            ),
            provider="groq", pipeline="vision",
        ))
        logger.info("VISION ── Groq (PRIMARY): 1 key loaded (%s)", model)

    # ── Google Gemini — PRIMARY VISION (2 keys, gemma-4 vision model) ──
    gemini_keys = _collect_keys(
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_FALLBACKS",
        "VISION_GOOGLE_API_KEY",
        "VISION_GOOGLE_API_KEY_FALLBACKS",
        "GOOGLE_API_KEY",
    )
    if gemini_keys:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = os.getenv("WORKER_VLM_MODEL", "gemma-4-31b-it")
            for idx, key in enumerate(gemini_keys):
                clients.append(ModelClient(
                    name=f"gemini-vision:{model}:{idx}",
                    client=ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=key,
                        temperature=0.0,
                        timeout=45,
                    ),
                    provider="google", pipeline="vision",
                ))
            logger.info("VISION ── Google Gemini (PRIMARY): %d keys loaded (%s)", len(gemini_keys), model)
        except ImportError:
            logger.warning("VISION ── langchain-google-genai not installed, skipping Gemini")
        except Exception as e:
            logger.error("VISION ── Gemini bootstrap failed: %s", e)

    # ── NVIDIA NIM Vision — SECONDARY (multi-model × ALL NVIDIA keys) ──
    # V16.1: Collect ALL NVIDIA keys so every vision model gets tried on every key
    nvidia_vision_keys = _collect_keys(
        "NVIDIA_VISION_API_KEY",
        "NVIDIA_NIM_API_KEY",
        "NVIDIA_NIM_API_KEYS",
    )
    if nvidia_vision_keys:
        from langchain_openai import ChatOpenAI
        base = os.getenv("NVIDIA_VISION_BASE_URL", "https://integrate.api.nvidia.com/v1")

        # OPTIONAL + env-configurable via NVIDIA_VISION_MODELS (comma-separated),
        # mirroring NVIDIA_TEXT_MODELS. Default leads with Llama 4 Maverick (the
        # chosen vision model); the rest are optional fallbacks. All probed
        # 2026-06 to respond <1s; the startup capability gate drops any that 404
        # or can't structure. To use ONLY Maverick, set
        # NVIDIA_VISION_MODELS="meta/llama-4-maverick-17b-128e-instruct".
        _nv_vision_raw = os.getenv(
            "NVIDIA_VISION_MODELS",
            "meta/llama-4-maverick-17b-128e-instruct,"      # Llama 4 Maverick — primary
            "meta/llama-3.2-11b-vision-instruct,"           # fast, lightweight fallback
            "nvidia/llama-3.1-nemotron-nano-vl-8b-v1,"      # fast nano VL fallback
            "meta/llama-3.2-90b-vision-instruct",           # strong 90B fallback
        )
        _nv_vision_timeout = int(os.getenv("NVIDIA_VISION_TIMEOUT", "30"))
        nvidia_vision_models = [
            (m.strip(), _nv_vision_timeout)
            for m in _nv_vision_raw.split(",") if m.strip()
        ]
        for key_idx, key in enumerate(nvidia_vision_keys):
            for model_name, timeout_s in nvidia_vision_models:
                clients.append(ModelClient(
                    name=f"nvidia-vision:{model_name}:{key_idx}",
                    client=ChatOpenAI(
                        model=model_name, api_key=key,
                        base_url=base, temperature=0.0, timeout=timeout_s,
                    ),
                    provider="nvidia", pipeline="vision",
                ))
        logger.info(
            "VISION ── NVIDIA NIM (SECONDARY): %d keys × %d models = %d instances",
            len(nvidia_vision_keys), len(nvidia_vision_models),
            len(nvidia_vision_keys) * len(nvidia_vision_models),
        )

    return clients


# ═══════════════════════════════════════════════════════════════════════════════
#  PREMIUM Pipeline — one paid key/model serves BOTH text and vision (V24)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_premium_pipeline() -> tuple[list["ModelClient"], list["ModelClient"]]:
    """Premium mode: a single top-tier multimodal model drives everything.

    Trusted by definition — no probe, no capability gate, no free-tier fallback
    juggling. OpenAI-compatible by default (covers OpenAI / OpenRouter / Together
    / Fireworks / any gateway); PREMIUM_PROVIDER=google uses the Gemini client.
    Returns (text_clients, vision_clients); both point at the premium model
    (vision uses PREMIUM_VISION_MODEL if set, else the same multimodal model).
    """
    cfg = _premium_config()
    keys, model = cfg["keys"], cfg["model"]
    if not keys or not model:
        logger.error("PREMIUM mode needs PREMIUM_API_KEY and PREMIUM_MODEL — none/partial set.")
        return [], []

    def _mk(m: str, key: str, idx: int, pipeline: str) -> "ModelClient":
        if cfg["provider"] == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            client = ChatGoogleGenerativeAI(
                model=m, google_api_key=key, temperature=0.0, timeout=cfg["timeout"])
        else:
            from langchain_openai import ChatOpenAI
            client = ChatOpenAI(
                model=m, api_key=key, base_url=cfg["base_url"],
                temperature=0.0, timeout=cfg["timeout"])
        return ModelClient(name=f"premium-{pipeline}:{m}:{idx}", client=client,
                           provider="premium", pipeline=pipeline)

    text = [_mk(model, k, i, "text") for i, k in enumerate(keys)]
    vmodel = cfg["vision_model"] or model
    vision = [_mk(vmodel, k, i, "vision") for i, k in enumerate(keys)]
    logger.info("⭐ PREMIUM mode: text='%s', vision='%s' on %d key(s) — no probe, no gate",
                model, vmodel, len(keys))
    return text, vision


# ═══════════════════════════════════════════════════════════════════════════════
#  ModelRegistry Singleton
# ═══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """Centralized model registry. Strict TEXT vs VISION separation."""

    _instance: "ModelRegistry | None" = None

    def __init__(self):
        # V24 dual-mode: premium (one trusted paid key) vs free (juggle + gate).
        self.mode = get_agent_mode()
        if self.mode == "premium":
            self._text_pipeline, self._vision_pipeline = _build_premium_pipeline()
            self._probed = True  # premium is trusted — never probe or capability-gate
            if not self._text_pipeline:
                logger.warning("⭐ PREMIUM misconfigured (need PREMIUM_API_KEY + "
                               "PREMIUM_MODEL) — falling back to FREE tier.")
                self.mode = "free"
                self._probed = False
        if self.mode == "free":
            self._text_pipeline = _build_text_pipeline()
            self._vision_pipeline = _build_vision_pipeline()
            self._probed = False
        self.health = ProviderHealthTracker()
        # UCRF: Higher tolerance — LLM failovers are expected, not critical failures
        # With 9-model chain, 3-4 timeouts per call is normal before success
        self.breaker = CircuitBreaker(
            window_size=50,
            min_calls=20,                  # Need 20+ calls before evaluating (was 10)
            failure_rate_threshold=0.70,   # 70% failure rate to trip (was 50%)
            open_wait_secs=10.0,           # Recover faster (was 30s)
            half_open_probes=3,
        )

        # Validate no overlap
        t = {m.name for m in self._text_pipeline}
        v = {m.name for m in self._vision_pipeline}
        overlap = t & v
        if overlap:
            raise RuntimeError(f"Pipeline overlap: {overlap}")

        logger.info(
            "ModelRegistry [%s mode] — TEXT: %d models, VISION: %d models",
            self.mode, len(self._text_pipeline), len(self._vision_pipeline),
        )

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def get_text_chain(self) -> list[ModelClient]:
        return list(self._text_pipeline)

    def get_text_chain_names(self) -> list[str]:
        return [m.name for m in self._text_pipeline]

    def has_text_models(self) -> bool:
        return len(self._text_pipeline) > 0

    def get_vision_chain(self) -> list[ModelClient]:
        return list(self._vision_pipeline)

    def get_vision_chain_names(self) -> list[str]:
        return [m.name for m in self._vision_pipeline]

    def has_vision_models(self) -> bool:
        return len(self._vision_pipeline) > 0

    def get_worker_chain(self) -> list[ModelClient]:
        """V24 role separation — the WORKER (action decisions, the critical path)
        draws only from the most capable models. Premium → the single premium
        model. Free → tier ≤ WORKER_MAX_TIER. Never returns empty (falls back to
        the full text chain), so the worker always has a model."""
        if self.mode == "premium":
            return list(self._text_pipeline)
        top = [mc for mc in self._text_pipeline
               if get_model_tier(mc.name) <= WORKER_MAX_TIER]
        return top or list(self._text_pipeline)

    def get_worker_chain_names(self) -> list[str]:
        return [mc.name for mc in self.get_worker_chain()]

    def _apply_capability_gate(
        self, pipeline: list[ModelClient],
        dead: set, incapable: set, label: str,
    ) -> list[ModelClient]:
        """Remove dead + incapable combos. SAFETY: if that empties the pipeline,
        restore the alive (non-dead) combos — preferring allowlisted ones — so a
        pipeline is NEVER empty (invariant #2)."""
        kept = [mc for mc in pipeline
                if _combo_of(mc) not in dead and _combo_of(mc) not in incapable]
        if kept:
            if len(kept) != len(pipeline):
                logger.info("🔬 Capability gate [%s]: %d → %d models",
                            label, len(pipeline), len(kept))
            return kept
        alive = [mc for mc in pipeline if _combo_of(mc) not in dead]
        allow = [mc for mc in alive
                 if normalize_model_id(ProviderHealthTracker._base_model_name(mc.name))
                 in AGENTIC_TEXT_ALLOWLIST]
        floor = allow or alive or pipeline
        logger.warning("🛟 Capability gate would EMPTY %s — restoring %d floor model(s)",
                       label, len(floor))
        return floor

    def summary(self) -> str:
        t = " → ".join(self.get_text_chain_names()) or "(none)"
        v = " → ".join(self.get_vision_chain_names()) or "(none)"
        return f"TEXT chain ({len(self._text_pipeline)}): {t}\nVISION chain ({len(self._vision_pipeline)}): {v}"

    # ── V17.0: Startup probe — prune dead models, seed latency ──────────────

    async def probe_and_prune(self, timeout: float = 8.0) -> None:
        """Probe one instance per (provider, base_model, pipeline) concurrently.

        - 404 / not-found / invalid-key → ALL instances of that combo are
          removed from the chain for this session (no per-step 404 tax).
        - Success → measured latency seeds the EWMA for every sibling instance.
        - Timeout → kept (may be transient) but seeded with a high latency so
          it sorts to the back of its tier.
        - 429 → model exists; kept, no penalty.

        Idempotent: runs once per process. Total wall cost ≈ one round-trip.
        """
        if self._probed:
            return
        self._probed = True

        from langchain_core.messages import HumanMessage

        # One representative instance per (provider, base_model, pipeline)
        reps: dict[tuple[str, str, str], ModelClient] = {}
        for mc in self._text_pipeline + self._vision_pipeline:
            key = (mc.provider, normalize_model_id(
                ProviderHealthTracker._base_model_name(mc.name)), mc.pipeline)
            reps.setdefault(key, mc)

        if not reps:
            return

        # V24 CAPABILITY probe (not just liveness): each model must produce a
        # valid structured object for a tiny agentic task. Alive-but-can't-
        # structure models (e.g. gemma-4-31b-it on NVIDIA) are EXCLUDED — that's
        # the agentic capability gate. See STRATEGY.md §2.
        from pydantic import BaseModel as _BM, Field as _F
        from langchain_core.messages import SystemMessage

        class _CapProbe(_BM):
            action: str = _F(description="one of: click, type, scroll, done")
            element_id: str = _F(description="an element id like e5")
            confidence: float = _F(description="0..1")

        cap_messages = [
            SystemMessage(content="You are a browser agent. Choose the next action and "
                                  "reply with the structured fields."),
            HumanMessage(content="Page elements: [e3] search box (empty), [e7] Submit "
                                 "button. Goal: type 'running shoes' into the search box. "
                                 "Give your next action."),
        ]

        _DEAD = ("404", "401", "410")
        _DEAD_L = ("not_found", "does not exist", "model_not_found", "invalid api key",
                   "invalid_api_key", "unauthorized", "decommissioned", "gone")

        def _kind(exc) -> str:
            s = str(exc); sl = s.lower()
            if any(d in s for d in _DEAD) or any(d in sl for d in _DEAD_L):
                return "dead"
            if "429" in s or "rate limit" in sl or "rate_limit" in sl:
                return "transient"
            return "other"

        async def _probe(combo_key, mc: ModelClient):
            t0 = time.monotonic()
            try:  # 1) strict structured output — success ⇒ the model CAN structure
                await asyncio.wait_for(
                    mc.client.with_structured_output(_CapProbe).ainvoke(cap_messages),
                    timeout=timeout)
                return combo_key, time.monotonic() - t0, "capable", "strict"
            except asyncio.TimeoutError:
                return combo_key, time.monotonic() - t0, "timeout", None
            except Exception as exc:
                k = _kind(exc)
                if k == "dead":
                    return combo_key, time.monotonic() - t0, "dead", str(exc)[:120]
                if k == "transient":
                    return combo_key, time.monotonic() - t0, "transient", str(exc)[:100]
                try:  # 2) strict rejected the schema — does JSON-mode rescue work?
                    await asyncio.wait_for(
                        _invoke_json_mode(mc.client, cap_messages, _CapProbe, timeout),
                        timeout=timeout)
                    return combo_key, time.monotonic() - t0, "capable", "json-mode"
                except asyncio.TimeoutError:
                    return combo_key, time.monotonic() - t0, "timeout", None
                except Exception as e2:
                    if _kind(e2) == "transient":
                        return combo_key, time.monotonic() - t0, "transient", str(e2)[:100]
                    # Responded but cannot structure via EITHER path → INCAPABLE.
                    return combo_key, time.monotonic() - t0, "incapable", str(e2)[:100]

        logger.info("🔬 Capability-probing %d (provider, model) combos (≤%.0fs)...",
                    len(reps), timeout)
        results = await asyncio.gather(*[_probe(k, mc) for k, mc in reps.items()])

        dead_combos: set = set()
        incapable_combos: set = set()
        for combo_key, elapsed, status, detail in results:
            provider, base, pipeline = combo_key
            if status == "capable":
                logger.info("✅ Capable: %s/%s (%s) — %.1fs%s", provider, base, pipeline,
                            elapsed, " [json-mode]" if detail == "json-mode" else "")
                for mc in self._text_pipeline + self._vision_pipeline:
                    if _combo_of(mc) == combo_key:
                        self.health.seed_latency(mc.name, elapsed)
            elif status == "dead":
                dead_combos.add(combo_key)
                logger.warning("💀 Prune DEAD: %s/%s (%s) — %s", provider, base, pipeline, detail)
            elif status == "incapable":
                incapable_combos.add(combo_key)
                logger.warning("🚫 Prune INCAPABLE (can't structure): %s/%s (%s) — %s",
                               provider, base, pipeline, detail)
            elif status == "timeout":
                logger.warning("🐢 Slow: %s/%s (%s) — timed out at %.0fs, kept (high-latency)",
                               provider, base, pipeline, timeout)
                for mc in self._text_pipeline + self._vision_pipeline:
                    if _combo_of(mc) == combo_key:
                        self.health.seed_latency(mc.name, timeout)
            else:  # transient
                logger.info("ℹ️ Transient (kept): %s/%s (%s) — %s", provider, base, pipeline, detail)

        if dead_combos or incapable_combos:
            bt, bv = len(self._text_pipeline), len(self._vision_pipeline)
            self._text_pipeline = self._apply_capability_gate(
                self._text_pipeline, dead_combos, incapable_combos, "TEXT")
            self._vision_pipeline = self._apply_capability_gate(
                self._vision_pipeline, dead_combos, incapable_combos, "VISION")
            logger.info("🔬 Capability gate — %d dead, %d incapable | TEXT %d→%d, VISION %d→%d",
                        len(dead_combos), len(incapable_combos),
                        bt, len(self._text_pipeline), bv, len(self._vision_pipeline))
        else:
            logger.info("🔬 Capability probe complete — all %d combos capable", len(reps))


# ═══════════════════════════════════════════════════════════════════════════════
#  Failover Invocation — V17.0 model-first ordering
# ═══════════════════════════════════════════════════════════════════════════════

def _as_model_clients(chain: list) -> list[ModelClient]:
    """Normalize chain entries to ModelClient (accepts raw LLM clients too)."""
    entries: list[ModelClient] = []
    for item in chain:
        if isinstance(item, ModelClient):
            entries.append(item)
        else:
            name = getattr(item, "model_name", getattr(item, "model", str(item)))
            entries.append(ModelClient(name=str(name), client=item,
                                       provider="unknown", pipeline="text"))
    return entries


def order_failover_chain(
    entries: list[ModelClient],
    health: ProviderHealthTracker | None,
) -> list[ModelClient]:
    """Sort by (model quality tier, expected cost).

    All instances of the best model group at the front — every key, every
    provider hosting it — so failover rotates keys/providers for the SAME
    model before ever falling to a weaker one. Within a tier, instances are
    ordered by E[time-to-answer]; sort stability preserves build order
    (Groq keys before NVIDIA) for fresh instances.
    """
    def sort_key(mc: ModelClient):
        tier = get_model_tier(mc.name)
        cost = health.expected_cost(mc.name) if health else 0.0
        return (tier, cost)

    return sorted(entries, key=sort_key)


def _extract_text(raw_response: Any) -> str:
    """Normalize provider response objects into plain text."""
    content = getattr(raw_response, "content", raw_response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _extract_json_payload(text: str) -> dict:
    """Extract the first JSON object from model output (handles ``` fences)."""
    candidate = text.strip()
    if "```json" in candidate:
        candidate = candidate.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in candidate:
        candidate = candidate.split("```", 1)[1].split("```", 1)[0]
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(candidate[start:end + 1])


async def _invoke_json_mode(llm, messages: list, schema: type, timeout: float) -> Any:
    """Plain JSON-mode fallback for providers that reject the strict schema.

    Appends the JSON Schema as an instruction, parses the raw reply, and
    validates it into the Pydantic model — so a top-tier model that fails
    structured output still serves instead of being benched.
    """
    from langchain_core.messages import HumanMessage

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    instruction = HumanMessage(content=(
        "Respond ONLY with a single JSON object that matches this JSON Schema. "
        "No markdown fences, no commentary, no extra keys.\n"
        f"SCHEMA: {schema_json}"
    ))
    raw = await asyncio.wait_for(
        llm.ainvoke(list(messages) + [instruction]), timeout=timeout,
    )
    payload = _extract_json_payload(_extract_text(raw))
    return schema.model_validate(payload)


async def invoke_with_failover(
    chain: list,
    messages: list,
    schema: type | None = None,
    *,
    breaker: CircuitBreaker | None = None,
    health: ProviderHealthTracker | None = None,
    timeout_seconds: float = 20.0,
    health_tracker: "ProviderHealthTracker | None" = None,
    base64_image: str | None = None,
) -> tuple[Any, str]:
    """V17.0 — Model-first failover with adaptive timeouts.

    Ordering: (model quality tier, E[cost]) — ALL instances of the best model
    (across every API key and provider) are exhausted before any weaker model.

    Per-instance behavior:
      - 429 → jittered cooldown on THAT instance only; the very next attempt
        is the SAME model on the next key/provider (no provider-wide skip,
        no sleep — Groq limits are per key/org).
      - Timeout → adaptive: clamp(3·latency_ewma, 6s, 20s); unknown models
        get the full cap. Failures update the EWMA so slow models sink.
      - Schema 400 → blacklist the (provider, model, schema) combo AND rescue
        the call immediately via JSON-mode on the same model.
      - Blacklisted combos are not skipped — they run in JSON mode, so a
        top-tier model never gets benched over a schema quirk.
    """
    # Accept either kwarg name for backward compatibility
    _health = health or health_tracker

    if breaker and breaker.tripped:
        raise RuntimeError(f"Circuit breaker tripped: {breaker.reason}")

    entries = _as_model_clients(chain)
    ordered = order_failover_chain(entries, _health)

    # Pass 1: instances not on 429 cooldown. Pass 2 (only if pass 1 fails):
    # cooled-down instances as a last resort — slow success beats none.
    if _health:
        hot = [mc for mc in ordered if not _health.in_cooldown(mc.name)]
        cooled = [mc for mc in ordered if _health.in_cooldown(mc.name)]
    else:
        hot, cooled = ordered, []
    passes = [p for p in (hot, cooled) if p]

    schema_name = getattr(schema, "__name__", "__structured_output__") if schema is not None else ""
    last_error: str = ""
    attempt_no = 0
    total = len(entries)

    for pass_entries in passes:
        for mc in pass_entries:
            name, provider, llm = mc.name, mc.provider, mc.client
            attempt_no += 1

            timeout = (
                _health.adaptive_timeout(name) if _health else timeout_seconds
            )

            # Vision payload injection for multimodal-capable models
            current_messages = messages
            if base64_image and messages:
                supports_vision = any(x in name.lower() for x in [
                    "gpt-4o", "claude-3-5", "gemini", "glm-4v", "vision", "pixtral", "gemma",
                ])
                from langchain_core.messages import HumanMessage
                last_msg = messages[-1]
                if supports_vision and isinstance(last_msg, HumanMessage):
                    multimodal_content = [
                        {"type": "text", "text": str(last_msg.content)},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ]
                    current_messages = messages[:-1] + [HumanMessage(content=multimodal_content)]

            # Blacklisted (provider, model, schema) combos go straight to JSON mode
            use_json_mode = bool(
                schema is not None and _health
                and _health.is_schema_blacklisted(name, schema_name)
            )

            t0 = time.monotonic()
            try:
                if schema is None:
                    response = await asyncio.wait_for(
                        llm.ainvoke(current_messages), timeout=timeout)
                elif use_json_mode:
                    response = await _invoke_json_mode(
                        llm, current_messages, schema, timeout)
                else:
                    structured = llm.with_structured_output(schema)
                    response = await asyncio.wait_for(
                        structured.ainvoke(current_messages), timeout=timeout)

                elapsed = time.monotonic() - t0
                if _health:
                    _health.record_success(name, latency=elapsed)
                if breaker:
                    breaker.record_success()
                return response, name

            except asyncio.TimeoutError:
                last_error = f"{name} timed out (>{timeout:.0f}s)"
                if _health:
                    _health.record_failure(name, latency=timeout)
                if breaker:
                    breaker.record_failure()
                logger.warning("⚠️  FAILOVER [%d/%d]: %s TIMED OUT (>%.0fs adaptive)",
                               attempt_no, total, name, timeout)

            except Exception as exc:
                err = str(exc)
                last_error = f"{name} — {err[:150]}"
                err_l = err.lower()
                is_rl = "429" in err or "rate_limit" in err_l or "rate limit" in err_l
                is_schema_err = (
                    ("400" in err or "bad request" in err_l)
                    and ("additionalproperties" in err_l or "schema" in err_l
                         or "response_format" in err_l)
                )

                if is_rl:
                    # Per-instance cooldown — the next loop iteration is the
                    # SAME model on the next key/provider. No sleep needed.
                    if _health:
                        _health.start_cooldown(name, 20.0 + random.uniform(0, 10.0))
                    logger.warning("⚠️  FAILOVER [%d/%d]: %s rate-limited → same model, next key",
                                   attempt_no, total, name)
                    continue

                if breaker:
                    breaker.record_failure()

                if is_schema_err and schema is not None:
                    # Register blacklist, then rescue THIS model via JSON mode
                    if _health:
                        _health.record_failure(name, error_msg=err)
                    logger.warning("⚠️  FAILOVER [%d/%d]: %s schema 400 → JSON-mode rescue",
                                   attempt_no, total, name)
                    try:
                        t1 = time.monotonic()
                        response = await _invoke_json_mode(
                            llm, current_messages, schema, timeout)
                        if _health:
                            _health.record_success(name, latency=time.monotonic() - t1)
                        if breaker:
                            breaker.record_success()
                        return response, name
                    except Exception as json_exc:  # noqa: BLE001
                        last_error = f"{name} JSON-mode — {str(json_exc)[:120]}"
                        logger.warning("⚠️  JSON-mode rescue failed for %s: %s",
                                       name, str(json_exc)[:150])
                        continue

                if _health:
                    _health.record_failure(name)
                logger.warning("⚠️  FAILOVER [%d/%d]: %s — %s",
                               attempt_no, total, name, err[:200])

    raise RuntimeError(f"ALL {total} models failed. Last: {last_error}")
