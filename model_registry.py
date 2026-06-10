"""ModelRegistry — Model-Agnostic Provider Layer for Agent First IDE v2.0.

Enforces STRICT separation between Text and Vision API pipelines.
No graph node should ever construct an LLM client directly.
All model access goes through this registry.

Reads ALL API keys from .env — NVIDIA, Groq, Google Gemini, DeepSeek.
"""

from __future__ import annotations

import asyncio
import os
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

    def __init__(self):
        self._health: dict[str, dict] = {}
        self._schema_blacklist: dict[str, set[str]] = {}  # base_model → {schema_names}

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

    def _get(self, name: str) -> dict:
        if name not in self._health:
            self._health[name] = {
                "consecutive_failures": 0,
                "quarantine_until": 0.0,
                "total_calls": 0,
                "total_failures": 0,
                "last_call_time": 0.0,
            }
        return self._health[name]

    def is_available(self, name: str, schema_name: str = "") -> bool:
        # V15.1: Check schema blacklist by BASE model name
        if schema_name:
            base = self._base_model_name(name)
            if base in self._schema_blacklist:
                bl = self._schema_blacklist[base]
                if schema_name in bl or "__structured_output__" in bl:
                    return False

        s = self._get(name)
        # V15.0: Cold-start reset — if model idle 5+ min, treat as fresh
        if s["last_call_time"] > 0 and time.monotonic() - s["last_call_time"] > 300:
            s["consecutive_failures"] = 0
            s["quarantine_until"] = 0.0

        return time.monotonic() >= s["quarantine_until"]

    def record_success(self, name: str) -> None:
        s = self._get(name)
        s["consecutive_failures"] = 0
        s["quarantine_until"] = 0.0
        s["total_calls"] += 1
        s["last_call_time"] = time.monotonic()

    def record_failure(self, name: str, is_rate_limit: bool = False, error_msg: str = "") -> None:
        s = self._get(name)
        s["consecutive_failures"] += 1
        s["total_calls"] += 1
        s["total_failures"] += 1
        s["last_call_time"] = time.monotonic()

        # V15.0: Cap quarantine at 16s (was min(120, 2^n))
        backoff = min(16.0, 2.0 ** s["consecutive_failures"])
        if is_rate_limit:
            backoff = min(16.0, backoff * 2)
        s["quarantine_until"] = time.monotonic() + backoff
        logger.warning(
            "🔒 '%s' quarantined %.1fs (streak=%d)", name, backoff, s["consecutive_failures"],
        )

        # V15.1 Patch A+C: Schema blacklist by BASE model name + per-schema keys
        if error_msg and ("400" in error_msg or "Bad Request" in error_msg):
            if "additionalProperties" in error_msg or "schema" in error_msg.lower():
                base = self._base_model_name(name)
                if base not in self._schema_blacklist:
                    self._schema_blacklist[base] = set()

                # Patch C: Extract specific schema name from error message
                # Error format: "response_format: 'ChecklistEvaluation': /properties/..."
                # We target the schema name after 'response_format:' specifically
                import re
                schema_match = re.search(r"response_format:\s*'(\w+)'", error_msg)
                if not schema_match:
                    # Fallback: look for CamelCase class names (Pydantic schema pattern)
                    schema_match = re.search(r"'([A-Z]\w{5,})'", error_msg)
                specific_schema = schema_match.group(1) if schema_match else None

                if specific_schema:
                    self._schema_blacklist[base].add(specific_schema)
                    logger.warning(
                        "🚫 Base model '%s' blacklisted for schema '%s' "
                        "(all %s instances will be skipped)",
                        base, specific_schema, name.split(":")[0],
                    )
                else:
                    # Catch-all: blacklist for all structured output
                    self._schema_blacklist[base].add("__structured_output__")
                    logger.warning(
                        "🚫 Base model '%s' blacklisted for ALL structured output",
                        base,
                    )

    def is_schema_blacklisted(self, name: str, schema_name: str = "__structured_output__") -> bool:
        """Check if a model is permanently blacklisted for a specific schema.

        V15.1: Uses base model name so all key-variants are covered.
        """
        base = self._base_model_name(name)
        if base not in self._schema_blacklist:
            return False
        bl = self._schema_blacklist[base]
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
                    self._probes_done = 0
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
    # V16.3: The user specifically requested openai/gpt-oss-120b as the ONLY text model
    # for NVIDIA, completely replacing the nemotron, qwen, and mistral models.
    nvidia_keys = _collect_keys("NVIDIA_NIM_API_KEY", "NVIDIA_NIM_API_KEYS")
    base = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nvidia_text_models = [
        ("openai/gpt-oss-120b", 30),
    ]
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
        cerebras_model = os.getenv("CEREBRAS_MODEL", "qwen-3-235b-a22b")
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

        nvidia_vision_models = [
            ("google/gemma-4-32b-it", 30),                    # User's preferred — fast, excellent vision
            ("meta/llama-4-maverick-17b-128e-instruct", 35),  # 1M context, strong native vision
            ("microsoft/phi-4-multimodal-instruct", 45),      # Lightweight 5.6B, text+image+audio
            ("meta/llama-3.2-90b-vision-instruct", 50),       # Strong 90B vision fallback
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
#  ModelRegistry Singleton
# ═══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """Centralized model registry. Strict TEXT vs VISION separation."""

    _instance: "ModelRegistry | None" = None

    def __init__(self):
        self._text_pipeline: list[ModelClient] = _build_text_pipeline()
        self._vision_pipeline: list[ModelClient] = _build_vision_pipeline()
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
            "ModelRegistry — TEXT: %d models, VISION: %d models",
            len(self._text_pipeline), len(self._vision_pipeline),
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

    def summary(self) -> str:
        t = " → ".join(self.get_text_chain_names()) or "(none)"
        v = " → ".join(self.get_vision_chain_names()) or "(none)"
        return f"TEXT chain ({len(self._text_pipeline)}): {t}\nVISION chain ({len(self._vision_pipeline)}): {v}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Failover Invocation
# ═══════════════════════════════════════════════════════════════════════════════

async def invoke_with_failover(
    chain: list[ModelClient],
    messages: list,
    schema: type | None = None,
    *,
    breaker: CircuitBreaker | None = None,
    health: ProviderHealthTracker | None = None,
    timeout_seconds: float = 30.0,
    health_tracker: "ProviderHealthTracker | None" = None,
) -> tuple[Any, str]:
    """V16.1 — Try each model in sequence. No per-model quarantine.

    Old logic quarantined individual models for 2-16s on failure. This was
    catastrophic when multiple models shared one API key (e.g., NVIDIA NIM)
    because quarantining model A also effectively blocked models B-E on the
    same key — the quarantine timer would expire and immediately re-trigger.

    New logic:
      1. Walk the chain sequentially, try every model.
      2. Only skip models that are schema-blacklisted (400/Bad Request on
         structured output — a permanent, not transient, issue).
      3. On rate-limit (429), wait 2s then continue to next model.
      4. No exponential backoff per model. Just try the next one.
    """
    import asyncio

    # Accept either kwarg name for backward compatibility
    _health = health or health_tracker

    if breaker and breaker.tripped:
        raise RuntimeError(f"Circuit breaker tripped: {breaker.reason}")

    last_error: str = ""
    attempts = 0

    for idx, mc in enumerate(chain):
        name = mc.name

        # Only skip: schema blacklist (permanent 400 errors on structured output)
        if _health and schema is not None:
            schema_name = schema.__name__ if hasattr(schema, '__name__') else "__structured_output__"
            if _health.is_schema_blacklisted(name, schema_name):
                logger.debug("Skipping '%s' — blacklisted for schema '%s'", name, schema_name)
                continue

        attempts += 1
        try:
            llm = mc.client
            if schema is not None:
                llm = llm.with_structured_output(schema)

            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout_seconds)

            if _health:
                _health.record_success(name)
            if breaker:
                breaker.record_success()
            return response, name

        except asyncio.TimeoutError:
            last_error = f"{name} timed out — waiting 5.0s"
            if breaker:
                breaker.record_failure()
            logger.warning("⚠️  FAILOVER [%d/%d]: %s TIMED OUT (>%ds)", idx + 1, len(chain), name, timeout_seconds)

        except Exception as exc:
            err = str(exc)
            last_error = f"{name} — {err[:150]}"
            is_rl = "429" in err or "rate" in err.lower()
            is_schema_err = "400" in err and ("additionalProperties" in err or "schema" in err.lower())

            # Only record schema blacklist (permanent issue)
            if _health and is_schema_err:
                _health.record_failure(name, is_rate_limit=False, error_msg=err)

            if breaker:
                breaker.record_failure()
            logger.warning("⚠️  FAILOVER [%d/%d]: %s — %s", idx + 1, len(chain), name, err[:200])

            # On rate limit, short 2s pause then move to next model immediately
            if is_rl:
                await asyncio.sleep(2.0)

    raise RuntimeError(f"ALL {len(chain)} models failed. Last: {last_error}")
