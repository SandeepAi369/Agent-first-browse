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
    """Per-model exponential backoff quarantine."""

    def __init__(self):
        self._health: dict[str, dict] = {}

    def _get(self, name: str) -> dict:
        if name not in self._health:
            self._health[name] = {
                "consecutive_failures": 0,
                "quarantine_until": 0.0,
                "total_calls": 0,
                "total_failures": 0,
            }
        return self._health[name]

    def is_available(self, name: str) -> bool:
        return time.monotonic() >= self._get(name)["quarantine_until"]

    def record_success(self, name: str) -> None:
        s = self._get(name)
        s["consecutive_failures"] = 0
        s["quarantine_until"] = 0.0
        s["total_calls"] += 1

    def record_failure(self, name: str, is_rate_limit: bool = False) -> None:
        s = self._get(name)
        s["consecutive_failures"] += 1
        s["total_calls"] += 1
        s["total_failures"] += 1
        backoff = min(120.0, 2.0 ** s["consecutive_failures"])
        if is_rate_limit:
            backoff = min(120.0, backoff * 2)
        s["quarantine_until"] = time.monotonic() + backoff
        logger.warning(
            "🔒 '%s' quarantined %.1fs (streak=%d)", name, backoff, s["consecutive_failures"],
        )

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
    """TEXT-ONLY pipeline for Manager & Helper. NEVER receives images."""
    from langchain_openai import ChatOpenAI

    clients: list[ModelClient] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    }

    # ── Groq — PRIMARY (all 3 keys from .env) ──
    groq_keys = _collect_keys(
        "GROQ_API_KEY",
        "GROQ_API_KEYS",
        "OPENAI_API_KEY_FALLBACKS",       # gsk_QTj79y... is a Groq key here
        "SUPERVISOR_MODEL_API_KEY_FALLBACKS",  # gsk_ZQTnjT..., gsk_QTj79y...
    )
    # Filter: only keys starting with "gsk_" are Groq keys
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

    # ── NVIDIA NIM — SECONDARY ──
    nvidia_keys = _collect_keys("NVIDIA_NIM_API_KEY", "NVIDIA_NIM_API_KEYS")
    model = os.getenv("NVIDIA_NIM_MODEL", "z-ai/glm-5.1")
    base = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    for idx, key in enumerate(nvidia_keys):
        clients.append(ModelClient(
            name=f"nvidia-text:{model}:{idx}",
            client=ChatOpenAI(
                model=model, api_key=key,
                base_url=base, temperature=0.0, timeout=25,
                default_headers=headers,
            ),
            provider="nvidia", pipeline="text",
        ))
    if nvidia_keys:
        logger.info("TEXT ── NVIDIA NIM (SECONDARY): %d keys loaded (%s)", len(nvidia_keys), model)

    # ── Cerebras — TERTIARY (fast inference, generous rate limits) ──
    cerebras_keys = _collect_keys("CEREBRAS_API_KEY", "CEREBRAS_API_KEYS")
    if cerebras_keys:
        cerebras_model = os.getenv("CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507")
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

    # ── Google Gemini — QUATERNARY TEXT FALLBACK (always available) ──
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

    # ── NVIDIA NIM Vision — SECONDARY (Phi-4 + Llama-3.2-90B) ──
    nvidia_vision_key = os.getenv("NVIDIA_VISION_API_KEY", "").strip()
    if nvidia_vision_key:
        from langchain_openai import ChatOpenAI
        base = os.getenv("NVIDIA_VISION_BASE_URL", "https://integrate.api.nvidia.com/v1")

        primary = os.getenv("NVIDIA_VISION_MODEL", "microsoft/phi-4-multimodal-instruct")
        clients.append(ModelClient(
            name=f"nvidia-vision:{primary}:0",
            client=ChatOpenAI(
                model=primary, api_key=nvidia_vision_key,
                base_url=base, temperature=0.0, timeout=60,
            ),
            provider="nvidia", pipeline="vision",
        ))
        logger.info("VISION ── NVIDIA NIM (SECONDARY): %s loaded", primary)

        fallback = os.getenv("NVIDIA_VISION_FALLBACK_MODEL", "meta/llama-3.2-90b-vision-instruct")
        clients.append(ModelClient(
            name=f"nvidia-vision:{fallback}:0",
            client=ChatOpenAI(
                model=fallback, api_key=nvidia_vision_key,
                base_url=base, temperature=0.0, timeout=60,
            ),
            provider="nvidia", pipeline="vision",
        ))
        logger.info("VISION ── NVIDIA NIM (TERTIARY): %s loaded", fallback)

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
        self.breaker = CircuitBreaker(max_requests=80, max_failures=15)

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
) -> tuple[Any, str]:
    """Try each model in the chain until one succeeds."""
    import asyncio

    if breaker and breaker.tripped:
        raise RuntimeError(f"Circuit breaker tripped: {breaker.reason}")

    last_error: str = ""

    for idx, mc in enumerate(chain):
        name = mc.name

        if health and not health.is_available(name):
            if not health.all_quarantined([m.name for m in chain]):
                continue

        try:
            llm = mc.client
            if schema is not None:
                llm = llm.with_structured_output(schema)

            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout_seconds)

            if health:
                health.record_success(name)
            if breaker:
                breaker.record_success()
            return response, name

        except asyncio.TimeoutError:
            last_error = f"{name} TIMEOUT (>{timeout_seconds}s)"
            if health:
                health.record_failure(name)
            if breaker:
                breaker.record_failure()
            logger.warning("⚠️ FAILOVER [%d/%d]: %s", idx + 1, len(chain), last_error)

        except Exception as exc:
            err = str(exc)
            last_error = f"{name}: {err[:150]}"
            is_rl = "429" in err or "rate" in err.lower()
            if health:
                health.record_failure(name, is_rate_limit=is_rl)
            if breaker:
                breaker.record_failure()
            logger.warning("⚠️ FAILOVER [%d/%d]: %s", idx + 1, len(chain), last_error)
            if is_rl:
                await asyncio.sleep(1.5)

    raise RuntimeError(f"ALL {len(chain)} models failed. Last: {last_error}")
