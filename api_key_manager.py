"""Smart API Key Manager — Universal Provider Routing Engine (v1.0).

Inspired by OpenRouter's auto-routing and CrewAI's LLM management:
  - Accepts ANY provider's API keys from .env
  - Tracks per-key health (success rate, latency, rate limit status)
  - Auto-rotates keys within a provider when one hits rate limits
  - Quarantines exhausted keys with exponential backoff
  - Provides a unified interface: get_best_client(capability) -> LLM
  - Self-healing: keys automatically return to rotation after quarantine expires

Usage:
    manager = APIKeyManager()
    client = manager.get_best_client("text")  # or "vision"
    manager.record_success("provider:model", latency_ms=320)
    manager.record_failure("provider:model", error_code=429)
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.logger import get_logger

logger = get_logger("api_key_manager")


# ═══════════════════════════════════════════════════════════════════════════════
#  Provider Registry — All supported LLM providers
# ═══════════════════════════════════════════════════════════════════════════════

class Capability(str, Enum):
    TEXT = "text"
    VISION = "vision"
    STRUCTURED = "structured"


@dataclass
class ProviderConfig:
    """Static configuration for an LLM provider."""
    name: str
    base_url: str
    model: str
    capabilities: set[Capability]
    timeout: int = 20
    priority: int = 50  # Lower = higher priority (0 = top)
    env_key: str = ""  # Primary env var name
    env_keys_multi: str = ""  # Comma-separated multi-key env var
    constructor: str = "openai"  # "openai" or "google"


# Master registry — add new providers here, they auto-activate if keys exist
PROVIDER_REGISTRY: list[ProviderConfig] = [
    ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        capabilities={Capability.TEXT, Capability.STRUCTURED},
        timeout=25,
        priority=10,
        env_key="GROQ_API_KEY",
        env_keys_multi="GROQ_API_KEYS",
    ),
    ProviderConfig(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        model="z-ai/glm-5.1",
        capabilities={Capability.TEXT, Capability.STRUCTURED},
        timeout=15,
        priority=20,
        env_key="NVIDIA_NIM_API_KEY",
        env_keys_multi="NVIDIA_NIM_API_KEYS",
    ),
    ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-pro",
        capabilities={Capability.TEXT, Capability.STRUCTURED},
        timeout=30,
        priority=30,
        env_key="DEEPSEEK_API_KEY",
        env_keys_multi="DEEPSEEK_API_KEYS",
    ),
    ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini",
        capabilities={Capability.TEXT, Capability.VISION, Capability.STRUCTURED},
        timeout=25,
        priority=40,
        env_key="OPENAI_API_KEY",
        env_keys_multi="OPENAI_API_KEY_FALLBACKS",
    ),
    ProviderConfig(
        name="google",
        base_url="",  # Google uses its own SDK
        model="gemma-4-31b-it",
        capabilities={Capability.TEXT, Capability.VISION, Capability.STRUCTURED},
        timeout=30,
        priority=50,
        env_key="GOOGLE_API_KEY",
        env_keys_multi="GOOGLE_API_KEY_FALLBACKS",
        constructor="google",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-Key Health Tracker
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KeyHealth:
    """Runtime health state for a single API key."""
    provider_name: str
    key_index: int
    total_calls: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    last_failure_at: float = 0.0
    quarantine_until: float = 0.0
    last_error_code: int = 0
    _latency_samples: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return (self.total_calls - self.total_failures) / self.total_calls

    @property
    def is_quarantined(self) -> bool:
        return time.monotonic() < self.quarantine_until

    @property
    def quarantine_remaining(self) -> float:
        return max(0.0, self.quarantine_until - time.monotonic())

    def record_success(self, latency_ms: float) -> None:
        self.total_calls += 1
        self.consecutive_failures = 0
        self._latency_samples.append(latency_ms)
        # Rolling average over last 10 samples
        if len(self._latency_samples) > 10:
            self._latency_samples = self._latency_samples[-10:]
        self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def record_failure(self, error_code: int = 0) -> None:
        self.total_calls += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure_at = time.monotonic()
        self.last_error_code = error_code
        # Exponential backoff quarantine: 2^failures seconds, capped at 120s
        backoff = min(120.0, 2.0 ** self.consecutive_failures)
        if error_code == 429:
            backoff = min(120.0, backoff * 2)  # Double for rate limits
        self.quarantine_until = time.monotonic() + backoff
        logger.warning(
            "🔒 Key %s[%d] quarantined for %.1fs (failures=%d, code=%d)",
            self.provider_name, self.key_index, backoff,
            self.consecutive_failures, error_code,
        )

    @property
    def score(self) -> float:
        """Composite score: lower = better. Used for sorting."""
        if self.is_quarantined:
            return 99999.0
        latency_factor = self.avg_latency_ms / 1000.0  # Normalize to seconds
        failure_penalty = self.consecutive_failures * 5.0
        return latency_factor + failure_penalty


# ═══════════════════════════════════════════════════════════════════════════════
#  Smart API Key Manager
# ═══════════════════════════════════════════════════════════════════════════════

SPOOFED_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    )
}


class APIKeyManager:
    """Universal LLM provider manager with health-aware routing.

    Automatically discovers API keys from environment, builds LLM clients,
    tracks per-key health, and routes requests to the best available provider.
    """

    _instance: APIKeyManager | None = None

    def __init__(self) -> None:
        self._clients: list[tuple[Any, ProviderConfig, KeyHealth]] = []
        self._build_all_clients()

    @classmethod
    def get(cls) -> APIKeyManager:
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _collect_keys(self, config: ProviderConfig) -> list[str]:
        """Collect all API keys for a provider from environment."""
        keys: list[str] = []
        # Primary key
        primary = os.getenv(config.env_key, "").strip()
        if primary:
            keys.append(primary)
        # Multi-key (comma/semicolon separated)
        if config.env_keys_multi:
            raw = os.getenv(config.env_keys_multi, "")
            for k in raw.replace(";", ",").split(","):
                k = k.strip()
                if k and k not in keys:
                    keys.append(k)
        return keys

    def _build_all_clients(self) -> None:
        """Discover keys and build LLM clients for all registered providers."""
        for provider_cfg in PROVIDER_REGISTRY:
            keys = self._collect_keys(provider_cfg)
            if not keys:
                continue

            for idx, api_key in enumerate(keys):
                try:
                    client = self._create_client(provider_cfg, api_key)
                    health = KeyHealth(provider_name=provider_cfg.name, key_index=idx)
                    self._clients.append((client, provider_cfg, health))
                except Exception as exc:
                    logger.warning(
                        "Failed to build %s client[%d]: %s", provider_cfg.name, idx, exc
                    )

            logger.info("✅ %s: %d key(s) loaded", provider_cfg.name, len(keys))

        total = len(self._clients)
        providers = len({cfg.name for _, cfg, _ in self._clients})
        logger.info(
            "APIKeyManager ready: %d client(s) across %d provider(s)", total, providers
        )

    def _create_client(self, cfg: ProviderConfig, api_key: str) -> Any:
        """Create an LLM client for the given provider config and key."""
        if cfg.constructor == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=cfg.model,
                google_api_key=api_key,
                temperature=0.0,
                timeout=cfg.timeout,
            )
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=cfg.model,
                api_key=api_key,
                base_url=cfg.base_url,
                temperature=0.0,
                timeout=cfg.timeout,
                default_headers=SPOOFED_HEADERS,
            )

    # ── Routing ──────────────────────────────────────────────────────────────

    def get_failover_chain(
        self, capability: Capability = Capability.TEXT
    ) -> list[tuple[Any, ProviderConfig, KeyHealth]]:
        """Return all clients sorted by health score, filtered by capability.

        Quarantined keys are placed at the end (they'll only be tried if
        all healthy keys fail).
        """
        eligible = [
            (client, cfg, health)
            for client, cfg, health in self._clients
            if capability in cfg.capabilities
        ]
        # Sort: healthy keys by (priority + score), quarantined keys last
        eligible.sort(key=lambda x: (x[2].is_quarantined, x[1].priority + x[2].score))
        return eligible

    def get_best_client(
        self, capability: Capability = Capability.TEXT
    ) -> tuple[Any, str] | None:
        """Return the single best available client.

        Returns: (client, "provider:model") or None
        """
        chain = self.get_failover_chain(capability)
        for client, cfg, health in chain:
            if not health.is_quarantined:
                return client, f"{cfg.name}:{cfg.model}"
        # All quarantined — return the one with shortest remaining quarantine
        if chain:
            return chain[0][0], f"{chain[0][1].name}:{chain[0][1].model}"
        return None

    def get_chain_names(self) -> list[str]:
        """Return human-readable names of all clients in priority order."""
        chain = self.get_failover_chain()
        return [f"{cfg.name}/{cfg.model}" for _, cfg, _ in chain]

    # ── Health Recording ─────────────────────────────────────────────────────

    def _find_health(self, client: Any) -> KeyHealth | None:
        for c, _, h in self._clients:
            if c is client:
                return h
        return None

    def record_success(self, client: Any, latency_ms: float = 0.0) -> None:
        health = self._find_health(client)
        if health:
            health.record_success(latency_ms)

    def record_failure(self, client: Any, error_code: int = 0) -> None:
        health = self._find_health(client)
        if health:
            health.record_failure(error_code)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def health_report(self) -> str:
        """Return a human-readable health report for all keys."""
        lines = ["═══ API Key Health Report ═══"]
        for _, cfg, health in self._clients:
            status = "🔒 QUARANTINED" if health.is_quarantined else "✅ HEALTHY"
            lines.append(
                f"  {cfg.name}[{health.key_index}] {status} "
                f"calls={health.total_calls} "
                f"success={health.success_rate:.0%} "
                f"latency={health.avg_latency_ms:.0f}ms "
                f"streak_fail={health.consecutive_failures}"
            )
        return "\n".join(lines)

    @property
    def total_clients(self) -> int:
        return len(self._clients)

    @property
    def healthy_clients(self) -> int:
        return sum(1 for _, _, h in self._clients if not h.is_quarantined)

    async def wait_for_any_available(self, timeout: float = 60.0) -> bool:
        """Wait until at least one key exits quarantine. Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.healthy_clients > 0:
                return True
            # Find shortest remaining quarantine
            min_wait = min(
                (h.quarantine_remaining for _, _, h in self._clients if h.is_quarantined),
                default=1.0,
            )
            wait_time = min(min_wait + 0.1, deadline - time.monotonic())
            if wait_time <= 0:
                break
            logger.info("⏳ All keys quarantined. Waiting %.1fs for recovery...", wait_time)
            await asyncio.sleep(wait_time)
        return self.healthy_clients > 0
