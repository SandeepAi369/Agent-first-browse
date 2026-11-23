"""Typed, eagerly-validated environment settings decoupled from legacy promoter config.

Consolidates runtime configuration across the browser engine, URL/upload safety,
zero-API eval connector, LLM providers, observability, and feature flags into
immutable, stdlib-only dataclasses:

  - Pure & side-effect-free `load_settings(env=None)` reads either an injected
    mapping or live `os.environ` (never caching stale values across test
    monkeypatches or silently mutating `os.environ`).
  - All API keys and tokens are wrapped in `SecretValue`, guaranteeing zero
    raw secret leaks in `repr()`, `str()`, logs, or `to_safe_dict()`.
  - Invalid numeric/boolean/enum/range env values fail eagerly with
    `ConfigValidationError` instead of failing deep inside browser loops.
  - Strictly excludes legacy `app.browser_promoter` campaign settings.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from core.errors import ConfigValidationError
from core.retries import (
    DEFAULT_RETRY_CONFIG,
    ResolvedRetryConfig,
    RetryOptions,
    resolve_retry_config,
)
from core.runtime_context import VALID_RUNTIME_MODES
from core.url_policy import DEFAULT_MAX_URL_LENGTH, UrlPolicy

_TRUE_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS: frozenset[str] = frozenset({"0", "false", "no", "off"})


class SecretValue:
    """Opaque secret wrapper that never reveals its raw value in `str()` or `repr()`."""

    __slots__ = ("_value",)

    def __init__(self, value: str = "") -> None:
        if not isinstance(value, str):
            raise ConfigValidationError(f"SecretValue requires a str, got {type(value).__name__}")
        self._value = value.strip()

    def get_secret_value(self) -> str:
        """Explicitly unwrap the raw secret string at the exact provider call site."""
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __repr__(self) -> str:
        return "SecretValue('***')" if self._value else "SecretValue('')"

    def __str__(self) -> str:
        return "***" if self._value else ""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretValue):
            return self._value == other._value
        return False

    def __hash__(self) -> int:
        return hash(self._value)


def parse_env_bool(
    env: Mapping[str, str],
    name: str,
    *,
    default: bool,
    strict: bool = True,
) -> bool:
    """Parse a boolean env var; raises `ConfigValidationError` on unrecognized tokens."""
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    token = raw.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    if not strict:
        return True
    raise ConfigValidationError(
        f"Environment variable {name} must be a boolean (true/false/1/0/yes/no/on/off), got {raw!r}"
    )


def parse_env_int(
    env: Mapping[str, str],
    name: str,
    *,
    default: int,
    min_value: int | None = None,
) -> int:
    """Parse an integer env var with optional lower-bound validation."""
    raw = env.get(name)
    if raw is None or not raw.strip():
        val = default
    else:
        try:
            val = int(raw.strip())
        except ValueError as exc:
            raise ConfigValidationError(
                f"Environment variable {name} must be an integer, got {raw!r}"
            ) from exc
    if min_value is not None and val < min_value:
        raise ConfigValidationError(
            f"Environment variable {name} must be >= {min_value}, got {val}"
        )
    return val


def parse_env_float(
    env: Mapping[str, str],
    name: str,
    *,
    default: float,
    min_value: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Parse a finite float env var with optional lower-bound validation."""
    raw = env.get(name)
    if raw is None or not raw.strip():
        val = float(default)
    else:
        try:
            val = float(raw.strip())
        except ValueError as exc:
            raise ConfigValidationError(
                f"Environment variable {name} must be a number, got {raw!r}"
            ) from exc
    if not math.isfinite(val):
        raise ConfigValidationError(f"Environment variable {name} must be finite, got {val!r}")
    if strictly_positive and val <= 0:
        raise ConfigValidationError(f"Environment variable {name} must be > 0, got {val}")
    if min_value is not None and val < min_value:
        raise ConfigValidationError(f"Environment variable {name} must be >= {min_value}, got {val}")
    return val


def parse_env_list(raw: str | None) -> tuple[str, ...]:
    """Split a comma/semicolon-delimited env string into trimmed non-empty items."""
    if not raw or not raw.strip():
        return ()
    normalized = raw.replace(";", ",")
    return tuple(item.strip() for item in normalized.split(",") if item.strip())


def _merge_secret_keys(env: Mapping[str, str], *var_names: str) -> tuple[SecretValue, ...]:
    seen: set[str] = set()
    keys: list[SecretValue] = []
    for name in var_names:
        for item in parse_env_list(env.get(name)):
            if item not in seen:
                seen.add(item)
                keys.append(SecretValue(item))
    return tuple(keys)


def _validate_positive_int_field(val: object, field_name: str) -> None:
    if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
        raise ConfigValidationError(f"{field_name} must be an integer > 0, got {val!r}")


def _validate_engine_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise ConfigValidationError(f"engine_mode must be 'legacy' or 'v2', got {mode!r}")
    norm = mode.strip().lower()
    if norm not in VALID_RUNTIME_MODES:
        raise ConfigValidationError(
            f"AGENT_ENGINE_MODE must be one of {VALID_RUNTIME_MODES}, got {mode!r}"
        )
    return norm


@dataclass(frozen=True)
class EngineSettings:
    """Core browser execution and retry configuration."""

    engine_mode: str = "legacy"
    headless: bool = False
    viewport_width: int = 1280
    viewport_height: int = 800
    local_cdp_endpoint: str = "http://localhost:9222"
    remote_websocket_url: str = ""
    startup_timeout_s: float = 18.0
    retry: ResolvedRetryConfig = DEFAULT_RETRY_CONFIG

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine_mode", _validate_engine_mode(self.engine_mode))
        _validate_positive_int_field(self.viewport_width, "engine.viewport_width")
        _validate_positive_int_field(self.viewport_height, "engine.viewport_height")
        if (
            isinstance(self.startup_timeout_s, bool)
            or not isinstance(self.startup_timeout_s, (int, float))
            or not math.isfinite(float(self.startup_timeout_s))
            or float(self.startup_timeout_s) <= 0
        ):
            raise ConfigValidationError(
                f"engine.startup_timeout_s must be finite > 0, got {self.startup_timeout_s!r}"
            )


@dataclass(frozen=True)
class SafetySettings:
    """Execution-layer URL, SSRF, and file-upload safety policy configuration."""

    allow_private_hosts: bool = False
    allow_about_blank: bool = True
    max_url_length: int = DEFAULT_MAX_URL_LENGTH
    allowed_hosts: tuple[str, ...] = ()
    blocked_hosts: tuple[str, ...] = ()
    upload_dirs: tuple[Path, ...] = ()
    dry_run_mode: bool = False

    def __post_init__(self) -> None:
        _validate_positive_int_field(self.max_url_length, "safety.max_url_length")

    def to_url_policy(self) -> UrlPolicy:
        """Build a `UrlPolicy` instance directly from these safety settings."""
        return UrlPolicy(
            allowed_hosts=self.allowed_hosts,
            blocked_hosts=self.blocked_hosts,
            allow_private_hosts=self.allow_private_hosts,
            allow_about_blank=self.allow_about_blank,
            max_url_length=self.max_url_length,
        )


@dataclass(frozen=True)
class EvalConnectorSettings:
    """Zero-API CLI daemon & MCP server configuration."""

    socket_path: Path = field(default_factory=lambda: Path.home() / ".cache" / "afb" / "eval.sock")
    idle_timeout_s: float = 1800.0
    max_mcp_output_chars: int = 24_000
    screenshot_dir: Path | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.idle_timeout_s, bool)
            or not isinstance(self.idle_timeout_s, (int, float))
            or not math.isfinite(float(self.idle_timeout_s))
            or float(self.idle_timeout_s) < 0
        ):
            raise ConfigValidationError(
                f"eval_connector.idle_timeout_s must be finite >= 0, got {self.idle_timeout_s!r}"
            )
        _validate_positive_int_field(self.max_mcp_output_chars, "eval_connector.max_mcp_output_chars")


@dataclass(frozen=True)
class ModelProviderSettings:
    """LLM provider API keys (masked) and default model selection."""

    openai_api_keys: tuple[SecretValue, ...] = ()
    openai_base_url: str = ""
    google_api_keys: tuple[SecretValue, ...] = ()
    groq_api_keys: tuple[SecretValue, ...] = ()
    nvidia_api_keys: tuple[SecretValue, ...] = ()
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    worker_vlm_model: str = "gemma-4-27b-it"
    supervisor_model: str = "gemma-4-27b-it"

    def configured_providers(self) -> tuple[str, ...]:
        """List providers with at least one non-empty API key configured."""
        active: list[str] = []
        if self.openai_api_keys:
            active.append("openai")
        if self.google_api_keys:
            active.append("google")
        if self.groq_api_keys:
            active.append("groq")
        if self.nvidia_api_keys:
            active.append("nvidia")
        return tuple(active)


@dataclass(frozen=True)
class ObservabilitySettings:
    """Telemetry sink and optional LangSmith tracing settings."""

    langsmith_enabled: bool = False
    langsmith_api_key: SecretValue = field(default_factory=SecretValue)
    langsmith_project: str = "agent-first-browse"
    langsmith_endpoint: str = ""
    jsonl_log_path: Path | None = None


@dataclass(frozen=True)
class FeatureFlagSnapshot:
    """Typed snapshot of live V29 + dual-mode feature switches."""

    v29_enabled: bool = True
    reality: bool = True
    reality_llm: bool = True
    clarity_consensus: bool = True
    target_lock: bool = True
    intent_journal: bool = True
    subgoal_lock: bool = True
    stagnation: bool = True
    smart_scroll: bool = True
    page_context: bool = True
    adaptive_perception: bool = True
    strict_viewport: bool = True
    diffing: bool = True
    hybrid_primitives: bool = True
    webdreamer: bool = True
    webdreamer_situational: bool = True
    lats: bool = False
    skill_memory_v2: bool = False
    v2_engine: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "V29_ENABLED": self.v29_enabled,
            "V29_REALITY": self.reality,
            "V29_REALITY_LLM": self.reality_llm,
            "V29_CLARITY_CONSENSUS": self.clarity_consensus,
            "V29_TARGET_LOCK": self.target_lock,
            "V29_INTENT_JOURNAL": self.intent_journal,
            "V29_SUBGOAL_LOCK": self.subgoal_lock,
            "V29_STAGNATION": self.stagnation,
            "V29_SMART_SCROLL": self.smart_scroll,
            "V29_PAGE_CONTEXT": self.page_context,
            "V29_ADAPTIVE_PERCEPTION": self.adaptive_perception,
            "V29_STRICT_VIEWPORT": self.strict_viewport,
            "V29_DIFFING": self.diffing,
            "V29_HYBRID_PRIMITIVES": self.hybrid_primitives,
            "V29_WEBDREAMER": self.webdreamer,
            "V29_WEBDREAMER_SITUATIONAL": self.webdreamer_situational,
            "V29_LATS": self.lats,
            "V29_SKILL_MEMORY_V2": self.skill_memory_v2,
            "V2_ENGINE": self.v2_engine,
        }


@dataclass(frozen=True)
class AppSettings:
    """Top-level typed settings aggregate for Agent-first-browse."""

    engine: EngineSettings
    safety: SafetySettings
    eval_connector: EvalConnectorSettings
    models: ModelProviderSettings
    observability: ObservabilitySettings
    flags: FeatureFlagSnapshot

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a JSON-ready diagnostic snapshot with zero raw secrets."""
        return {
            "engine": {
                "engine_mode": self.engine.engine_mode,
                "headless": self.engine.headless,
                "viewport": [self.engine.viewport_width, self.engine.viewport_height],
                "startup_timeout_s": self.engine.startup_timeout_s,
                "retry": {
                    "max_attempts": self.engine.retry.max_attempts,
                    "base_delay_ms": self.engine.retry.base_delay_ms,
                    "max_delay_ms": self.engine.retry.max_delay_ms,
                },
            },
            "safety": {
                "allow_private_hosts": self.safety.allow_private_hosts,
                "allow_about_blank": self.safety.allow_about_blank,
                "max_url_length": self.safety.max_url_length,
                "allowed_hosts_count": len(self.safety.allowed_hosts),
                "blocked_hosts_count": len(self.safety.blocked_hosts),
                "upload_dirs_count": len(self.safety.upload_dirs),
                "dry_run_mode": self.safety.dry_run_mode,
            },
            "eval_connector": {
                "socket_path": str(self.eval_connector.socket_path),
                "idle_timeout_s": self.eval_connector.idle_timeout_s,
                "max_mcp_output_chars": self.eval_connector.max_mcp_output_chars,
                "screenshot_dir": (
                    str(self.eval_connector.screenshot_dir)
                    if self.eval_connector.screenshot_dir is not None
                    else None
                ),
            },
            "models": {
                "configured_providers": list(self.models.configured_providers()),
                "openai_key_count": len(self.models.openai_api_keys),
                "google_key_count": len(self.models.google_api_keys),
                "groq_key_count": len(self.models.groq_api_keys),
                "nvidia_key_count": len(self.models.nvidia_api_keys),
                "worker_vlm_model": self.models.worker_vlm_model,
                "supervisor_model": self.models.supervisor_model,
            },
            "observability": {
                "langsmith_enabled": self.observability.langsmith_enabled,
                "langsmith_has_key": bool(self.observability.langsmith_api_key),
                "langsmith_project": self.observability.langsmith_project,
                "jsonl_log_path": (
                    str(self.observability.jsonl_log_path)
                    if self.observability.jsonl_log_path is not None
                    else None
                ),
            },
            "flags": self.flags.to_dict(),
        }


def _load_engine_settings(env: Mapping[str, str]) -> EngineSettings:
    raw_mode = env.get("AGENT_ENGINE_MODE", "").strip()
    mode = _validate_engine_mode(raw_mode) if raw_mode else "legacy"

    retry_opts = RetryOptions(
        max_attempts=(
            parse_env_int(env, "AGENT_RETRY_MAX_ATTEMPTS", default=DEFAULT_RETRY_CONFIG.max_attempts)
            if env.get("AGENT_RETRY_MAX_ATTEMPTS", "").strip()
            else None
        ),
        base_delay_ms=(
            parse_env_float(env, "AGENT_RETRY_BASE_DELAY_MS", default=DEFAULT_RETRY_CONFIG.base_delay_ms)
            if env.get("AGENT_RETRY_BASE_DELAY_MS", "").strip()
            else None
        ),
        max_delay_ms=(
            parse_env_float(env, "AGENT_RETRY_MAX_DELAY_MS", default=DEFAULT_RETRY_CONFIG.max_delay_ms)
            if env.get("AGENT_RETRY_MAX_DELAY_MS", "").strip()
            else None
        ),
    )
    resolved_retry = resolve_retry_config(retry_opts, defaults=DEFAULT_RETRY_CONFIG)

    return EngineSettings(
        engine_mode=mode,
        headless=parse_env_bool(env, "BROWSER_HEADLESS", default=False),
        viewport_width=parse_env_int(env, "AGENT_VIEWPORT_WIDTH", default=1280, min_value=1),
        viewport_height=parse_env_int(env, "AGENT_VIEWPORT_HEIGHT", default=800, min_value=1),
        local_cdp_endpoint=env.get("LOCAL_CDP_ENDPOINT", "http://localhost:9222").strip(),
        remote_websocket_url=env.get("REMOTE_WEBSOCKET_URL", "").strip(),
        startup_timeout_s=parse_env_float(
            env, "LOCAL_BROWSER_STARTUP_TIMEOUT_SECONDS", default=18.0, strictly_positive=True
        ),
        retry=resolved_retry,
    )


def _load_safety_settings(env: Mapping[str, str]) -> SafetySettings:
    raw_uploads = env.get("AGENT_UPLOAD_DIRS", "").strip()
    upload_dirs = tuple(
        Path(part).expanduser().resolve()
        for part in raw_uploads.split(os.pathsep)
        if part.strip()
    )
    return SafetySettings(
        allow_private_hosts=parse_env_bool(env, "AGENT_ALLOW_PRIVATE_HOSTS", default=False),
        allow_about_blank=parse_env_bool(env, "AGENT_ALLOW_ABOUT_BLANK", default=True),
        max_url_length=parse_env_int(
            env, "AGENT_MAX_URL_LENGTH", default=DEFAULT_MAX_URL_LENGTH, min_value=1
        ),
        allowed_hosts=parse_env_list(env.get("AGENT_ALLOWED_HOSTS")),
        blocked_hosts=parse_env_list(env.get("AGENT_BLOCKED_HOSTS")),
        upload_dirs=upload_dirs,
        dry_run_mode=parse_env_bool(env, "DRY_RUN_MODE", default=False),
    )


def _load_eval_connector_settings(env: Mapping[str, str]) -> EvalConnectorSettings:
    raw_sock = env.get("AGENT_EVAL_SOCKET", "").strip()
    sock = Path(raw_sock) if raw_sock else Path.home() / ".cache" / "afb" / "eval.sock"
    raw_shots = env.get("AGENT_EVAL_SCREENSHOT_DIR", "").strip()
    return EvalConnectorSettings(
        socket_path=sock,
        idle_timeout_s=parse_env_float(
            env, "AGENT_EVAL_IDLE_TIMEOUT_S", default=1800.0, min_value=0.0
        ),
        max_mcp_output_chars=parse_env_int(
            env, "AGENT_EVAL_MAX_OUTPUT_CHARS", default=24_000, min_value=1
        ),
        screenshot_dir=Path(raw_shots) if raw_shots else None,
    )


def _load_model_settings(env: Mapping[str, str]) -> ModelProviderSettings:
    return ModelProviderSettings(
        openai_api_keys=_merge_secret_keys(env, "OPENAI_API_KEY", "OPENAI_API_KEY_FALLBACKS"),
        openai_base_url=(
            env.get("OPENAI_BASE_URL", "").strip() or env.get("OPENAI_API_BASE", "").strip()
        ),
        google_api_keys=_merge_secret_keys(
            env,
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY_FALLBACKS",
            "GEMINI_API_KEY_FALLBACKS",
        ),
        groq_api_keys=_merge_secret_keys(env, "GROQ_API_KEY", "GROQ_API_KEY_FALLBACKS"),
        nvidia_api_keys=_merge_secret_keys(env, "NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"),
        nvidia_base_url=env.get(
            "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ).strip(),
        worker_vlm_model=env.get("WORKER_VLM_MODEL", "gemma-4-27b-it").strip() or "gemma-4-27b-it",
        supervisor_model=(
            env.get("SUPERVISOR_LLM_MODEL", "gemma-4-27b-it").strip() or "gemma-4-27b-it"
        ),
    )


def _load_observability_settings(env: Mapping[str, str]) -> ObservabilitySettings:
    tracing_v2 = parse_env_bool(env, "LANGCHAIN_TRACING_V2", default=False)
    tracing_ls = parse_env_bool(env, "LANGSMITH_TRACING", default=False)
    raw_key = env.get("LANGCHAIN_API_KEY", "").strip() or env.get("LANGSMITH_API_KEY", "").strip()
    raw_jsonl = env.get("AGENT_TELEMETRY_JSONL", "").strip()
    return ObservabilitySettings(
        langsmith_enabled=tracing_v2 or tracing_ls,
        langsmith_api_key=SecretValue(raw_key),
        langsmith_project=env.get("LANGCHAIN_PROJECT", "agent-first-browse").strip()
        or "agent-first-browse",
        langsmith_endpoint=env.get("LANGCHAIN_ENDPOINT", "").strip(),
        jsonl_log_path=Path(raw_jsonl) if raw_jsonl else None,
    )


def _load_feature_flags(env: Mapping[str, str], engine_mode: str) -> FeatureFlagSnapshot:
    master = parse_env_bool(env, "V29_ENABLED", default=True, strict=False)

    def _sub(name: str, default: bool = True) -> bool:
        return master and parse_env_bool(env, name, default=default, strict=False)

    reality = _sub("V29_REALITY", True)
    adaptive = _sub("V29_ADAPTIVE_PERCEPTION", True)
    dreamer = _sub("V29_WEBDREAMER", True)

    return FeatureFlagSnapshot(
        v29_enabled=master,
        reality=reality,
        reality_llm=reality and parse_env_bool(env, "V29_REALITY_LLM", default=True, strict=False),
        clarity_consensus=_sub("V29_CLARITY_CONSENSUS", True),
        target_lock=_sub("V29_TARGET_LOCK", True),
        intent_journal=_sub("V29_INTENT_JOURNAL", True),
        subgoal_lock=_sub("V29_SUBGOAL_LOCK", True),
        stagnation=_sub("V29_STAGNATION", True),
        smart_scroll=_sub("V29_SMART_SCROLL", True),
        page_context=_sub("V29_PAGE_CONTEXT", True),
        adaptive_perception=adaptive,
        strict_viewport=adaptive
        and parse_env_bool(env, "V29_STRICT_VIEWPORT", default=True, strict=False),
        diffing=_sub("V29_DIFFING", True),
        hybrid_primitives=_sub("V29_HYBRID_PRIMITIVES", True),
        webdreamer=dreamer,
        webdreamer_situational=dreamer
        and parse_env_bool(env, "V29_WEBDREAMER_SITUATIONAL", default=True, strict=False),
        lats=_sub("V29_LATS", False),
        skill_memory_v2=_sub("V29_SKILL_MEMORY_V2", False),
        v2_engine=(engine_mode == "v2"),
    )


def load_settings(env: Mapping[str, str] | None = None) -> AppSettings:
    """Parse and eagerly validate all Agent-first-browse runtime settings.

    Reads from `env` when supplied, or live `os.environ` by default. Never
    caches silently and never mutates `os.environ`.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    engine = _load_engine_settings(source)
    return AppSettings(
        engine=engine,
        safety=_load_safety_settings(source),
        eval_connector=_load_eval_connector_settings(source),
        models=_load_model_settings(source),
        observability=_load_observability_settings(source),
        flags=_load_feature_flags(source, engine.engine_mode),
    )


__all__ = [
    "AppSettings",
    "EngineSettings",
    "EvalConnectorSettings",
    "FeatureFlagSnapshot",
    "ModelProviderSettings",
    "ObservabilitySettings",
    "SafetySettings",
    "SecretValue",
    "load_settings",
    "parse_env_bool",
    "parse_env_float",
    "parse_env_int",
    "parse_env_list",
]
