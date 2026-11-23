"""Real-world unit tests for src/core/settings.py."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import core
from core import (
    AppSettings,
    ConfigValidationError,
    EngineSettings,
    EvalConnectorSettings,
    FeatureFlagSnapshot,
    ModelProviderSettings,
    ObservabilitySettings,
    SafetySettings,
    SecretValue,
    load_settings,
    parse_env_bool,
    parse_env_float,
    parse_env_int,
    parse_env_list,
    validate_url,
)


# ── 1. Public Barrel & SecretValue Masking ─────────────────────────────────────


def test_core_barrel_exports_chunk7_symbols_and_secret_masking():
    for name in (
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
    ):
        assert name in core.__all__
        assert hasattr(core, name)

    raw_secret = "sk-" + "S3cr3tValu3" * 3
    wrapped = SecretValue(f"  {raw_secret}  ")
    assert bool(wrapped) is True
    assert len(wrapped) == len(raw_secret)
    assert wrapped.get_secret_value() == raw_secret
    assert raw_secret not in str(wrapped)
    assert raw_secret not in repr(wrapped)
    assert wrapped == SecretValue(raw_secret)
    assert wrapped != raw_secret

    empty = SecretValue("")
    assert bool(empty) is False
    assert str(empty) == ""
    assert repr(empty) == "SecretValue('')"

    with pytest.raises(ConfigValidationError):
        SecretValue(123)  # type: ignore[arg-type]


# ── 2. Defaults, Decoupling & Full Env Parsing ─────────────────────────────────


def test_load_settings_empty_env_defaults_and_no_promoter_fields():
    settings = load_settings(env={})
    assert isinstance(settings, AppSettings)
    assert settings.engine.engine_mode == "legacy"
    assert settings.engine.headless is False
    assert (settings.engine.viewport_width, settings.engine.viewport_height) == (1280, 800)
    assert settings.safety.allow_private_hosts is False
    assert settings.safety.allow_about_blank is True
    assert settings.models.configured_providers() == ()
    assert settings.observability.langsmith_enabled is False
    assert settings.flags.v29_enabled is True
    assert settings.flags.v2_engine is False
    assert settings.flags.lats is False

    safe = settings.to_safe_dict()
    encoded = json.dumps(safe)
    assert "promotion" not in encoded.lower()
    assert "campaign" not in encoded.lower()


def test_load_settings_full_env_dedup_keys_url_policy_bridge_and_zero_secret_leaks(tmp_path):
    fake_openai_1 = "sk-" + "Alpha111111111111111111111"
    fake_openai_2 = "sk-" + "Beta2222222222222222222222"
    fake_google = "AIza" + "G" * 34
    fake_groq = "gsk_" + "R" * 28
    fake_nvidia = "nvapi-" + "N" * 28
    fake_ls = "ls__" + "TracingKey999999999999999"

    upload_a = tmp_path / "uploads_a"
    upload_b = tmp_path / "uploads_b"
    env = {
        "AGENT_ENGINE_MODE": " V2 ",
        "BROWSER_HEADLESS": "true",
        "AGENT_VIEWPORT_WIDTH": "1440",
        "AGENT_VIEWPORT_HEIGHT": "900",
        "LOCAL_BROWSER_STARTUP_TIMEOUT_SECONDS": "25.5",
        "AGENT_RETRY_MAX_ATTEMPTS": "5",
        "AGENT_RETRY_BASE_DELAY_MS": "150",
        "AGENT_RETRY_MAX_DELAY_MS": "2500",
        "AGENT_ALLOW_PRIVATE_HOSTS": "yes",
        "AGENT_ALLOWED_HOSTS": "docs.example.com; *.api.example.com",
        "AGENT_BLOCKED_HOSTS": "evil.example.com",
        "AGENT_UPLOAD_DIRS": f"{upload_a}{os.pathsep}{upload_b}",
        "DRY_RUN_MODE": "1",
        "AGENT_EVAL_SOCKET": str(tmp_path / "eval.sock"),
        "AGENT_EVAL_IDLE_TIMEOUT_S": "600",
        "AGENT_EVAL_MAX_OUTPUT_CHARS": "12000",
        "AGENT_EVAL_SCREENSHOT_DIR": str(tmp_path / "shots"),
        "OPENAI_API_KEY": fake_openai_1,
        "OPENAI_API_KEY_FALLBACKS": f"{fake_openai_1}, {fake_openai_2}",
        "GOOGLE_API_KEY": fake_google,
        "GEMINI_API_KEY": fake_google,
        "GROQ_API_KEY": fake_groq,
        "NVIDIA_NIM_API_KEY": fake_nvidia,
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": fake_ls,
        "LANGCHAIN_PROJECT": "afb-v2",
        "AGENT_TELEMETRY_JSONL": str(tmp_path / "events.jsonl"),
    }

    settings = load_settings(env=env)
    assert settings.engine.engine_mode == "v2"
    assert settings.engine.headless is True
    assert settings.engine.viewport_width == 1440
    assert settings.engine.viewport_height == 900
    assert settings.engine.retry.max_attempts == 5
    assert settings.engine.retry.base_delay_ms == 150.0
    assert settings.engine.retry.max_delay_ms == 2500.0
    assert settings.flags.v2_engine is True

    # Safety -> UrlPolicy bridge works seamlessly
    policy = settings.safety.to_url_policy()
    assert validate_url("https://docs.example.com/intro", policy).ok is True
    assert validate_url("https://v1.api.example.com/run", policy).ok is True
    assert validate_url("https://evil.example.com/", policy).ok is False
    assert settings.safety.upload_dirs == (upload_a.resolve(), upload_b.resolve())
    assert settings.safety.dry_run_mode is True

    # Eval connector & observability
    assert settings.eval_connector.socket_path == tmp_path / "eval.sock"
    assert settings.eval_connector.idle_timeout_s == 600.0
    assert settings.eval_connector.max_mcp_output_chars == 12000
    assert settings.eval_connector.screenshot_dir == tmp_path / "shots"
    assert settings.observability.langsmith_enabled is True
    assert settings.observability.langsmith_project == "afb-v2"
    assert settings.observability.jsonl_log_path == tmp_path / "events.jsonl"

    # Keys deduplicated while preserving order
    assert [k.get_secret_value() for k in settings.models.openai_api_keys] == [
        fake_openai_1,
        fake_openai_2,
    ]
    assert [k.get_secret_value() for k in settings.models.google_api_keys] == [fake_google]
    assert settings.models.configured_providers() == ("openai", "google", "groq", "nvidia")

    # Critical security guarantee: ZERO raw secrets in repr(), str(), or to_safe_dict()
    safe_blob = repr(settings) + "\n" + str(settings) + "\n" + json.dumps(settings.to_safe_dict())
    for raw in (fake_openai_1, fake_openai_2, fake_google, fake_groq, fake_nvidia, fake_ls):
        assert raw not in safe_blob


# ── 3. Feature Flag Kill-Switch & Live Monkeypatch Reactivity ──────────────────


def test_feature_flag_master_kill_switch_and_sub_gating():
    killed = load_settings(env={"V29_ENABLED": "0", "AGENT_ENGINE_MODE": "v2"})
    assert killed.flags.v29_enabled is False
    assert killed.flags.reality is False
    assert killed.flags.reality_llm is False
    assert killed.flags.adaptive_perception is False
    assert killed.flags.strict_viewport is False
    assert killed.flags.webdreamer is False
    assert killed.flags.webdreamer_situational is False
    # Dual-mode engine switch stays independent of V29 kill-switch
    assert killed.flags.v2_engine is True

    partial = load_settings(
        env={
            "V29_REALITY": "0",
            "V29_ADAPTIVE_PERCEPTION": "false",
            "V29_WEBDREAMER": "off",
            "V29_LATS": "1",
        }
    )
    assert partial.flags.reality is False
    assert partial.flags.reality_llm is False
    assert partial.flags.adaptive_perception is False
    assert partial.flags.strict_viewport is False
    assert partial.flags.webdreamer is False
    assert partial.flags.webdreamer_situational is False
    assert partial.flags.lats is True


def test_load_settings_reads_live_os_environ_without_caching_or_mutating(monkeypatch):
    monkeypatch.setenv("AGENT_ENGINE_MODE", "legacy")
    monkeypatch.setenv("V29_ENABLED", "1")
    before_keys = set(os.environ.keys())

    s1 = load_settings()
    assert s1.engine.engine_mode == "legacy"
    assert s1.flags.v29_enabled is True

    monkeypatch.setenv("AGENT_ENGINE_MODE", "v2")
    monkeypatch.setenv("V29_ENABLED", "0")
    s2 = load_settings()
    assert s2.engine.engine_mode == "v2"
    assert s2.flags.v29_enabled is False
    assert set(os.environ.keys()) == before_keys


# ── 4. Eager Validation Error Coverage ─────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_env",
    [
        {"AGENT_ENGINE_MODE": "turbo"},
        {"BROWSER_HEADLESS": "maybe"},
        {"AGENT_VIEWPORT_WIDTH": "not-an-int"},
        {"AGENT_VIEWPORT_WIDTH": "0"},
        {"AGENT_VIEWPORT_HEIGHT": "-100"},
        {"LOCAL_BROWSER_STARTUP_TIMEOUT_SECONDS": "0"},
        {"LOCAL_BROWSER_STARTUP_TIMEOUT_SECONDS": "nan"},
        {"AGENT_RETRY_MAX_ATTEMPTS": "0"},
        {"AGENT_RETRY_BASE_DELAY_MS": "5000"},  # exceeds default max_delay_ms=3000
        {"AGENT_MAX_URL_LENGTH": "0"},
        {"AGENT_EVAL_IDLE_TIMEOUT_S": "-5"},
        {"AGENT_EVAL_MAX_OUTPUT_CHARS": "0"},
    ],
)
def test_load_settings_rejects_invalid_env_eagerly(bad_env):
    with pytest.raises(ConfigValidationError):
        load_settings(env=bad_env)


def test_parse_helpers_and_direct_dataclass_validation():
    assert parse_env_bool({"F": "YES"}, "F", default=False) is True
    assert parse_env_bool({"F": "off"}, "F", default=True) is False
    assert parse_env_int({"N": "42"}, "N", default=1, min_value=10) == 42
    assert parse_env_float({"X": "2.5"}, "X", default=1.0, strictly_positive=True) == 2.5
    assert parse_env_list("a, b; c , ") == ("a", "b", "c")

    with pytest.raises(ConfigValidationError):
        EngineSettings(viewport_width=True)  # type: ignore[arg-type]
    with pytest.raises(ConfigValidationError):
        SafetySettings(max_url_length=0)
    with pytest.raises(ConfigValidationError):
        EvalConnectorSettings(idle_timeout_s=-1.0)
    assert isinstance(ModelProviderSettings(), ModelProviderSettings)
    assert isinstance(ObservabilitySettings(), ObservabilitySettings)
    assert isinstance(FeatureFlagSnapshot(), FeatureFlagSnapshot)
    assert Path(EvalConnectorSettings().socket_path).name == "eval.sock"
