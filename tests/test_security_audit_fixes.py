"""Unit tests for security audit fixes: path confinement & in-flight secret redaction."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import record_visual_demo  # noqa: E402

import model_registry  # noqa: E402

# ── 1. scripts/record_visual_demo.py path confinement ───────────────────────────


def test_resolve_confined_out_dir_allows_workspace_and_temp(tmp_path: Path) -> None:
    allowed = (tmp_path / "workspace", tmp_path / "tmp_root")
    allowed[0].mkdir()
    allowed[1].mkdir()

    ok_ws = record_visual_demo._resolve_confined_out_dir(
        allowed[0] / "docs" / "assets" / "raw",
        allowed_roots=allowed,
    )
    assert ok_ws == (allowed[0] / "docs" / "assets" / "raw").resolve()

    ok_tmp = record_visual_demo._resolve_confined_out_dir(
        allowed[1] / "demo_run",
        allowed_roots=allowed,
    )
    assert ok_tmp == (allowed[1] / "demo_run").resolve()


def test_resolve_confined_out_dir_blocks_escape(tmp_path: Path) -> None:
    allowed = (tmp_path / "workspace",)
    allowed[0].mkdir()
    outside = tmp_path / "outside_dir"
    outside.mkdir()

    with pytest.raises(ValueError, match="outside allowed"):
        record_visual_demo._resolve_confined_out_dir(outside, allowed_roots=allowed)

    with pytest.raises(ValueError, match="outside allowed"):
        record_visual_demo._resolve_confined_out_dir(
            allowed[0] / ".." / "outside_dir",
            allowed_roots=allowed,
        )


def test_validate_video_filename_rejects_traversal() -> None:
    assert record_visual_demo._validate_video_filename("demo_1280x720.webm") == "demo_1280x720.webm"
    for bad in ("", "   ", ".", "..", "../escape.webm", "sub/demo.webm", r"sub\demo.webm"):
        with pytest.raises(ValueError, match="Invalid video filename"):
            record_visual_demo._validate_video_filename(bad)


def test_safe_rmtree_confined_deletes_valid_subdir_and_blocks_symlinks_and_escapes(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    out_dir = root / "out"
    tmp_dir = out_dir / "_recording_tmp"
    tmp_dir.mkdir(parents=True)
    (tmp_dir / "clip.webm").write_bytes(b"demo")

    # Legitimate subdirectory inside confined parent is removed cleanly
    record_visual_demo._safe_rmtree_confined(
        tmp_dir,
        expected_parent=out_dir,
        allowed_roots=(root,),
    )
    assert not tmp_dir.exists()
    assert out_dir.exists()

    # Refuses deleting expected_parent itself
    with pytest.raises(ValueError, match="outside expected parent"):
        record_visual_demo._safe_rmtree_confined(
            out_dir,
            expected_parent=out_dir,
            allowed_roots=(root,),
        )

    # Refuses deleting a symlink pointing outside out_dir
    victim = tmp_path / "victim_dir"
    victim.mkdir()
    (victim / "keep.txt").write_text("safe", encoding="utf-8")
    symlink_tmp = out_dir / "_recording_tmp"
    symlink_tmp.symlink_to(victim, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        record_visual_demo._safe_rmtree_confined(
            symlink_tmp,
            expected_parent=out_dir,
            allowed_roots=(root,),
        )
    assert (victim / "keep.txt").exists()


# ── 2. src/model_registry.py in-flight secret redaction & safe repr ───────────


def test_premium_config_masks_secrets_in_repr_and_str(monkeypatch: pytest.MonkeyPatch) -> None:
    synth_key = "custom-vendor-secret-" + "9" * 16
    monkeypatch.setenv("AGENT_MODE", "auto")
    monkeypatch.setenv("PREMIUM_API_KEY", synth_key)
    monkeypatch.delenv("PREMIUM_API_KEYS", raising=False)

    assert model_registry.get_agent_mode() == "premium"

    cfg = model_registry._premium_config()
    assert len(cfg["keys"]) == 1
    assert synth_key not in repr(cfg)
    assert synth_key not in str(cfg)
    assert "***" in repr(cfg["keys"][0])
    assert str(cfg["keys"][0]) == "***"
    assert cfg["keys"][0].get_secret_value() == synth_key


def test_redact_secret_text_masks_known_and_custom_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    synth_groq = "gsk_" + "A" * 28
    synth_openai = "sk-" + "B" * 28
    synth_google = "AIza" + "C" * 34
    synth_custom = "csk-vendor-" + "Z" * 20

    monkeypatch.setenv("PREMIUM_API_KEY", synth_custom)
    raw = (
        f"401 Unauthorized: Bearer {synth_groq} | "
        f"Incorrect API key provided: {synth_openai} | "
        f"GET https://generativelanguage.googleapis.com/v1?key={synth_google} | "
        f"custom={synth_custom}"
    )
    redacted = model_registry._redact_secret_text(raw)

    for secret in (synth_groq, synth_openai, synth_google, synth_custom):
        assert secret not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.asyncio
async def test_invoke_with_failover_redacts_secrets_from_logs_and_raised_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    synth_key = "gsk_" + "K" * 28
    synth_custom = "vendor-tok-" + "Q" * 18
    monkeypatch.setenv("PREMIUM_API_KEY", synth_custom)

    class _LeakyLLM:
        async def ainvoke(self, _messages):
            raise RuntimeError(
                f"401 invalid api_key={synth_key} for custom token {synth_custom}"
            )

    mc = model_registry.ModelClient("groq:test-model:0", _LeakyLLM(), "groq", "text")
    with (
        caplog.at_level(logging.WARNING, logger="model_registry"),
        pytest.raises(RuntimeError) as exc_info,
    ):
        await model_registry.invoke_with_failover([mc], ["hello"], schema=None)

    assert synth_key not in str(exc_info.value)
    assert synth_custom not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)

    logged_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert synth_key not in logged_text
    assert synth_custom not in logged_text


def test_as_model_clients_redacts_fallback_repr_credentials() -> None:
    synth_key = "sk-" + "R" * 28

    class _RawClientWithLeakyRepr:
        def __str__(self) -> str:
            return f"ChatOpenAI(api_key='{synth_key}', model='gpt-5')"

    clients = model_registry._as_model_clients([_RawClientWithLeakyRepr()])
    assert len(clients) == 1
    assert synth_key not in clients[0].name
    assert synth_key not in repr(clients[0])
