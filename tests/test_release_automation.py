"""Tests for automated release version sync and GitHub Actions release pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import release  # noqa: E402


def _create_mock_repo(tmp_path: Path, version: str = "0.1.1") -> Path:
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "eval_connector").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "agent-first-browse"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (tmp_path / "server.json").write_text(
        json.dumps(
            {
                "name": release.MCP_SERVER_NAME,
                "version": version,
                "packages": [
                    {
                        "registryType": "pypi",
                        "identifier": "agent-first-browse",
                        "version": version,
                        "transport": {"type": "stdio"},
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "app" / "__init__.py").write_text(
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "eval_connector" / "mcp_server.py").write_text(
        f'SERVER_VERSION = "{version}"\n',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        f"# Demo\n\n{release.MCP_OWNERSHIP_MARKER}\n",
        encoding="utf-8",
    )
    (tmp_path / "glama.json").write_text(
        json.dumps({"maintainers": ["SandeepAi369"]}) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_live_repo_versions_and_markers_are_in_sync() -> None:
    """The real repository must pass release consistency checks cleanly."""
    assert release.check_consistency(REPO_ROOT) == []


def test_compute_next_version_bumps_patch_minor_major() -> None:
    assert release.compute_next_version("0.1.1", "patch") == "0.1.2"
    assert release.compute_next_version("0.1.1", "minor") == "0.2.0"
    assert release.compute_next_version("0.1.1", "major") == "1.0.0"
    with pytest.raises(ValueError):
        release.compute_next_version("not-a-version", "patch")


def test_apply_version_updates_all_files_atomically(tmp_path: Path) -> None:
    root = _create_mock_repo(tmp_path, "0.1.1")
    changed = release.apply_version("0.2.0", root)
    assert len(changed) == 4
    assert release.check_consistency(root, expected_tag="v0.2.0") == []

    snap = release.read_versions(root)
    assert snap.pyproject == "0.2.0"
    assert snap.server_json == "0.2.0"
    assert snap.server_json_packages == ("0.2.0",)
    assert snap.app_init == "0.2.0"
    assert snap.mcp_server == "0.2.0"


def test_check_consistency_catches_version_drift_and_missing_marker(tmp_path: Path) -> None:
    root = _create_mock_repo(tmp_path, "0.1.1")
    (root / "src" / "eval_connector" / "mcp_server.py").write_text(
        'SERVER_VERSION = "0.1.0"\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text("# No marker here\n", encoding="utf-8")

    errors = release.check_consistency(root, expected_tag="v0.1.2")
    assert any("SERVER_VERSION" in e for e in errors)
    assert any("README.md" in e for e in errors)
    assert any("Release tag 'v0.1.2'" in e for e in errors)


def test_release_workflow_declares_required_oidc_and_sync_jobs() -> None:
    """Ensure .github/workflows/release.yml retains OIDC publish jobs and idempotency."""
    wf = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "id-token: write" in wf
    assert "pypa/gh-action-pypi-publish@release/v1" in wf
    assert "mcp-publisher login github-oidc" in wf
    assert "mcp-publisher publish" in wf
    assert "gh release create" in wf
    assert "scripts/release.py --check" in wf
