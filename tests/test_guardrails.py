"""Real-world unit & temporary-fixture tests for scripts/check_guardrails.py.

Verifies both:
  - Positive pass behavior against the live Agent-first-browse repository.
  - Negative violation detection against isolated temporary git/filesystem fixtures
    (frozen-file edits/deletions, UNIVERSAL-ONLY brand leaks, broken __all__ &
    CLI script exports, raw secret leaks with redacted reporting, and empty or
    script-style test files) — without modifying any real source files.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_guardrails.py"


def _load_guardrails_module():
    spec = importlib.util.spec_from_file_location("check_guardrails", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_guardrails"] = mod
    spec.loader.exec_module(mod)
    return mod


guardrails = _load_guardrails_module()


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _build_minimal_repo(tmp_path: Path, *, init_git: bool = True) -> Path:
    """Create an isolated mini-repo matching Agent-first-browse's guarded layout."""
    repo = tmp_path / "mini_repo"
    repo.mkdir()

    for rel in guardrails.FROZEN_FILES:
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# frozen legacy module\nVALUE = 1\n", encoding="utf-8")

    for rel in guardrails.UNIVERSAL_ONLY_FILES:
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            '"""Generic universal DOM perception module."""\n'
            'ROLES = ("button", "textbox", "link", "checkbox")\n',
            encoding="utf-8",
        )

    v2_init = repo / "src" / "v2_engine" / "__init__.py"
    v2_init.parent.mkdir(parents=True, exist_ok=True)
    v2_init.write_text(
        'ENGINE_MODE_V2 = "v2"\n'
        '__all__ = ["ENGINE_MODE_V2"]\n',
        encoding="utf-8",
    )

    eval_cli = repo / "src" / "eval_connector" / "cli.py"
    eval_cli.parent.mkdir(parents=True, exist_ok=True)
    eval_cli.write_text("def main():\n    return 0\n", encoding="utf-8")

    eval_init = repo / "src" / "eval_connector" / "__init__.py"
    eval_init.write_text(
        'from eval_connector.cli import main\n'
        '__all__ = ["main"]\n',
        encoding="utf-8",
    )

    (repo / "pyproject.toml").write_text(
        '[project]\n'
        'name = "mini-repo"\n'
        '[project.scripts]\n'
        'mini-eval = "eval_connector.cli:main"\n',
        encoding="utf-8",
    )

    (repo / ".gitignore").write_text(".env\n__pycache__/\n", encoding="utf-8")
    (repo / ".env.example").write_text("OPENAI_API_KEY=sk-...\n", encoding="utf-8")

    test_file = repo / "tests" / "test_sample.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(
        "def test_ok():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    if init_git:
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.name", "Test Bot")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "initial fixture commit")

    return repo


# ── 1. Live Repository Verification ────────────────────────────────────────────


def test_live_repo_all_guardrails_pass():
    report = guardrails.run_all_checks(REPO_ROOT, base_ref="main")
    assert report.passed is True
    assert report.error_count == 0

    by_name = {r.name: r for r in report.results}
    assert by_name["frozen_files"].passed is True
    assert by_name["frozen_files"].checked_count == len(guardrails.FROZEN_FILES)
    assert by_name["universal_only"].passed is True
    assert by_name["universal_only"].checked_count == len(guardrails.UNIVERSAL_ONLY_FILES)
    assert by_name["exports"].passed is True
    assert by_name["exports"].checked_count >= 4
    assert by_name["secrets"].passed is True
    assert by_name["secrets"].checked_count >= 50
    assert by_name["test_hygiene"].passed is True
    # After Chunk 8 cleanup, live repo has zero legacy script-test warnings
    warn_paths = {f.path for f in by_name["test_hygiene"].findings if f.severity == "warning"}
    assert warn_paths == set(guardrails.KNOWN_LEGACY_SCRIPT_TESTS)


def test_live_repo_cli_json_passes_under_normal_and_strict_test_hygiene():
    ok_proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--base-ref", "main", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok_proc.returncode == 0, ok_proc.stderr
    payload = json.loads(ok_proc.stdout)
    assert payload["passed"] is True
    assert payload["error_count"] == 0
    assert payload["warning_count"] == 0

    strict_proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--base-ref", "main", "--strict-test-hygiene", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict_proc.returncode == 0, strict_proc.stderr
    strict_payload = json.loads(strict_proc.stdout)
    assert strict_payload["passed"] is True
    assert strict_payload["error_count"] == 0
    assert strict_payload["warning_count"] == 0


# ── 2. Frozen §6 File Detection (Real Git Diffs) ───────────────────────────────


def test_frozen_files_pass_when_clean(tmp_path):
    repo = _build_minimal_repo(tmp_path)
    res = guardrails.check_frozen_files(repo, base_ref="main")
    assert res.passed is True
    assert res.findings == []


def test_frozen_files_catches_unstaged_staged_and_committed_edits(tmp_path):
    repo = _build_minimal_repo(tmp_path)

    # Unstaged edit to a frozen file
    cdp_click = repo / "src" / "cdp_click.py"
    cdp_click.write_text("# modified unstaged\n", encoding="utf-8")
    res_unstaged = guardrails.check_frozen_files(repo)
    assert res_unstaged.passed is False
    assert any(f.path == "src/cdp_click.py" and f.severity == "error" for f in res_unstaged.findings)

    # Staged edit to a frozen file
    _git(repo, "add", "src/cdp_click.py")
    res_staged = guardrails.check_frozen_files(repo)
    assert res_staged.passed is False
    assert any(f.path == "src/cdp_click.py" and f.severity == "error" for f in res_staged.findings)

    # Committed edit on a feature branch relative to main
    _git(repo, "checkout", "-b", "feat/accidental-touch")
    _git(repo, "commit", "-m", "oops touched frozen file")
    res_branch = guardrails.check_frozen_files(repo, base_ref="main")
    assert res_branch.passed is False
    assert any(f.path == "src/cdp_click.py" and f.severity == "error" for f in res_branch.findings)


def test_frozen_files_catches_deleted_file_and_invalid_base_ref(tmp_path):
    repo = _build_minimal_repo(tmp_path)
    (repo / "src" / "virtual_display.py").unlink()

    res_missing = guardrails.check_frozen_files(repo, changed_files=[])
    assert res_missing.passed is False
    assert any(f.path == "src/virtual_display.py" and "missing" in f.message.lower() for f in res_missing.findings)

    res_bad_ref = guardrails.check_frozen_files(repo, base_ref="nonexistent-ref-999")
    assert res_bad_ref.passed is False
    assert any("base-ref" in f.message for f in res_bad_ref.findings)


# ── 3. UNIVERSAL-ONLY Perception Guard ─────────────────────────────────────────


def test_universal_only_catches_brand_literals_and_syntax_errors(tmp_path):
    repo = _build_minimal_repo(tmp_path, init_git=False)

    # Clean baseline passes
    assert guardrails.check_universal_only(repo).passed is True

    # Inject site/brand literal in code and domain in comment
    v2_perc = repo / "src" / "v2_engine" / "perception.py"
    v2_perc.write_text(
        '"""Generic perception."""\n'
        '# Workaround for dev.to layout\n'
        'BAD_HOST = "https://www.reddit.com/r/test"\n',
        encoding="utf-8",
    )
    res = guardrails.check_universal_only(repo)
    assert res.passed is False
    messages = " ".join(f.message for f in res.findings)
    assert "dev.to" in messages
    assert "reddit" in messages
    assert all(f.line is not None for f in res.findings)

    # Syntax error fails loudly
    (repo / "src" / "perception_engine.py").write_text("def broken(:\n", encoding="utf-8")
    res_syntax = guardrails.check_universal_only(repo)
    assert res_syntax.passed is False
    assert any("SyntaxError" in f.message for f in res_syntax.findings)


# ── 4. Public __all__ Barrels & pyproject.toml CLI Scripts ─────────────────────


def test_exports_pass_and_auto_detect_core_package(tmp_path):
    repo = _build_minimal_repo(tmp_path, init_git=False)
    core_init = repo / "src" / "core" / "__init__.py"
    core_init.parent.mkdir(parents=True, exist_ok=True)
    core_init.write_text('CORE_OK = True\n__all__ = ["CORE_OK"]\n', encoding="utf-8")

    res = guardrails.check_public_exports(repo)
    assert res.passed is True
    # v2_engine + eval_connector + optional core + 1 pyproject script = 4
    assert res.checked_count == 4


def test_exports_catches_missing_symbol_duplicates_and_broken_cli_script(tmp_path):
    repo = _build_minimal_repo(tmp_path, init_git=False)

    # Unresolved symbol in __all__
    (repo / "src" / "v2_engine" / "__init__.py").write_text(
        'ENGINE_MODE_V2 = "v2"\n'
        '__all__ = ["ENGINE_MODE_V2", "NON_EXISTENT_EXPORT"]\n',
        encoding="utf-8",
    )
    # Duplicate symbol in eval_connector __all__
    (repo / "src" / "eval_connector" / "__init__.py").write_text(
        'from eval_connector.cli import main\n'
        '__all__ = ["main", "main"]\n',
        encoding="utf-8",
    )
    # Broken script entrypoint in pyproject.toml
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "mini-repo"\n'
        '[project.scripts]\n'
        'broken-cli = "eval_connector.cli:missing_entry"\n',
        encoding="utf-8",
    )

    res = guardrails.check_public_exports(repo)
    assert res.passed is False
    msgs = [f.message for f in res.findings]
    assert any("NON_EXISTENT_EXPORT" in m for m in msgs)
    assert any("Duplicate symbol 'main'" in m for m in msgs)
    assert any("broken-cli" in m for m in msgs)


# ── 5. Raw Secret Leak Scanner (With Strict Redaction) ─────────────────────────


def test_secrets_skips_gitignored_env_and_catches_leaks_without_echoing_raw_secret(tmp_path):
    repo = _build_minimal_repo(tmp_path, init_git=True)

    # Construct synthetic secret-shaped strings dynamically so this test file stays clean
    fake_openai = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
    fake_github = "ghp_" + "9876543210abcdefghij1234567890"
    fake_google = "AIza" + "SyD" + "X" * 32
    fake_pem = "-----BEGIN " + "RSA PRIVATE KEY-----"
    fake_groq = "gsk_" + "Z" * 28

    # Gitignored .env must NOT cause false failures
    (repo / ".env").write_text(f"OPENAI_API_KEY={fake_openai}\n", encoding="utf-8")
    assert guardrails.check_secret_leaks(repo).passed is True

    # Untracked new source file with multiple simulated leaks MUST be caught before staging
    leak_file = repo / "src" / "accidental_leak.py"
    leak_file.write_text(
        f'KEY1 = "{fake_openai}"\n'
        f'KEY2 = "{fake_github}"\n'
        f'KEY3 = "{fake_google}"\n'
        f'PEM = "{fake_pem}"\n'
        f'GROQ = "{fake_groq}"\n',
        encoding="utf-8",
    )

    res = guardrails.check_secret_leaks(repo)
    assert res.passed is False
    assert len(res.findings) == 5

    # Critical security guarantee: raw secret values must NEVER be echoed in findings
    combined_messages = "\n".join(f.message for f in res.findings)
    for secret in (fake_openai, fake_github, fake_google, fake_pem, fake_groq):
        assert secret not in combined_messages


# ── 6. Test Hygiene Checker ────────────────────────────────────────────────────


def test_hygiene_accepts_sync_async_and_class_tests_and_rejects_empty_files(tmp_path):
    repo = _build_minimal_repo(tmp_path, init_git=False)

    (repo / "tests" / "test_async_and_cls.py").write_text(
        "async def test_async_case():\n"
        "    assert True\n\n"
        "class TestGrouped:\n"
        "    def test_method_case(self):\n"
        "        assert 1 == 1\n",
        encoding="utf-8",
    )
    # Files under tests/live/ and tests/fakes/ must be ignored
    live_dir = repo / "tests" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "test_live_script.py").write_text("print('live only')\n", encoding="utf-8")

    res_clean = guardrails.check_test_hygiene(repo, known_legacy_debt=())
    assert res_clean.passed is True
    assert res_clean.findings == []

    # Add an empty test file and an import-time assert script
    (repo / "tests" / "test_empty_regression.py").write_text("# oops empty\n", encoding="utf-8")
    (repo / "tests" / "test_bare_assert.py").write_text("assert 1 + 1 == 2\n", encoding="utf-8")

    res_fail = guardrails.check_test_hygiene(repo, known_legacy_debt=())
    assert res_fail.passed is False
    bad_paths = {f.path for f in res_fail.findings if f.severity == "error"}
    assert bad_paths == {"tests/test_empty_regression.py", "tests/test_bare_assert.py"}


def test_hygiene_warns_on_stale_debt_allowlist_entry(tmp_path):
    repo = _build_minimal_repo(tmp_path, init_git=False)
    # tests/test_sample.py ALREADY has def test_ok(), so listing it as debt is stale
    res = guardrails.check_test_hygiene(repo, known_legacy_debt=("tests/test_sample.py",))
    assert res.passed is True
    assert any(
        f.severity == "warning" and "Stale entry" in f.message and f.path == "tests/test_sample.py"
        for f in res.findings
    )

    # Strict mode turns tracked debt warnings into hard errors
    (repo / "tests" / "test_tracked_empty.py").write_text("# legacy script\n", encoding="utf-8")
    res_warn = guardrails.check_test_hygiene(
        repo, strict=False, known_legacy_debt=("tests/test_tracked_empty.py",)
    )
    assert res_warn.passed is True
    res_strict = guardrails.check_test_hygiene(
        repo, strict=True, known_legacy_debt=("tests/test_tracked_empty.py",)
    )
    assert res_strict.passed is False
    assert any(f.severity == "error" and f.path == "tests/test_tracked_empty.py" for f in res_strict.findings)


# ── 7. End-to-End CLI Failure Exit Code Test ───────────────────────────────────


def test_cli_exits_nonzero_on_simulated_fixture_violation(tmp_path):
    repo = _build_minimal_repo(tmp_path, init_git=True)
    (repo / "tests" / "test_empty_new.py").write_text("", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(repo), "--base-ref", "main", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    data = json.loads(proc.stdout)
    assert data["passed"] is False
    assert data["error_count"] >= 1
    hygiene = next(r for r in data["results"] if r["name"] == "test_hygiene")
    assert hygiene["passed"] is False
    assert any(f["path"] == "tests/test_empty_new.py" for f in hygiene["findings"])
