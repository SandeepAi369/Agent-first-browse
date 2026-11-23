#!/usr/bin/env python3
"""Automated architectural & safety guardrail verifier for Agent-first-browse.

Zero-dependency (Python 3.11+ stdlib only) checker enforcing:
  1. Frozen §6 preservation files remain unmodified (docs/AUDIT_REPORT_agent_first_browse.md §6).
  2. UNIVERSAL-ONLY rule in src/perception_engine.py and src/v2_engine/perception.py.
  3. Public package __all__ barrels & pyproject.toml CLI script entrypoints resolve cleanly.
  4. Raw secret / credential leak detection across repository files (redacted output only).
  5. Test-suite hygiene (every tests/test_*.py defines real pytest test functions, with
     explicitly tracked legacy debt warnings and zero tolerance for new empty test files).

Usage:
    .venv/bin/python scripts/check_guardrails.py
    .venv/bin/python scripts/check_guardrails.py --base-ref main
    .venv/bin/python scripts/check_guardrails.py --strict-test-hygiene --json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

Severity = Literal["error", "warning"]

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]

# §6 Preservation Constraints — DO NOT TOUCH (docs/AUDIT_REPORT_agent_first_browse.md)
FROZEN_FILES: tuple[str, ...] = (
    "src/dom_parser.py",
    "src/cdp_click.py",
    "src/cdp_input.py",
    "src/ghost_input.py",
    "src/app/browser_promoter/cdp_stealth_launcher.py",
    "src/virtual_display.py",
)

# Perception files subject to the UNIVERSAL-ONLY rule (no site/brand literals)
UNIVERSAL_ONLY_FILES: tuple[str, ...] = (
    "src/perception_engine.py",
    "src/v2_engine/perception.py",
)

# Public packages whose __init__.py must declare and resolve explicit __all__ barrels
PUBLIC_EXPORT_PACKAGES: tuple[str, ...] = (
    "v2_engine",
    "eval_connector",
)

# Future core package automatically verified once src/core/__init__.py exists
OPTIONAL_EXPORT_PACKAGES: tuple[str, ...] = ("core",)

# Audited pre-existing zero-test legacy script files (scheduled for cleanup in Chunk 8)
KNOWN_LEGACY_SCRIPT_TESTS: frozenset[str] = frozenset(
    {
        "tests/test_brain_v16.py",
        "tests/test_v151_blacklist.py",
        "tests/test_form_fill.py",
        "tests/test_read_only.py",
    }
)

# Forbidden site/brand/domain tokens in UNIVERSAL-ONLY perception files
_FORBIDDEN_BRAND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("amazon", re.compile(r"(?<![A-Za-z0-9_])amazon(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("flipkart", re.compile(r"(?<![A-Za-z0-9_])flipkart(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("reddit", re.compile(r"(?<![A-Za-z0-9_])reddit(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("github", re.compile(r"(?<![A-Za-z0-9_])github(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("twitter", re.compile(r"(?<![A-Za-z0-9_])twitter(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("youtube", re.compile(r"(?<![A-Za-z0-9_])youtube(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("linkedin", re.compile(r"(?<![A-Za-z0-9_])linkedin(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("facebook", re.compile(r"(?<![A-Za-z0-9_])facebook(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("instagram", re.compile(r"(?<![A-Za-z0-9_])instagram(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("hashnode", re.compile(r"(?<![A-Za-z0-9_])hashnode(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("meesho", re.compile(r"(?<![A-Za-z0-9_])meesho(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("saucedemo", re.compile(r"(?<![A-Za-z0-9_])saucedemo(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("chatgpt", re.compile(r"(?<![A-Za-z0-9_])chatgpt(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("imdb", re.compile(r"(?<![A-Za-z0-9_])imdb(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("walmart", re.compile(r"(?<![A-Za-z0-9_])walmart(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("netflix", re.compile(r"(?<![A-Za-z0-9_])netflix(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("spotify", re.compile(r"(?<![A-Za-z0-9_])spotify(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("stackoverflow", re.compile(r"(?<![A-Za-z0-9_])stackoverflow(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("wikipedia", re.compile(r"(?<![A-Za-z0-9_])wikipedia(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("dev.to", re.compile(r"(?<![A-Za-z0-9_])dev\.to(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("google.com", re.compile(r"(?<![A-Za-z0-9_])google\.com(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("bing.com", re.compile(r"(?<![A-Za-z0-9_])bing\.com(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("x.com", re.compile(r"(?<![A-Za-z0-9_])x\.com(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("ebay.com", re.compile(r"(?<![A-Za-z0-9_])ebay\.com(?![A-Za-z0-9_])", re.IGNORECASE)),
)

# High-signal raw secret patterns (never echo matched raw values)
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_api_key", re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{20,}\b")),
    ("github_pat", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("groq_api_key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("nvidia_api_key", re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b")),
    (
        "private_key_header",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
    ),
    (
        "url_embedded_credentials",
        re.compile(r"https?://[^/\s:@]+:[^/\s@]{12,}@(?:github\.com|gitlab\.com|bitbucket\.org|api\.)"),
    ),
)

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".venv-windows",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "logs",
        "persistence",
        "dist",
        "build",
    }
)

_SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".exe",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".woff",
        ".woff2",
        ".ttf",
    }
)


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    path: str
    message: str
    line: int | None = None


@dataclass
class CheckResult:
    name: str
    passed: bool
    checked_count: int
    findings: list[Finding] = field(default_factory=list)


@dataclass
class GuardrailReport:
    repo_root: str
    passed: bool
    results: list[CheckResult]

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results for f in r.findings if f.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results for f in r.findings if f.severity == "warning")

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "results": [asdict(r) for r in self.results],
        }


def _norm_rel_path(path_str: str) -> str:
    return path_str.replace("\\", "/").lstrip("./")


def _redact_match(raw: str) -> str:
    if raw.startswith("-----BEGIN"):
        return "<PRIVATE_KEY_HEADER>"
    prefix = raw[:4] if len(raw) >= 8 else raw[:2]
    return f"{prefix}****(len={len(raw)})"


def _run_git(repo_root: Path, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _collect_git_changed_files(
    repo_root: Path,
    base_ref: str | None,
) -> tuple[set[str], list[Finding]]:
    findings: list[Finding] = []
    changed: set[str] = set()

    code, _, _ = _run_git(repo_root, ["rev-parse", "--is-inside-work-tree"])
    if code != 0:
        sev: Severity = "error" if base_ref else "warning"
        findings.append(
            Finding(
                check="frozen_files",
                severity=sev,
                path=".",
                message="Not a git repository; cannot compute git diff for frozen files.",
            )
        )
        return changed, findings

    for diff_args in (["diff", "--name-only"], ["diff", "--cached", "--name-only"]):
        rc, out, err = _run_git(repo_root, diff_args)
        if rc != 0:
            findings.append(
                Finding(
                    check="frozen_files",
                    severity="error",
                    path=".",
                    message=f"git {' '.join(diff_args)} failed: {err.strip()}",
                )
            )
        else:
            changed.update(_norm_rel_path(line) for line in out.splitlines() if line.strip())

    if base_ref:
        rc, out, err = _run_git(repo_root, ["diff", "--name-only", f"{base_ref}...HEAD"])
        if rc != 0:
            findings.append(
                Finding(
                    check="frozen_files",
                    severity="error",
                    path=".",
                    message=f"git diff against base-ref '{base_ref}' failed: {err.strip()}",
                )
            )
        else:
            changed.update(_norm_rel_path(line) for line in out.splitlines() if line.strip())

    return changed, findings


# ── Check 1: Frozen §6 Preservation Files ──────────────────────────────────────


def check_frozen_files(
    repo_root: Path,
    *,
    base_ref: str | None = None,
    changed_files: Iterable[str] | None = None,
    frozen_files: Iterable[str] = FROZEN_FILES,
) -> CheckResult:
    frozen = tuple(_norm_rel_path(p) for p in frozen_files)
    frozen_set = set(frozen)
    findings: list[Finding] = []

    for rel in frozen:
        if not (repo_root / rel).is_file():
            findings.append(
                Finding(
                    check="frozen_files",
                    severity="error",
                    path=rel,
                    message="Frozen §6 module is missing (deleted or moved).",
                )
            )

    if changed_files is not None:
        modified = {_norm_rel_path(p) for p in changed_files if p.strip()}
    else:
        modified, git_findings = _collect_git_changed_files(repo_root, base_ref)
        findings.extend(git_findings)

    for rel in sorted(modified & frozen_set):
        findings.append(
            Finding(
                check="frozen_files",
                severity="error",
                path=rel,
                message=(
                    "Frozen §6 legacy module was modified "
                    "(see docs/AUDIT_REPORT_agent_first_browse.md §6)."
                ),
            )
        )

    passed = not any(f.severity == "error" for f in findings)
    return CheckResult(
        name="frozen_files",
        passed=passed,
        checked_count=len(frozen),
        findings=findings,
    )


# ── Check 2: UNIVERSAL-ONLY Perception Guard ───────────────────────────────────


def _scan_universal_file(file_path: Path, rel: str) -> list[Finding]:
    findings: list[Finding] = []
    if not file_path.is_file():
        return [
            Finding(
                check="universal_only",
                severity="error",
                path=rel,
                message="UNIVERSAL-ONLY target file is missing.",
            )
        ]

    source = file_path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=rel)
    except SyntaxError as exc:
        return [
            Finding(
                check="universal_only",
                severity="error",
                path=rel,
                line=exc.lineno,
                message=f"SyntaxError while parsing file: {exc.msg}",
            )
        ]

    for lineno, line_text in enumerate(source.splitlines(), start=1):
        for label, pattern in _FORBIDDEN_BRAND_PATTERNS:
            if pattern.search(line_text):
                findings.append(
                    Finding(
                        check="universal_only",
                        severity="error",
                        path=rel,
                        line=lineno,
                        message=(
                            f"Forbidden site/brand token '{label}' violates "
                            "UNIVERSAL-ONLY perception rule."
                        ),
                    )
                )
    return findings


def check_universal_only(
    repo_root: Path,
    *,
    target_files: Iterable[str] = UNIVERSAL_ONLY_FILES,
) -> CheckResult:
    targets = tuple(_norm_rel_path(p) for p in target_files)
    findings: list[Finding] = []
    for rel in targets:
        findings.extend(_scan_universal_file(repo_root / rel, rel))

    passed = not any(f.severity == "error" for f in findings)
    return CheckResult(
        name="universal_only",
        passed=passed,
        checked_count=len(targets),
        findings=findings,
    )


# ── Check 3: Public __all__ Barrels & pyproject.toml Script Entrypoints ────────


def _find_all_node(tree: ast.Module) -> ast.AST | None:
    all_node: ast.AST | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    all_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                all_node = node.value
    return all_node


def _parse_all_entries(
    all_node: ast.List | ast.Tuple,
    rel: str,
) -> tuple[list[str], list[Finding]]:
    names: list[str] = []
    findings: list[Finding] = []
    seen: set[str] = set()

    for elt in all_node.elts:
        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str) or not elt.value.strip():
            findings.append(
                Finding(
                    check="exports",
                    severity="error",
                    path=rel,
                    line=getattr(elt, "lineno", None),
                    message="__all__ must contain only non-empty string literals.",
                )
            )
            continue
        if elt.value in seen:
            findings.append(
                Finding(
                    check="exports",
                    severity="error",
                    path=rel,
                    line=getattr(elt, "lineno", None),
                    message=f"Duplicate symbol '{elt.value}' in __all__.",
                )
            )
        seen.add(elt.value)
        names.append(elt.value)

    if not names:
        findings.append(
            Finding(
                check="exports",
                severity="error",
                path=rel,
                message="__all__ must not be empty.",
            )
        )
    return names, findings


def _extract_static_all(init_path: Path, rel: str) -> tuple[list[str] | None, list[Finding]]:
    if not init_path.is_file():
        return None, [
            Finding(
                check="exports",
                severity="error",
                path=rel,
                message="Public package __init__.py is missing.",
            )
        ]

    source = init_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        return None, [
            Finding(
                check="exports",
                severity="error",
                path=rel,
                line=exc.lineno,
                message=f"SyntaxError in __init__.py: {exc.msg}",
            )
        ]

    all_node = _find_all_node(tree)
    if not isinstance(all_node, (ast.List, ast.Tuple)):
        return None, [
            Finding(
                check="exports",
                severity="error",
                path=rel,
                message="Public package must define an explicit static __all__ list/tuple.",
            )
        ]

    return _parse_all_entries(all_node, rel)


def _load_pyproject_scripts(repo_root: Path) -> tuple[dict[str, str], list[Finding]]:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return {}, []
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [
            Finding(
                check="exports",
                severity="error",
                path="pyproject.toml",
                message=f"Failed to parse pyproject.toml: {exc}",
            )
        ]
    raw_scripts = data.get("project", {}).get("scripts", {})
    if not isinstance(raw_scripts, dict):
        return {}, [
            Finding(
                check="exports",
                severity="error",
                path="pyproject.toml",
                message="[project.scripts] must be a table.",
            )
        ]
    return {str(k): str(v) for k, v in raw_scripts.items()}, []


def _verify_runtime_imports(
    repo_root: Path,
    package_exports: dict[str, list[str]],
    script_specs: dict[str, str],
) -> list[Finding]:
    """Import packages and CLI script targets in a clean child interpreter."""
    payload = json.dumps({"packages": package_exports, "scripts": script_specs})
    probe_code = "\n".join(
        [
            "import importlib, json, sys",
            "spec = json.loads(sys.argv[1])",
            "errors = []",
            "for pkg, names in spec['packages'].items():",
            "    pkg_path = f'src/{pkg}/__init__.py'",
            "    try:",
            "        mod = importlib.import_module(pkg)",
            "    except Exception as exc:",
            "        msg = f'Import failed: {type(exc).__name__}: {exc}'",
            "        errors.append({'path': pkg_path, 'message': msg})",
            "        continue",
            "    for name in names:",
            "        if not hasattr(mod, name):",
            "            msg = f'Exported symbol {name!r} in __all__ is not defined on module'",
            "            errors.append({'path': pkg_path, 'message': msg})",
            "for cli_name, target in spec['scripts'].items():",
            "    if ':' not in target:",
            "        msg = f'Invalid script entrypoint {cli_name}={target!r}'",
            "        errors.append({'path': 'pyproject.toml', 'message': msg})",
            "        continue",
            "    mod_name, attr = target.split(':', 1)",
            "    try:",
            "        mod = importlib.import_module(mod_name)",
            "        obj = getattr(mod, attr)",
            "        if not callable(obj):",
            "            msg = f'Script {cli_name} target {target!r} is not callable'",
            "            errors.append({'path': 'pyproject.toml', 'message': msg})",
            "    except Exception as exc:",
            "        msg = f'Script {cli_name} ({target}) failed: {type(exc).__name__}: {exc}'",
            "        errors.append({'path': 'pyproject.toml', 'message': msg})",
            "print(json.dumps(errors))",
        ]
    )

    env = os.environ.copy()
    src_dir = str((repo_root / "src").resolve())
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pp}" if existing_pp else src_dir

    proc = subprocess.run(
        [sys.executable, "-c", probe_code, payload],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return [
            Finding(
                check="exports",
                severity="error",
                path="src",
                message=f"Export runtime verification subprocess failed: {proc.stderr.strip()}",
            )
        ]

    try:
        raw_errors = json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError as exc:
        return [
            Finding(
                check="exports",
                severity="error",
                path="src",
                message=f"Could not parse export verifier output: {exc}",
            )
        ]

    return [
        Finding(
            check="exports",
            severity="error",
            path=str(item.get("path", "src")),
            message=str(item.get("message", "Unknown export error")),
        )
        for item in raw_errors
    ]


def check_public_exports(
    repo_root: Path,
    *,
    packages: Iterable[str] | None = None,
    check_pyproject_scripts: bool = True,
) -> CheckResult:
    if packages is None:
        pkg_list = list(PUBLIC_EXPORT_PACKAGES)
        for opt in OPTIONAL_EXPORT_PACKAGES:
            if (repo_root / "src" / opt / "__init__.py").is_file():
                pkg_list.append(opt)
    else:
        pkg_list = list(packages)

    findings: list[Finding] = []
    package_exports: dict[str, list[str]] = {}

    for pkg in pkg_list:
        rel = f"src/{pkg}/__init__.py"
        names, pkg_findings = _extract_static_all(repo_root / rel, rel)
        findings.extend(pkg_findings)
        if names is not None and not pkg_findings:
            package_exports[pkg] = names

    script_specs: dict[str, str] = {}
    if check_pyproject_scripts:
        script_specs, script_findings = _load_pyproject_scripts(repo_root)
        findings.extend(script_findings)

    if package_exports or script_specs:
        findings.extend(_verify_runtime_imports(repo_root, package_exports, script_specs))

    passed = not any(f.severity == "error" for f in findings)
    return CheckResult(
        name="exports",
        passed=passed,
        checked_count=len(pkg_list) + len(script_specs),
        findings=findings,
    )


# ── Check 4: Raw Secret Leak Scanner ───────────────────────────────────────────


def _should_skip_secret_path(rel: str) -> bool:
    norm = _norm_rel_path(rel)
    parts = norm.split("/")
    if any(part in _SKIP_DIRS for part in parts):
        return True
    name = parts[-1]
    if name == ".env" or name.endswith(".env.local") or name == ".git-credentials":
        return True
    suffix = Path(name).suffix.lower()
    return suffix in _SKIP_EXTENSIONS


def _discover_files_for_secret_scan(repo_root: Path) -> list[str]:
    rc, _, _ = _run_git(repo_root, ["rev-parse", "--is-inside-work-tree"])
    if rc == 0:
        files: set[str] = set()
        for git_args in (
            ["ls-files"],
            ["ls-files", "--others", "--exclude-standard"],
        ):
            code, out, _ = _run_git(repo_root, git_args)
            if code == 0:
                files.update(_norm_rel_path(line) for line in out.splitlines() if line.strip())
        return sorted(f for f in files if not _should_skip_secret_path(f))

    discovered: list[str] = []
    for root, dirs, filenames in os.walk(repo_root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for fname in sorted(filenames):
            full = Path(root) / fname
            rel = _norm_rel_path(str(full.relative_to(repo_root)))
            if not _should_skip_secret_path(rel):
                discovered.append(rel)
    return discovered


def _scan_file_for_secrets(file_path: Path, rel: str) -> list[Finding]:
    if not file_path.is_file():
        return []
    try:
        raw_bytes = file_path.read_bytes()
    except OSError:
        return []
    if b"\x00" in raw_bytes[:8192]:
        return []

    text = raw_bytes.decode("utf-8", errors="ignore")
    findings: list[Finding] = []
    for lineno, line_text in enumerate(text.splitlines(), start=1):
        for secret_kind, pattern in _SECRET_PATTERNS:
            for match in pattern.finditer(line_text):
                findings.append(
                    Finding(
                        check="secrets",
                        severity="error",
                        path=rel,
                        line=lineno,
                        message=(
                            f"Potential raw secret ({secret_kind}) detected: "
                            f"{_redact_match(match.group(0))}"
                        ),
                    )
                )
    return findings


def check_secret_leaks(
    repo_root: Path,
    *,
    scan_files: Iterable[str] | None = None,
) -> CheckResult:
    if scan_files is not None:
        targets = sorted(_norm_rel_path(p) for p in scan_files if not _should_skip_secret_path(p))
    else:
        targets = _discover_files_for_secret_scan(repo_root)

    findings: list[Finding] = []
    for rel in targets:
        findings.extend(_scan_file_for_secrets(repo_root / rel, rel))

    passed = not any(f.severity == "error" for f in findings)
    return CheckResult(
        name="secrets",
        passed=passed,
        checked_count=len(targets),
        findings=findings,
    )


# ── Check 5: Test-Suite Hygiene ────────────────────────────────────────────────


def _count_pytest_functions(tree: ast.Module) -> int:
    count = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                count += 1
        elif isinstance(node, ast.ClassDef):
            is_test_cls = node.name.startswith("Test") or any(
                (isinstance(base, ast.Name) and base.id == "TestCase")
                or (isinstance(base, ast.Attribute) and base.attr == "TestCase")
                for base in node.bases
            )
            if is_test_cls:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith(
                        "test_"
                    ):
                        count += 1
    return count


def _discover_test_files(repo_root: Path) -> list[str]:
    tests_dir = repo_root / "tests"
    if not tests_dir.is_dir():
        return []
    found: list[str] = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        rel = _norm_rel_path(str(path.relative_to(repo_root)))
        parts = rel.split("/")
        if "live" in parts or "fakes" in parts or "__pycache__" in parts:
            continue
        found.append(rel)
    return found


def check_test_hygiene(
    repo_root: Path,
    *,
    strict: bool = False,
    known_legacy_debt: Iterable[str] = KNOWN_LEGACY_SCRIPT_TESTS,
) -> CheckResult:
    debt = {_norm_rel_path(p) for p in known_legacy_debt}
    test_files = _discover_test_files(repo_root)
    findings: list[Finding] = []
    seen_zero_test_debt: set[str] = set()

    for rel in test_files:
        full = repo_root / rel
        try:
            tree = ast.parse(full.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as exc:
            findings.append(
                Finding(
                    check="test_hygiene",
                    severity="error",
                    path=rel,
                    line=exc.lineno,
                    message=f"SyntaxError in test file: {exc.msg}",
                )
            )
            continue

        test_count = _count_pytest_functions(tree)
        if test_count == 0:
            if rel in debt:
                seen_zero_test_debt.add(rel)
                findings.append(
                    Finding(
                        check="test_hygiene",
                        severity="error" if strict else "warning",
                        path=rel,
                        message=(
                            "Tracked legacy script-style test file defines 0 pytest test_* "
                            "functions (scheduled for migration in Chunk 8)."
                        ),
                    )
                )
            else:
                findings.append(
                    Finding(
                        check="test_hygiene",
                        severity="error",
                        path=rel,
                        message=(
                            "Test file defines 0 pytest test_* functions "
                            "(empty or import-time assert script)."
                        ),
                    )
                )

    for stale in sorted(debt - seen_zero_test_debt):
        findings.append(
            Finding(
                check="test_hygiene",
                severity="warning",
                path=stale,
                message=(
                    "Stale entry in KNOWN_LEGACY_SCRIPT_TESTS: file now has pytest "
                    "tests or was moved/removed — remove it from the debt allowlist."
                ),
            )
        )

    passed = not any(f.severity == "error" for f in findings)
    return CheckResult(
        name="test_hygiene",
        passed=passed,
        checked_count=len(test_files),
        findings=findings,
    )


# ── Orchestrator & CLI ─────────────────────────────────────────────────────────


def run_all_checks(
    repo_root: Path = DEFAULT_REPO_ROOT,
    *,
    base_ref: str | None = None,
    changed_files: Iterable[str] | None = None,
    strict_test_hygiene: bool = False,
) -> GuardrailReport:
    root = Path(repo_root).resolve()
    results = [
        check_frozen_files(root, base_ref=base_ref, changed_files=changed_files),
        check_universal_only(root),
        check_public_exports(root),
        check_secret_leaks(root),
        check_test_hygiene(root, strict=strict_test_hygiene),
    ]
    overall_passed = all(r.passed for r in results)
    return GuardrailReport(repo_root=str(root), passed=overall_passed, results=results)


def _format_human_report(report: GuardrailReport) -> str:
    lines = [f"Guardrail check against: {report.repo_root}"]
    for res in report.results:
        status = "PASS" if res.passed else "FAIL"
        warns = sum(1 for f in res.findings if f.severity == "warning")
        errs = sum(1 for f in res.findings if f.severity == "error")
        lines.append(
            f"  [{status}] {res.name:<15} checked={res.checked_count:<4} "
            f"errors={errs} warnings={warns}"
        )
        for f in res.findings:
            loc = f"{f.path}:{f.line}" if f.line is not None else f.path
            lines.append(f"         {f.severity.upper():<7} {loc} — {f.message}")

    summary = "PASSED" if report.passed else "FAILED"
    lines.append(
        f"Result: {summary} (errors={report.error_count}, warnings={report.warning_count})"
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify Agent-first-browse architectural and safety guardrails."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root directory to inspect (default: project root).",
    )
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Optional git base ref (e.g. 'main') to also check committed frozen-file diffs.",
    )
    parser.add_argument(
        "--changed-files",
        nargs="*",
        default=None,
        help="Optional explicit list of changed relative file paths for frozen-file verification.",
    )
    parser.add_argument(
        "--strict-test-hygiene",
        action="store_true",
        help="Treat tracked legacy zero-test files as hard errors instead of warnings.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_all_checks(
        args.repo_root,
        base_ref=args.base_ref,
        changed_files=args.changed_files,
        strict_test_hygiene=args.strict_test_hygiene,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(_format_human_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
