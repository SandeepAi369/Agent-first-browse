#!/usr/bin/env python3
"""Release version synchronization, drift verification, and tagging helper.

Keeps every versioned file in `Agent-first-browse` strictly aligned from one
command so PyPI, GitHub Releases, and the Official MCP Registry never drift:

  - `pyproject.toml`                      (`project.version`)
  - `server.json`                         (`version` + `packages[*].version`)
  - `src/app/__init__.py`                 (`__version__`)
  - `src/eval_connector/mcp_server.py`    (`SERVER_VERSION`)
  - `README.md`                           (`mcp-name: io.github.SandeepAi369/agent-first-browse`)

Usage:
    # Verify all files agree (used in CI on every PR and release)
    python scripts/release.py --check
    python scripts/release.py --check --expected-tag v0.1.2

    # Bump version across all files (patch | minor | major) or set explicitly
    python scripts/release.py --bump patch
    python scripts/release.py --version 0.2.0

    # Bump, commit (no Co-authored-by trailer), tag vX.Y.Z, and push to trigger CI release
    python scripts/release.py --bump patch --commit --tag --push
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER_NAME = "io.github.SandeepAi369/agent-first-browse"
MCP_OWNERSHIP_MARKER = f"mcp-name: {MCP_SERVER_NAME}"
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:([-.][0-9A-Za-z.-]+))?$")

_PYPROJECT_VER_RE = re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)
_APP_VER_RE = re.compile(r'^(__version__\s*=\s*")([^"]+)(")', re.MULTILINE)
_MCP_VER_RE = re.compile(r'^(SERVER_VERSION\s*=\s*")([^"]+)(")', re.MULTILINE)


@dataclass(frozen=True)
class VersionSnapshot:
    pyproject: str
    server_json: str
    server_json_packages: tuple[str, ...]
    app_init: str
    mcp_server: str
    has_readme_marker: bool


def validate_semver(version: str) -> str:
    """Validate and return a clean semantic version string (without leading `v`)."""
    cleaned = version.strip()
    if not SEMVER_RE.match(cleaned):
        raise ValueError(f"Invalid semantic version: {version!r} (expected X.Y.Z)")
    return cleaned


def compute_next_version(current: str, bump: str) -> str:
    """Compute next `patch`, `minor`, or `major` semantic version from `current`."""
    m = SEMVER_RE.match(current.strip())
    if not m:
        raise ValueError(f"Cannot bump non-semver version: {current!r}")
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "major":
        return f"{major + 1}.0.0"
    raise ValueError(f"Unsupported bump kind: {bump!r} (expected patch, minor, or major)")


def read_versions(root: Path = REPO_ROOT) -> VersionSnapshot:
    """Read all tracked version declarations and the README MCP ownership marker."""
    pyproject_data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    pyproject_ver = str(pyproject_data["project"]["version"])

    server_data = json.loads((root / "server.json").read_text(encoding="utf-8"))
    server_ver = str(server_data.get("version", ""))
    pkg_vers = tuple(
        str(pkg.get("version", ""))
        for pkg in (server_data.get("packages") or [])
        if isinstance(pkg, dict)
    )

    app_text = (root / "src" / "app" / "__init__.py").read_text(encoding="utf-8")
    app_match = _APP_VER_RE.search(app_text)
    app_ver = app_match.group(2) if app_match else ""

    mcp_text = (root / "src" / "eval_connector" / "mcp_server.py").read_text(encoding="utf-8")
    mcp_match = _MCP_VER_RE.search(mcp_text)
    mcp_ver = mcp_match.group(2) if mcp_match else ""

    readme_text = (root / "README.md").read_text(encoding="utf-8")
    has_marker = any(line.strip() == MCP_OWNERSHIP_MARKER for line in readme_text.splitlines())

    return VersionSnapshot(
        pyproject=pyproject_ver,
        server_json=server_ver,
        server_json_packages=pkg_vers,
        app_init=app_ver,
        mcp_server=mcp_ver,
        has_readme_marker=has_marker,
    )


def check_consistency(
    root: Path = REPO_ROOT,
    *,
    expected_tag: str | None = None,
) -> list[str]:
    """Return a list of drift/integrity errors (empty list means completely aligned)."""
    snap = read_versions(root)
    errors: list[str] = []

    try:
        canonical = validate_semver(snap.pyproject)
    except ValueError as exc:
        errors.append(f"pyproject.toml: {exc}")
        canonical = snap.pyproject

    if snap.server_json != canonical:
        errors.append(
            f"server.json top-level version {snap.server_json!r} != pyproject.toml {canonical!r}"
        )
    if not snap.server_json_packages:
        errors.append("server.json has no packages entries")
    else:
        for idx, pkg_ver in enumerate(snap.server_json_packages):
            if pkg_ver != canonical:
                errors.append(
                    f"server.json packages[{idx}].version {pkg_ver!r} != pyproject.toml {canonical!r}"
                )

    if snap.app_init != canonical:
        errors.append(
            f"src/app/__init__.py __version__ {snap.app_init!r} != pyproject.toml {canonical!r}"
        )
    if snap.mcp_server != canonical:
        errors.append(
            f"src/eval_connector/mcp_server.py SERVER_VERSION {snap.mcp_server!r} != pyproject.toml {canonical!r}"
        )
    if not snap.has_readme_marker:
        errors.append(
            f"README.md is missing the visible ownership line: {MCP_OWNERSHIP_MARKER!r}"
        )

    glama_path = root / "glama.json"
    if not glama_path.exists():
        errors.append("glama.json is missing at repository root")
    else:
        try:
            glama = json.loads(glama_path.read_text(encoding="utf-8"))
            if not glama.get("maintainers"):
                errors.append("glama.json has empty maintainers list")
        except json.JSONDecodeError as exc:
            errors.append(f"glama.json is invalid JSON: {exc}")

    if expected_tag is not None:
        tag_clean = expected_tag.strip()
        if not tag_clean.startswith("v"):
            errors.append(f"Expected release tag to start with 'v', got {tag_clean!r}")
        else:
            tag_ver = tag_clean[1:]
            if tag_ver != canonical:
                errors.append(
                    f"Release tag {tag_clean!r} ({tag_ver}) does not match pyproject.toml version {canonical!r}"
                )

    return errors


def _replace_once(path: Path, pattern: re.Pattern[str], new_version: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = pattern.subn(rf"\g<1>{new_version}\g<3>", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not locate version declaration in {path}")
    path.write_text(updated, encoding="utf-8", newline="\n")


def apply_version(new_version: str, root: Path = REPO_ROOT) -> list[str]:
    """Update all versioned files atomically to `new_version` and return modified relpaths."""
    ver = validate_semver(new_version)

    _replace_once(root / "pyproject.toml", _PYPROJECT_VER_RE, ver)
    _replace_once(root / "src" / "app" / "__init__.py", _APP_VER_RE, ver)
    _replace_once(root / "src" / "eval_connector" / "mcp_server.py", _MCP_VER_RE, ver)

    server_path = root / "server.json"
    server_data = json.loads(server_path.read_text(encoding="utf-8"))
    server_data["version"] = ver
    for pkg in server_data.get("packages") or []:
        if isinstance(pkg, dict):
            pkg["version"] = ver
    server_path.write_text(json.dumps(server_data, indent=2) + "\n", encoding="utf-8", newline="\n")

    return [
        "pyproject.toml",
        "server.json",
        "src/app/__init__.py",
        "src/eval_connector/mcp_server.py",
    ]


def _run_git(args: list[str], root: Path) -> None:
    subprocess.run(["git", *args], cwd=root, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize and validate release versions across pyproject.toml, server.json, and src/.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Verify all version files and markers match.")
    mode.add_argument(
        "--bump",
        choices=("patch", "minor", "major"),
        help="Increment current pyproject.toml version across all tracked files.",
    )
    mode.add_argument("--version", dest="target_version", help="Set an explicit X.Y.Z version across all tracked files.")
    mode.add_argument("--print-version", action="store_true", help="Print the canonical version from pyproject.toml.")

    parser.add_argument(
        "--expected-tag",
        help="With --check, assert that `v<version>` matches this git tag (e.g. v0.1.2).",
    )
    parser.add_argument("--commit", action="store_true", help="Commit bumped files without any Co-authored-by trailer.")
    parser.add_argument("--tag", action="store_true", help="Create annotated git tag vX.Y.Z.")
    parser.add_argument("--push", action="store_true", help="Push main branch and tag vX.Y.Z to origin.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    root: Path = args.repo_root.resolve()

    if args.print_version:
        print(read_versions(root).pyproject)
        return 0

    if args.check:
        errors = check_consistency(root, expected_tag=args.expected_tag)
        if errors:
            print("Release version consistency check FAILED:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        snap = read_versions(root)
        print(f"Release version consistency check PASSED (version={snap.pyproject})")
        return 0

    current = read_versions(root).pyproject
    target = compute_next_version(current, args.bump) if args.bump else validate_semver(args.target_version)
    changed = apply_version(target, root)
    errors = check_consistency(root)
    if errors:
        print("Post-bump validation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Updated version {current} -> {target} across: {', '.join(changed)}")
    tag_name = f"v{target}"

    if args.commit:
        _run_git(["add", *changed], root)
        _run_git(["commit", "-m", f"release({tag_name}): bump version to {target}"], root)
    if args.tag:
        _run_git(["tag", "-a", tag_name, "-m", f"Release {tag_name}"], root)
    if args.push:
        _run_git(["push", "origin", "HEAD", tag_name], root)
        print(f"Pushed commit and tag {tag_name} to origin (automated release workflow will publish to PyPI, GitHub Releases, and MCP Registry).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
