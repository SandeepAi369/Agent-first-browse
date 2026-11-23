"""Integration tests for Chunk 2 (Stateful CLI Daemon & `scripts/agent_eval.py`).

Verifies:
  - Daemon auto-spawn, Unix socket `0600` permissions, PID file lifecycle,
    stale socket cleanup, and idle auto-shutdown.
  - Full multi-invocation CLI workflow via `scripts/agent_eval.py`:
    `status` → `scenarios` → `start-pair` → `sessions` → `compare` →
    `perceive` → `act` (positional shorthand + `--auto-perceive`) →
    `load-scenario` → `screenshot` → `report --json-out` → `stop --daemon`.
  - Clean error handling for unknown sessions and malformed daemon methods
    without crashing the daemon process.

Run: .venv/bin/python -m pytest tests/test_eval_connector_cli.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from eval_connector.daemon import (
    ensure_daemon_running,
    is_daemon_alive,
    pid_path_for,
    send_daemon_request,
    stop_daemon,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "agent_eval.py"


def _make_short_sock() -> tuple[Path, Path]:
    """Create a short /tmp directory so Unix socket path stays well under 108 bytes."""
    d = Path(tempfile.mkdtemp(prefix="afb_cli_"))
    return d, d / "eval.sock"


def _run_cli(sock: Path, *args: str, parse_json: bool = True) -> tuple[int, dict | str, str]:
    cmd = [sys.executable, str(CLI), "--socket", str(sock)]
    if parse_json:
        cmd.append("--json")
    cmd.extend(args)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    if parse_json and proc.returncode == 0 and proc.stdout.strip():
        return proc.returncode, json.loads(proc.stdout), proc.stderr
    return proc.returncode, proc.stdout, proc.stderr


def _find_ref(elements: list[dict], needle: str, kind: str | None = None) -> str:
    low = needle.lower()
    for el in elements:
        if kind and (el.get("kind") or "").lower() != kind:
            continue
        if low in (el.get("text") or "").lower():
            return str(el["id"])
    raise AssertionError(f"Element '{needle}' (kind={kind}) not found in {[e.get('text') for e in elements]}")


def test_daemon_socket_permissions_stale_recovery_and_idle_shutdown():
    tmp_dir, sock = _make_short_sock()
    try:
        # Create a dead/stale Unix socket file first to verify stale cleanup on start
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(sock))
        stale.close()
        assert sock.exists()

        info = ensure_daemon_running(sock, idle_timeout=1.5, startup_timeout=10.0)
        assert info["ok"] is True
        assert info["pid"] > 0
        assert pid_path_for(sock).exists()

        # Socket permissions must be owner-only (0600)
        mode = stat.S_IMODE(os.stat(sock).st_mode)
        assert mode == 0o600, oct(mode)

        # Wait for idle auto-shutdown to clean up socket and PID file
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and sock.exists():
            time.sleep(0.1)
        assert not sock.exists()
        assert not pid_path_for(sock).exists()
        assert is_daemon_alive(sock) is None
    finally:
        stop_daemon(sock)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_cli_end_to_end_persistent_dual_mode_workflow():
    tmp_dir, sock = _make_short_sock()
    try:
        # 1. Initially stopped
        rc, st0, _ = _run_cli(sock, "status")
        assert rc == 0 and st0["running"] is False

        # 2. Scenarios auto-spawns daemon and lists all 5 deep scenarios
        rc, sc_res, _ = _run_cli(sock, "scenarios")
        assert rc == 0
        assert len(sc_res["scenarios"]) >= 5

        # 3. Start twin legacy + v2 sessions on modal_occlusion
        rc, pair, _ = _run_cli(sock, "start-pair", "modal", "--scenario", "modal_occlusion")
        assert rc == 0 and pair["ok"] is True

        rc, sess, _ = _run_cli(sock, "sessions")
        assert rc == 0 and len(sess["sessions"]) == 2

        # 4. Side-by-side compare across persistent sessions
        rc, cmp_res, _ = _run_cli(sock, "compare", "modal_legacy", "modal_v2")
        assert rc == 0
        sc = cmp_res["step_comparison"]
        assert sc["v2_occluded_filtered"] >= 30
        assert sc["token_savings_pct"] >= 70.0

        # 5. Act across separate CLI invocations (proving stateful browser persistence!)
        rc, perc, _ = _run_cli(sock, "perceive", "modal_v2", "--no-settle")
        assert rc == 0
        accept_ref = _find_ref(perc["elements"], "Accept Essential Cookies", kind="button")

        rc, act_res, _ = _run_cli(sock, "act", "modal_v2", "click", accept_ref, "--auto-perceive")
        assert rc == 0 and act_res["ok"] is True
        assert act_res["perception"]["element_count"] >= 20

        # 6. Load multistep_form & run positional shorthand actions (type, select_option, click)
        rc, load_res, _ = _run_cli(sock, "load-scenario", "modal_v2", "multistep_form")
        assert rc == 0 and load_res["ok"] is True

        rc, p1, _ = _run_cli(sock, "perceive", "modal_v2", "--no-settle")
        assert rc == 0 and "TopSecret#987" not in p1["markdown"]
        name_ref = _find_ref(p1["elements"], "Full Name", kind="input")
        plan_ref = _find_ref(p1["elements"], "Plan", kind="input")
        cont_ref = _find_ref(p1["elements"], "Continue to Review", kind="button")

        rc, r_t, _ = _run_cli(sock, "act", "modal_v2", "type", name_ref, "Sandeep")
        assert rc == 0 and r_t["ok"] is True

        rc, r_s, _ = _run_cli(sock, "act", "modal_v2", "select_option", plan_ref, "Pro Team")
        assert rc == 0 and r_s["ok"] is True

        rc, r_c, _ = _run_cli(sock, "act", "modal_v2", "click", cont_ref, "--auto-perceive")
        assert rc == 0 and r_c["ok"] is True
        assert r_c["perception"]["metrics"]["snapshot_kind"] == "delta"
        assert "TopSecret#987" not in r_c["perception"]["markdown"]

        # 7. Screenshot + Report export
        shot_path = tmp_dir / "modal_v2.png"
        rc, shot, _ = _run_cli(sock, "screenshot", "modal_v2", "--out", str(shot_path))
        assert rc == 0 and shot["ok"] is True
        assert shot_path.exists() and shot["bytes"] > 500

        rep_path = tmp_dir / "eval_report.json"
        rc, rep, _ = _run_cli(sock, "report", "--json-out", str(rep_path))
        assert rc == 0 and rep["ok"] is True
        assert rep_path.exists()
        saved = json.loads(rep_path.read_text(encoding="utf-8"))
        assert "modal_legacy" in saved["sessions"] and "modal_v2" in saved["sessions"]

        # 8. Human-readable output smoke check (without --json)
        rc, text_out, _ = _run_cli(sock, "report", parse_json=False)
        assert rc == 0 and "| `modal_v2` | v2 |" in text_out

        # 9. Clean daemon shutdown
        rc, stop_res, _ = _run_cli(sock, "stop", "--daemon")
        assert rc == 0 and stop_res.get("shutdown") is True
        assert not sock.exists()

        rc, st_end, _ = _run_cli(sock, "status")
        assert rc == 0 and st_end["running"] is False
    finally:
        stop_daemon(sock)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_cli_and_daemon_error_resilience():
    tmp_dir, sock = _make_short_sock()
    try:
        ensure_daemon_running(sock, idle_timeout=60.0)

        # Unknown session returns non-zero exit code without killing daemon
        rc, _, stderr = _run_cli(sock, "perceive", "missing_session", parse_json=False)
        assert rc == 1
        assert "missing_session" in stderr

        # Unknown daemon method returns structured error envelope
        bad = send_daemon_request(sock, "nonexistent_method", {})
        assert bad["ok"] is False
        assert "Unknown daemon method" in bad["error"]

        # Daemon is still healthy after errors
        assert is_daemon_alive(sock) is not None
    finally:
        stop_daemon(sock)
        shutil.rmtree(tmp_dir, ignore_errors=True)
