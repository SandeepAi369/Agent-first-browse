"""eval_connector.daemon — Persistent Unix-Socket Daemon for Zero-API Agent Evaluation.

Decouples live Chromium lifecycle from individual CLI commands (following the
2026 Playwright CLI / Vercel `agent-browser` daemon pattern) so cookies, DOM
state, `window.__aid`, and V2 delta history persist across step-by-step commands.

Protocol:
  Line-delimited JSON (NDJSON) over a user-private (`0600`) Unix domain socket:
    Request:  {"method": "<name>", "params": {...}}\n
    Response: {"ok": true, "data": {...}}\n  |  {"ok": false, "error": "..."}\n
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eval_connector.scenarios import list_scenarios
from eval_connector.session import DualModeSessionManager

logger = logging.getLogger("eval_connector.daemon")

_STREAM_LIMIT = 16 * 1024 * 1024  # 16 MB max single NDJSON line
_DEFAULT_IDLE_TIMEOUT = 1800.0  # 30 minutes


def default_socket_path() -> Path:
    """Return the default Unix domain socket path (kept short for 108-byte sun_path limit)."""
    env_sock = os.environ.get("AGENT_EVAL_SOCKET", "").strip()
    if env_sock:
        return Path(env_sock)
    return Path.home() / ".cache" / "afb" / "eval.sock"


def pid_path_for(sock_path: Path) -> Path:
    return sock_path.with_suffix(".pid")


def log_path_for(sock_path: Path) -> Path:
    return sock_path.with_suffix(".log")


class EvalDaemonServer:
    """Asyncio Unix-socket server hosting one `DualModeSessionManager`."""

    def __init__(
        self,
        sock_path: Path,
        *,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        screenshot_dir: Path | None = None,
    ) -> None:
        self.sock_path = Path(sock_path)
        self.pid_path = pid_path_for(self.sock_path)
        self.idle_timeout = max(0.0, float(idle_timeout))
        self.manager = DualModeSessionManager(screenshot_dir=screenshot_dir)
        self._server: asyncio.AbstractServer | None = None
        self._stop_event = asyncio.Event()
        self._started_at = time.monotonic()
        self._last_activity = self._started_at

    def _prepare_socket_path(self) -> None:
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.sock_path.parent, 0o700)
        except OSError:
            pass

        if self.sock_path.exists():
            # Probe whether an active daemon is already listening on this socket
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.5)
            try:
                probe.connect(str(self.sock_path))
                raise RuntimeError(f"Eval daemon already running at {self.sock_path}")
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                # Stale socket left over from an unclean exit — safe to remove
                try:
                    self.sock_path.unlink()
                except FileNotFoundError:
                    pass
            finally:
                probe.close()

    async def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._last_activity = time.monotonic()

        if method == "ping":
            sessions = await self.manager.list_sessions()
            return {
                "ok": True,
                "pid": os.getpid(),
                "socket": str(self.sock_path),
                "uptime_s": round(time.monotonic() - self._started_at, 1),
                "session_count": len(sessions),
            }

        if method == "scenarios":
            return {"ok": True, "scenarios": list_scenarios()}

        if method == "sessions":
            return {"ok": True, "sessions": await self.manager.list_sessions()}

        if method == "start":
            vp = params.get("viewport") or [1280, 800]
            return await self.manager.start_session(
                params["name"],
                params.get("mode", "v2"),
                url=params.get("url"),
                scenario=params.get("scenario"),
                headless=bool(params.get("headless", True)),
                viewport=(int(vp[0]), int(vp[1])),
            )

        if method == "start_pair":
            vp = params.get("viewport") or [1280, 800]
            return await self.manager.start_pair(
                params.get("prefix", "eval"),
                url=params.get("url"),
                scenario=params.get("scenario"),
                headless=bool(params.get("headless", True)),
                viewport=(int(vp[0]), int(vp[1])),
            )

        if method == "load_scenario":
            return await self.manager.load_scenario(params["name"], params["scenario"])

        if method == "perceive":
            return await self.manager.perceive(
                params["name"],
                objective=params.get("objective", ""),
                force_full=bool(params.get("force_full", False)),
                settle=bool(params.get("settle", True)),
            )

        if method == "act":
            return await self.manager.act(
                params["name"],
                params["verb"],
                element_id=params.get("element_id"),
                text=params.get("text"),
                url=params.get("url"),
                x=params.get("x"),
                y=params.get("y"),
                key=params.get("key"),
                direction=params.get("direction", "down"),
                amount=int(params.get("amount", 500)),
                wait_ms=int(params.get("wait_ms", 800)),
                auto_perceive=bool(params.get("auto_perceive", False)),
                objective=params.get("objective", ""),
            )

        if method == "compare":
            return await self.manager.compare(
                params["legacy_name"],
                params["v2_name"],
                objective=params.get("objective", ""),
                refresh=bool(params.get("refresh", True)),
                settle=bool(params.get("settle", True)),
            )

        if method == "screenshot":
            return await self.manager.screenshot(
                params["name"],
                out_path=params.get("out_path"),
                full_page=bool(params.get("full_page", False)),
            )

        if method == "report":
            return await self.manager.report(names=params.get("names"))

        if method == "close":
            return await self.manager.close_session(params["name"])

        if method == "close_all":
            return await self.manager.close_all()

        if method == "shutdown":
            closed = await self.manager.close_all()
            self._stop_event.set()
            return {"ok": True, "shutdown": True, "closed": closed.get("closed", [])}

        raise ValueError(f"Unknown daemon method: '{method}'")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=15.0)
            if not raw:
                return
            try:
                req = json.loads(raw.decode("utf-8"))
                if not isinstance(req, dict) or not isinstance(req.get("method"), str):
                    raise ValueError("Invalid request envelope: expected JSON object with 'method'")
                method = req["method"]
                params = req.get("params") or {}
                if not isinstance(params, dict):
                    raise ValueError("'params' must be a JSON object")
                data = await self._dispatch(method, params)
                resp: dict[str, Any] = {"ok": True, "data": data}
            except Exception as e:
                logger.debug("Daemon request error: %s", e)
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}

            payload = (json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8")
            writer.write(payload)
            await writer.drain()
        except Exception as e:
            logger.debug("Daemon connection error: %s", e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _idle_watchdog(self) -> None:
        if self.idle_timeout <= 0:
            return
        poll = min(1.0, max(0.05, self.idle_timeout / 4.0))
        while not self._stop_event.is_set():
            await asyncio.sleep(poll)
            if (time.monotonic() - self._last_activity) >= self.idle_timeout:
                logger.info("Idle timeout (%.1fs) reached — shutting down eval daemon", self.idle_timeout)
                self._stop_event.set()
                break

    async def serve_forever(self) -> None:
        self._prepare_socket_path()
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.sock_path),
            limit=_STREAM_LIMIT,
        )
        try:
            os.chmod(self.sock_path, 0o600)
        except OSError:
            pass
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        logger.info("Eval daemon listening on %s (pid=%d, idle=%.0fs)",
                    self.sock_path, os.getpid(), self.idle_timeout)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass

        watchdog = asyncio.create_task(self._idle_watchdog())
        try:
            await self._stop_event.wait()
        finally:
            watchdog.cancel()
            with _suppress_all():
                await watchdog
            if self._server is not None:
                self._server.close()
                with _suppress_all():
                    await self._server.wait_closed()
            with _suppress_all():
                await self.manager.close_all()
            for p in (self.sock_path, self.pid_path):
                with _suppress_all():
                    if p.exists():
                        p.unlink()
            logger.info("Eval daemon shut down cleanly.")


class _suppress_all:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Synchronous Client Helpers (used by scripts/agent_eval.py & tests)
# ═══════════════════════════════════════════════════════════════════════════════

def send_daemon_request(
    sock_path: Path,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Send a single NDJSON request to the running daemon and return its response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(sock_path))
        payload = (json.dumps({"method": method, "params": params or {}}, ensure_ascii=False) + "\n").encode("utf-8")
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)

        chunks: list[bytes] = []
        while True:
            buf = sock.recv(65536)
            if not buf:
                break
            chunks.append(buf)
            if b"\n" in buf:
                break
        raw = b"".join(chunks).decode("utf-8").strip()
        if not raw:
            return {"ok": False, "error": "Empty response from eval daemon"}
        return json.loads(raw)
    finally:
        sock.close()


def is_daemon_alive(sock_path: Path) -> dict[str, Any] | None:
    """Return ping payload if daemon is listening on `sock_path`, else None."""
    if not sock_path.exists():
        return None
    try:
        res = send_daemon_request(sock_path, "ping", timeout=2.0)
        if res.get("ok"):
            return res.get("data")
    except OSError:
        return None
    return None


def ensure_daemon_running(
    sock_path: Path,
    *,
    idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
    startup_timeout: float = 10.0,
) -> dict[str, Any]:
    """Ensure the eval daemon is running on `sock_path`, spawning it if needed."""
    alive = is_daemon_alive(sock_path)
    if alive is not None:
        return alive

    sock_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path_for(sock_path)
    src_dir = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{existing_pp}" if existing_pp else src_dir

    cmd = [
        sys.executable,
        "-m",
        "eval_connector.daemon",
        "--socket",
        str(sock_path),
        "--idle-timeout",
        str(idle_timeout),
    ]
    with open(log_file, "ab") as lf:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=lf,
            env=env,
            start_new_session=True,
        )

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        alive = is_daemon_alive(sock_path)
        if alive is not None:
            return alive
        if proc.poll() is not None:
            tail = ""
            try:
                tail = log_file.read_text(encoding="utf-8", errors="replace")[-400:]
            except OSError:
                pass
            raise RuntimeError(f"Eval daemon exited early (rc={proc.returncode}): {tail}")
        time.sleep(0.1)

    raise TimeoutError(f"Timed out waiting for eval daemon at {sock_path}")


def stop_daemon(sock_path: Path, timeout: float = 5.0) -> dict[str, Any]:
    """Gracefully stop the running daemon on `sock_path`."""
    if not sock_path.exists():
        return {"ok": True, "already_stopped": True}
    try:
        res = send_daemon_request(sock_path, "shutdown", timeout=timeout)
    except OSError:
        # Stale socket — remove it cleanly
        with _suppress_all():
            sock_path.unlink()
        with _suppress_all():
            pid_path_for(sock_path).unlink()
        return {"ok": True, "already_stopped": True}

    deadline = time.monotonic() + timeout
    while sock_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return res.get("data", {"ok": True, "shutdown": True}) if res.get("ok") else res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent Agent-first-browse Eval Daemon")
    parser.add_argument("--socket", type=str, default=str(default_socket_path()))
    parser.add_argument("--idle-timeout", type=float, default=_DEFAULT_IDLE_TIMEOUT)
    parser.add_argument("--screenshot-dir", type=str, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    server = EvalDaemonServer(
        Path(args.socket),
        idle_timeout=args.idle_timeout,
        screenshot_dir=Path(args.screenshot_dir) if args.screenshot_dir else None,
    )
    asyncio.run(server.serve_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
