"""eval_connector.mcp_server — Pure-Python Production Browser MCP stdio Server (Zero Extra Deps).

Implements the Model Context Protocol (JSON-RPC 2.0 over newline-delimited stdio)
using only Python stdlib + `DualModeSessionManager`, exposing 7 lean browser control
and diagnostic tools so any MCP host (Claude Desktop, Claude Code, Cursor, VS Code,
Antigravity, Gemini CLI) can drive real Chromium browsers directly with zero external
LLM API keys.

Primary production tools exposed via `tools/list` (7):
  1. `browser_start`      — Launch an isolated Chromium session (`v2`, `legacy`, or `pair`)
  2. `browser_perceive`   — Capture compact ref-indexed page state + step telemetry
  3. `browser_act`        — Execute browser action (`goto`, `click`, `type`, etc.)
  4. `browser_screenshot` — Save PNG screenshot to disk for visual verification
  5. `browser_stop`       — Close one or all active browser sessions
  6. `browser_compare`    — Side-by-side latency/token/occlusion/visibility diff
  7. `browser_metrics`    — Cumulative Markdown + JSON telemetry, sessions & scenarios

Backwards compatibility:
  All legacy `eval_*` tool names (`eval_start`, `eval_perceive`, `eval_act`,
  `eval_screenshot`, `eval_stop`, `eval_compare`, `eval_metrics`) remain fully
  routed in `tools/call` and `call_tool()`.

Run via console entrypoint:
    agent-browse-mcp
    agent-eval-mcp        # backwards-compatible alias
Or directly from source:
    PYTHONPATH=src python -m eval_connector.mcp_server
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from eval_connector.scenarios import list_scenarios
from eval_connector.session import DualModeSessionManager

logger = logging.getLogger("eval_connector.mcp_server")

SERVER_NAME = "agent-first-browse"
SERVER_VERSION = "0.1.1"
DEFAULT_PROTOCOL_VERSION = "2025-03-26"
SUPPORTED_PROTOCOL_VERSIONS = {
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
}

SERVER_INSTRUCTIONS = (
    "Production zero-API Browser MCP Server for Agent-first-browse. "
    "Use browser_start(url=...) to launch an isolated Chromium session (defaults to the fast v2 "
    "engine; also supports mode='legacy' or twin mode='pair'), browser_perceive to inspect "
    "compact ref-indexed elements and telemetry, browser_act to execute multi-step workflows, "
    "browser_screenshot for visual verification, browser_stop when finished, and "
    "browser_compare / browser_metrics for side-by-side engine diagnostics. "
    "Legacy eval_* tool aliases remain supported."
)

# ═══════════════════════════════════════════════════════════════════════════════
#  7 Production MCP Tool Definitions (+ Legacy eval_* Alias Map)
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "browser_start",
        "description": (
            "Launch an isolated Chromium browser session (defaults to fast 'v2' engine; "
            "also supports 'legacy' or twin 'pair' comparison mode). Accepts any live URL "
            "or built-in stress scenario (modal_occlusion, spa_rerender, sparse_aria, "
            "multistep_form, a11y_visibility)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Session name (or prefix when mode='pair'). Default: 'browser'.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["v2", "legacy", "pair"],
                    "description": "Engine mode: 'v2' (default, fast), 'legacy', or 'pair' (twin sessions).",
                },
                "url": {
                    "type": "string",
                    "description": "Optional live URL to open (e.g. https://news.ycombinator.com).",
                },
                "scenario": {
                    "type": "string",
                    "enum": [
                        "modal_occlusion",
                        "spa_rerender",
                        "sparse_aria",
                        "multistep_form",
                        "a11y_visibility",
                    ],
                    "description": "Optional built-in stress-test scenario to load.",
                },
                "headless": {
                    "type": "boolean",
                    "description": "Run headless Chromium (default true). Set false for headed/Xvfb.",
                },
                "viewport_width": {"type": "integer", "description": "Viewport width (default 1280)."},
                "viewport_height": {"type": "integer", "description": "Viewport height (default 800)."},
            },
            "required": [],
        },
    },
    {
        "name": "browser_perceive",
        "description": (
            "Inspect the active browser session with accessibility-tree-first perception and "
            "return compact ref-indexed elements ([e1], [e2], ...) plus latency/token telemetry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": {"type": "string", "description": "Active session name."},
                "objective": {"type": "string", "description": "Optional task objective for goal-relevance recall."},
                "force_full": {"type": "boolean", "description": "Force full snapshot (bypass V2 delta mode)."},
                "settle": {"type": "boolean", "description": "Wait for adaptive DOM settle first (default true)."},
            },
            "required": ["session"],
        },
    },
    {
        "name": "browser_act",
        "description": (
            "Execute a browser action inside an active session (navigate, click, type, hover, "
            "select, key press, scroll, or wait) with stale-ref semantic rebinding and optional "
            "auto_perceive."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": {"type": "string", "description": "Active session name."},
                "verb": {
                    "type": "string",
                    "enum": [
                        "goto", "click", "type", "hover", "select_option",
                        "press_enter", "press_key", "scroll", "scroll_directional", "wait",
                    ],
                    "description": "Action verb to execute.",
                },
                "element_id": {"type": "string", "description": "Target element ref ID (e.g. 'e5')."},
                "text": {"type": "string", "description": "Text to type or option value to select."},
                "url": {"type": "string", "description": "Target URL for 'goto'."},
                "x": {"type": "number", "description": "Optional fallback X coordinate."},
                "y": {"type": "number", "description": "Optional fallback Y coordinate."},
                "key": {"type": "string", "description": "Key or chord for 'press_key' (e.g. 'Escape')."},
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Direction for 'scroll_directional'.",
                },
                "amount": {"type": "integer", "description": "Scroll pixels (default 500)."},
                "wait_ms": {"type": "integer", "description": "Milliseconds for 'wait' (default 800)."},
                "auto_perceive": {"type": "boolean", "description": "Perceive immediately after acting."},
                "objective": {"type": "string", "description": "Optional objective passed to auto_perceive."},
            },
            "required": ["session", "verb"],
        },
    },
    {
        "name": "browser_screenshot",
        "description": (
            "Capture a PNG screenshot of an active browser session to disk and return its path, "
            "dimensions, and byte size."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": {"type": "string", "description": "Active session name."},
                "out_path": {"type": "string", "description": "Optional output PNG path."},
                "full_page": {"type": "boolean", "description": "Capture full scrollable page (default false)."},
            },
            "required": ["session"],
        },
    },
    {
        "name": "browser_stop",
        "description": "Close a single named browser session or shut down all active sessions cleanly.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": {
                    "type": "string",
                    "description": "Session name to close. Omit (or set all=true) to close all sessions.",
                },
                "all": {"type": "boolean", "description": "Close all active sessions (default false)."},
            },
            "required": [],
        },
    },
    {
        "name": "browser_compare",
        "description": (
            "Compare an active legacy session and v2 session side-by-side: latency speedup, "
            "prompt token savings %, occluded elements pruned, Tier-2 escalation, and visibility diff."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "legacy_session": {"type": "string", "description": "Name of the legacy session."},
                "v2_session": {"type": "string", "description": "Name of the v2 session."},
                "objective": {"type": "string", "description": "Optional objective for perception."},
                "refresh": {"type": "boolean", "description": "Re-perceive both sessions now (default true)."},
                "settle": {"type": "boolean", "description": "Run page settle during refresh (default true)."},
            },
            "required": ["legacy_session", "v2_session"],
        },
    },
    {
        "name": "browser_metrics",
        "description": (
            "Return cumulative session latency/token telemetry, Markdown comparison table, "
            "active sessions, and built-in stress scenario catalog."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sessions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of session names to include.",
                },
                "include_scenarios": {
                    "type": "boolean",
                    "description": "Include built-in scenario catalog in response (default true).",
                },
            },
            "required": [],
        },
    },
]

TOOL_ALIASES: dict[str, str] = {
    "eval_start": "browser_start",
    "eval_perceive": "browser_perceive",
    "eval_act": "browser_act",
    "eval_screenshot": "browser_screenshot",
    "eval_stop": "browser_stop",
    "eval_compare": "browser_compare",
    "eval_metrics": "browser_metrics",
}

_TOOL_NAMES: set[str] = {t["name"] for t in TOOL_DEFINITIONS}
_ALLOWED_TOOL_NAMES: frozenset[str] = frozenset(_TOOL_NAMES | set(TOOL_ALIASES))


def resolve_tool_name(name: str) -> str:
    """Resolve a primary `browser_*` or legacy `eval_*` tool name to its canonical name."""
    return TOOL_ALIASES.get(name, name)


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON-RPC 2.0 MCP Protocol Server
# ═══════════════════════════════════════════════════════════════════════════════

class BrowserMCPServer:
    """Pure-Python MCP JSON-RPC 2.0 browser server backed by `DualModeSessionManager`."""

    def __init__(self, screenshot_dir: Path | None = None) -> None:
        self.manager = DualModeSessionManager(screenshot_dir=screenshot_dir)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch any primary `browser_*` tool (or legacy `eval_*` alias) and return structured data."""
        args = arguments or {}
        canonical = resolve_tool_name(name)

        if canonical == "browser_start":
            default_session = "eval" if name == "eval_start" else "browser"
            session_name = str(args.get("name") or default_session).strip()
            mode = str(args.get("mode") or "v2").strip().lower()
            headless = bool(args.get("headless", True))
            vw = int(args.get("viewport_width") or 1280)
            vh = int(args.get("viewport_height") or 800)
            url = args.get("url")
            scenario = args.get("scenario")

            if mode == "pair":
                return await self.manager.start_pair(
                    session_name,
                    url=url,
                    scenario=scenario,
                    headless=headless,
                    viewport=(vw, vh),
                )
            return await self.manager.start_session(
                session_name,
                mode,
                url=url,
                scenario=scenario,
                headless=headless,
                viewport=(vw, vh),
            )

        if canonical == "browser_perceive":
            session = str(args.get("session") or "").strip()
            if not session:
                raise ValueError("'session' is required")
            return await self.manager.perceive(
                session,
                objective=str(args.get("objective") or ""),
                force_full=bool(args.get("force_full", False)),
                settle=bool(args.get("settle", True)),
            )

        if canonical == "browser_act":
            session = str(args.get("session") or "").strip()
            verb = str(args.get("verb") or "").strip()
            if not session or not verb:
                raise ValueError("'session' and 'verb' are required")
            return await self.manager.act(
                session,
                verb,
                element_id=args.get("element_id"),
                text=args.get("text"),
                url=args.get("url"),
                x=args.get("x"),
                y=args.get("y"),
                key=args.get("key"),
                direction=str(args.get("direction") or "down"),
                amount=int(args.get("amount", 500)),
                wait_ms=int(args.get("wait_ms", 800)),
                auto_perceive=bool(args.get("auto_perceive", False)),
                objective=str(args.get("objective") or ""),
            )

        if canonical == "browser_screenshot":
            session = str(args.get("session") or "").strip()
            if not session:
                raise ValueError("'session' is required")
            return await self.manager.screenshot(
                session,
                out_path=args.get("out_path"),
                full_page=bool(args.get("full_page", False)),
            )

        if canonical == "browser_stop":
            stop_session = args.get("session")
            if args.get("all") or not stop_session:
                return await self.manager.close_all()
            return await self.manager.close_session(str(stop_session))

        if canonical == "browser_compare":
            leg = str(args.get("legacy_session") or "").strip()
            v2 = str(args.get("v2_session") or "").strip()
            if not leg or not v2:
                raise ValueError("'legacy_session' and 'v2_session' are required")
            return await self.manager.compare(
                leg,
                v2,
                objective=str(args.get("objective") or ""),
                refresh=bool(args.get("refresh", True)),
                settle=bool(args.get("settle", True)),
            )

        if canonical == "browser_metrics":
            names = args.get("sessions")
            rep = await self.manager.report(names=names if isinstance(names, list) else None)
            rep["active_sessions"] = await self.manager.list_sessions()
            if args.get("include_scenarios", True):
                rep["scenarios"] = list_scenarios()
            return rep

        raise KeyError(f"Unknown tool '{name}'")

    async def handle_message(self, msg: Any) -> dict[str, Any] | None:
        """Process a decoded JSON-RPC 2.0 message; return response dict or None for notifications."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return _error_response(None, -32600, "Invalid Request: expected JSON-RPC 2.0 object")

        has_id = "id" in msg
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if not isinstance(method, str):
            return _error_response(req_id, -32600, "Invalid Request: missing string 'method'") if has_id else None

        # Notifications (no id) require no JSON-RPC response
        if not has_id or method.startswith("notifications/"):
            return None

        if not isinstance(params, dict):
            return _error_response(req_id, -32602, "Invalid params: expected JSON object")

        if method == "initialize":
            client_ver = params.get("protocolVersion")
            negotiated = client_ver if client_ver in SUPPORTED_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
            return _ok_response(req_id, {
                "protocolVersion": negotiated,
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
                "instructions": SERVER_INSTRUCTIONS,
            })

        if method == "ping":
            return _ok_response(req_id, {})

        if method == "tools/list":
            return _ok_response(req_id, {"tools": TOOL_DEFINITIONS})

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(tool_name, str) or tool_name not in _ALLOWED_TOOL_NAMES:
                return _error_response(req_id, -32602, f"Unknown tool: {tool_name!r}")
            if not isinstance(arguments, dict):
                return _error_response(req_id, -32602, "Invalid tool arguments: expected JSON object")
            try:
                data = await self.call_tool(tool_name, arguments)
                return _ok_response(req_id, {
                    "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}],
                    "isError": False,
                })
            except Exception as e:
                logger.debug("Tool '%s' failed: %s", tool_name, e)
                return _ok_response(req_id, {
                    "content": [{"type": "text", "text": json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})}],
                    "isError": True,
                })

        return _error_response(req_id, -32601, f"Method not found: {method}")

    async def serve_stdio(self) -> None:
        """Run the newline-delimited JSON-RPC 2.0 loop over stdin/stdout until EOF.

        Redirects `sys.stdout` and any existing stdout logging handlers to `sys.stderr`
        so third-party/legacy loggers (e.g. `app.logger`) can never corrupt the
        JSON-RPC protocol stream on stdout.
        """
        proto_out = sys.stdout
        sys.stdout = sys.stderr
        _redirect_stdout_log_handlers(proto_out, sys.stderr)

        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, stop_event.set)

        logger.info("MCP stdio server '%s' ready (%d tools)", SERVER_NAME, len(TOOL_DEFINITIONS))

        try:
            while not stop_event.is_set():
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    msg = json.loads(stripped)
                except json.JSONDecodeError as e:
                    _write_stream(proto_out, _error_response(None, -32700, f"Parse error: {e}"))
                    continue

                resp = await self.handle_message(msg)
                if resp is not None:
                    _write_stream(proto_out, resp)
        finally:
            try:
                await self.manager.close_all()
            except Exception as e:
                logger.debug("Error during MCP shutdown cleanup: %s", e)
            sys.stdout = proto_out
            logger.info("MCP stdio server shut down cleanly.")


# Backwards-compatible alias for code importing EvalMCPServer directly
EvalMCPServer = BrowserMCPServer


def _ok_response(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error_response(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _redirect_stdout_log_handlers(proto_out: Any, err_stream: Any) -> None:
    """Move any logging.StreamHandler bound to stdout onto stderr."""
    all_loggers = [
        logging.getLogger(),
        *(lg for lg in logging.Logger.manager.loggerDict.values() if isinstance(lg, logging.Logger)),
    ]
    for lg in all_loggers:
        for handler in lg.handlers:
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is proto_out:
                handler.setStream(err_stream)


def _write_stream(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent-first-browse Production Browser MCP stdio Server")
    parser.add_argument("--screenshot-dir", type=str, default=None, help="Directory for browser_screenshot PNGs.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging to stderr.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    server = BrowserMCPServer(
        screenshot_dir=Path(args.screenshot_dir) if args.screenshot_dir else None,
    )
    asyncio.run(server.serve_stdio())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
