"""JSON-RPC 2.0 MCP stdio protocol, alias routing, packaging, and live Chromium tests.

Verifies:
  - Exact 7 production `browser_*` tools exposed via `tools/list` with valid JSON Schemas.
  - 100% backwards-compatible routing for all 7 legacy `eval_*` aliases via `tools/call`
    and `BrowserMCPServer.call_tool()`.
  - `pyproject.toml` CLI entrypoints (`agent-browse-mcp` primary + `agent-eval-mcp` alias),
    `server.json`, `.mcp.json.example`, and documentation paths.
  - Full MCP stdio subprocess lifecycle (`python -m eval_connector.mcp_server`):
    `initialize` (`serverInfo.name == "agent-first-browse"`) → `notifications/initialized` →
    `ping` → `tools/list` → JSON-RPC error codes → live Chromium workflow using both
    primary `browser_*` tools and legacy `eval_*` aliases → EOF cleanup.

Run: .venv/bin/python -m pytest tests/test_eval_connector_mcp.py -v
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from eval_connector.mcp_server import (
    DEFAULT_PROTOCOL_VERSION,
    SERVER_NAME,
    TOOL_ALIASES,
    TOOL_DEFINITIONS,
    BrowserMCPServer,
    EvalMCPServer,
    resolve_tool_name,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

PRIMARY_TOOLS = [
    "browser_start",
    "browser_perceive",
    "browser_act",
    "browser_screenshot",
    "browser_stop",
    "browser_compare",
    "browser_metrics",
]

LEGACY_ALIASES = {
    "eval_start": "browser_start",
    "eval_perceive": "browser_perceive",
    "eval_act": "browser_act",
    "eval_screenshot": "browser_screenshot",
    "eval_stop": "browser_stop",
    "eval_compare": "browser_compare",
    "eval_metrics": "browser_metrics",
}


class _MCPStdioClient:
    """Minimal JSON-RPC 2.0 stdio client for driving the real MCP subprocess."""

    def __init__(self, screenshot_dir: Path) -> None:
        env = dict(os.environ)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{SRC}:{existing_pp}" if existing_pp else str(SRC)
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "eval_connector.mcp_server",
                "--screenshot-dir",
                str(screenshot_dir),
            ],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._next_id = 1

    def send_raw(self, raw_line: str) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(raw_line + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, f"MCP server closed stdout unexpectedly. stderr={self.read_stderr()}"
        return json.loads(line)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        assert self.proc.stdin is not None
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        resp = self.send_raw(json.dumps(payload))
        assert resp.get("id") == req_id
        return resp

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> tuple[bool, dict[str, Any]]:
        resp = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        assert "result" in resp, resp
        result = resp["result"]
        text = result["content"][0]["text"]
        return bool(result.get("isError", False)), json.loads(text)

    def read_stderr(self) -> str:
        if self.proc.stderr is None:
            return ""
        try:
            return self.proc.stderr.read()
        except Exception:
            return ""

    def close(self) -> int:
        if self.proc.stdin is not None:
            with contextlib.suppress(Exception):
                self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait(timeout=5)


class _FakeSessionManager:
    """In-memory stub verifying primary `browser_*` and legacy `eval_*` routing parity."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def start_session(self, name: str, mode: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("start_session", {"name": name, "mode": mode, **kwargs}))
        return {"ok": True, "session": name, "mode": mode}

    async def start_pair(self, prefix: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("start_pair", {"prefix": prefix, **kwargs}))
        return {"ok": True, "prefix": prefix}

    async def perceive(self, session: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("perceive", {"session": session, **kwargs}))
        return {"ok": True, "session": session}

    async def act(self, session: str, verb: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("act", {"session": session, "verb": verb, **kwargs}))
        return {"ok": True, "session": session, "verb": verb}

    async def screenshot(self, session: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("screenshot", {"session": session, **kwargs}))
        return {"ok": True, "session": session}

    async def compare(self, legacy_session: str, v2_session: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("compare", {"legacy": legacy_session, "v2": v2_session, **kwargs}))
        return {"ok": True, "legacy": legacy_session, "v2": v2_session}

    async def report(self, names: list[str] | None = None) -> dict[str, Any]:
        self.calls.append(("report", {"names": names}))
        return {"ok": True, "markdown_table": "| stub |"}

    async def list_sessions(self) -> list[dict[str, Any]]:
        return [{"name": "work", "mode": "v2"}]

    async def close_session(self, session: str) -> dict[str, Any]:
        self.calls.append(("close_session", {"session": session}))
        return {"ok": True, "closed": [session]}

    async def close_all(self) -> dict[str, Any]:
        self.calls.append(("close_all", {}))
        return {"ok": True, "closed": ["work"]}


def _find_ref(elements: list[dict], needle: str, kind: str | None = None) -> str:
    low = needle.lower()
    for el in elements:
        if kind and (el.get("kind") or "").lower() != kind:
            continue
        if low in (el.get("text") or "").lower():
            return str(el["id"])
    raise AssertionError(f"Element '{needle}' not found in {[e.get('text') for e in elements]}")


def test_mcp_tool_definitions_exact_seven_tools():
    assert SERVER_NAME == "agent-first-browse"
    assert EvalMCPServer is BrowserMCPServer

    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert names == PRIMARY_TOOLS
    for tool in TOOL_DEFINITIONS:
        assert tool["inputSchema"]["type"] == "object"
        assert "properties" in tool["inputSchema"]

    assert TOOL_ALIASES == LEGACY_ALIASES
    for legacy, primary in LEGACY_ALIASES.items():
        assert resolve_tool_name(legacy) == primary
        assert resolve_tool_name(primary) == primary


@pytest.mark.asyncio
async def test_mcp_tools_list_and_tools_call_route_primary_and_legacy_aliases():
    server = BrowserMCPServer()
    fake = _FakeSessionManager()
    server.manager = fake  # type: ignore[assignment]

    # initialize advertises clean production server name
    init_resp = await server.handle_message({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": DEFAULT_PROTOCOL_VERSION},
    })
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "agent-first-browse"

    # tools/list exposes ONLY the 7 clean browser_* tools (no duplicate eval_* clutter)
    list_resp = await server.handle_message({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    })
    assert list_resp is not None
    listed = [t["name"] for t in list_resp["result"]["tools"]]
    assert listed == PRIMARY_TOOLS
    assert all(not n.startswith("eval_") for n in listed)

    # Every primary browser_* tool and legacy eval_* alias routes cleanly via tools/call
    cases: list[tuple[str, str, dict[str, Any], str]] = [
        ("browser_start", "eval_start", {"name": "work", "url": "https://example.com"}, "start_session"),
        ("browser_perceive", "eval_perceive", {"session": "work"}, "perceive"),
        ("browser_act", "eval_act", {"session": "work", "verb": "click", "element_id": "e1"}, "act"),
        ("browser_screenshot", "eval_screenshot", {"session": "work"}, "screenshot"),
        ("browser_compare", "eval_compare", {"legacy_session": "s_leg", "v2_session": "s_v2"}, "compare"),
        ("browser_metrics", "eval_metrics", {"include_scenarios": True}, "report"),
        ("browser_stop", "eval_stop", {"session": "work"}, "close_session"),
    ]
    req_id = 10
    for primary, legacy, arguments, expected_method in cases:
        fake.calls.clear()
        p_resp = await server.handle_message({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": primary, "arguments": arguments},
        })
        req_id += 1
        assert p_resp is not None and p_resp["result"]["isError"] is False
        p_data = json.loads(p_resp["result"]["content"][0]["text"])
        p_call = fake.calls[-1]

        fake.calls.clear()
        l_resp = await server.handle_message({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": legacy, "arguments": arguments},
        })
        req_id += 1
        assert l_resp is not None and l_resp["result"]["isError"] is False
        l_data = json.loads(l_resp["result"]["content"][0]["text"])
        l_call = fake.calls[-1]

        assert p_call[0] == expected_method
        assert l_call == p_call
        assert l_data == p_data

    # Default session name stays clean for browser_start and backwards-compatible for eval_start
    fake.calls.clear()
    await server.call_tool("browser_start", {})
    assert fake.calls[-1][1]["name"] == "browser"
    await server.call_tool("eval_start", {})
    assert fake.calls[-1][1]["name"] == "eval"

    # Unknown tool still rejected with JSON-RPC -32602
    bad_resp = await server.handle_message({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "tools/call",
        "params": {"name": "browser_nonexistent", "arguments": {}},
    })
    assert bad_resp is not None and bad_resp["error"]["code"] == -32602


def test_mcp_cli_entrypoints_and_packaging_metadata():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert "agent-browse-mcp" in scripts
    assert "agent-eval-mcp" in scripts
    assert scripts["agent-browse-mcp"] == "eval_connector.mcp_server:main"
    assert scripts["agent-eval-mcp"] == scripts["agent-browse-mcp"]

    # Both console scripts resolve to the exact same callable entrypoint
    resolved = []
    for script_name in ("agent-browse-mcp", "agent-eval-mcp"):
        mod_name, attr_name = scripts[script_name].split(":", 1)
        fn = getattr(importlib.import_module(mod_name), attr_name)
        assert callable(fn)
        resolved.append(fn)
    assert resolved[0] is resolved[1]

    # server.json advertises primary agent-browse-mcp entrypoint
    server_json = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    runtime_args = server_json["packages"][0]["runtimeArguments"]
    assert any(arg.get("value") == "agent-browse-mcp" for arg in runtime_args)

    # .mcp.json.example uses clean agent-first-browse key + agent-browse-mcp
    example_cfg = json.loads((ROOT / ".mcp.json.example").read_text(encoding="utf-8"))
    assert "agent-first-browse" in example_cfg["mcpServers"]
    assert "agent-browse-mcp" in example_cfg["mcpServers"]["agent-first-browse"]["args"]

    # Primary MCP guide and backwards-compatible stub both exist
    assert (ROOT / "docs" / "MCP_SERVER.md").is_file()
    assert (ROOT / "docs" / "EVAL_CONNECTOR.md").is_file()
    assert "MCP_SERVER.md" in (ROOT / "docs" / "EVAL_CONNECTOR.md").read_text(encoding="utf-8")


def test_mcp_stdio_protocol_errors_and_live_dual_mode_workflow(tmp_path: Path):
    client = _MCPStdioClient(screenshot_dir=tmp_path)
    try:
        # 1. Initialize handshake & protocol negotiation
        init_resp = client.request("initialize", {
            "protocolVersion": DEFAULT_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "pytest-client", "version": "1.0"},
        })
        assert init_resp["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION
        assert init_resp["result"]["serverInfo"]["name"] == "agent-first-browse"
        assert "tools" in init_resp["result"]["capabilities"]

        # Notification must NOT produce a response (verified by following ping matching its own id)
        client.notify("notifications/initialized")
        ping_resp = client.request("ping")
        assert ping_resp["result"] == {}

        # 2. tools/list returns exact 7 clean browser_* tools
        list_resp = client.request("tools/list")
        listed_names = [t["name"] for t in list_resp["result"]["tools"]]
        assert listed_names == PRIMARY_TOOLS

        # 3. JSON-RPC 2.0 error codes
        parse_err = client.send_raw("{not-valid-json")
        assert parse_err["error"]["code"] == -32700

        invalid_req = client.send_raw(json.dumps({"hello": "world"}))
        assert invalid_req["error"]["code"] == -32600

        method_err = client.request("resources/unknown")
        assert method_err["error"]["code"] == -32601

        tool_err = client.request("tools/call", {"name": "browser_nonexistent", "arguments": {}})
        assert tool_err["error"]["code"] == -32602

        # Runtime tool failure sets isError=True without crashing the server
        is_err, missing_data = client.call_tool("browser_perceive", {"session": "no_such_session"})
        assert is_err is True and missing_data["ok"] is False

        # 4. Live Chromium paired workflow over stdio MCP using primary browser_* tools
        is_err, started = client.call_tool("browser_start", {
            "name": "mcp",
            "mode": "pair",
            "scenario": "modal_occlusion",
            "headless": True,
        })
        assert is_err is False and started["ok"] is True

        is_err, cmp_data = client.call_tool("browser_compare", {
            "legacy_session": "mcp_legacy",
            "v2_session": "mcp_v2",
            "settle": True,
        })
        assert is_err is False
        assert cmp_data["step_comparison"]["v2_occluded_filtered"] >= 30
        assert cmp_data["step_comparison"]["token_savings_pct"] >= 70.0

        is_err, perc = client.call_tool("browser_perceive", {"session": "mcp_v2", "settle": False})
        assert is_err is False
        accept_ref = _find_ref(perc["elements"], "Accept Essential Cookies", kind="button")

        is_err, acted = client.call_tool("browser_act", {
            "session": "mcp_v2",
            "verb": "click",
            "element_id": accept_ref,
            "auto_perceive": True,
        })
        assert is_err is False and acted["ok"] is True
        assert acted["perception"]["element_count"] >= 20

        shot_file = tmp_path / "mcp_v2.png"
        is_err, shot = client.call_tool("browser_screenshot", {
            "session": "mcp_v2",
            "out_path": str(shot_file),
        })
        assert is_err is False and shot["ok"] is True
        assert shot_file.exists() and shot["bytes"] > 500

        is_err, metrics = client.call_tool("browser_metrics", {"include_scenarios": True})
        assert is_err is False and metrics["ok"] is True
        assert "| `mcp_v2` | v2 |" in metrics["markdown_table"]
        assert len(metrics["scenarios"]) >= 5

        is_err, stopped = client.call_tool("browser_stop", {"all": True})
        assert is_err is False and stopped["ok"] is True
        assert set(stopped["closed"]) == {"mcp_legacy", "mcp_v2"}

        # 5. Live Chromium backwards-compatibility smoke check via legacy eval_* aliases
        is_err, leg_started = client.call_tool("eval_start", {
            "name": "legacy_alias",
            "mode": "v2",
            "scenario": "modal_occlusion",
            "headless": True,
        })
        assert is_err is False and leg_started["ok"] is True

        is_err, leg_perc = client.call_tool("eval_perceive", {"session": "legacy_alias", "settle": True})
        assert is_err is False and leg_perc["element_count"] >= 1

        is_err, leg_stopped = client.call_tool("eval_stop", {"session": "legacy_alias"})
        assert is_err is False and leg_stopped["ok"] is True
    finally:
        rc = client.close()
        assert rc == 0
