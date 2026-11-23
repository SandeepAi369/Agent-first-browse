"""JSON-RPC 2.0 MCP stdio protocol and live Chromium tests for Chunk 3.

Verifies:
  - Full MCP stdio subprocess lifecycle (`python -m eval_connector.mcp_server`):
    `initialize` → `notifications/initialized` (no response) → `ping` →
    `tools/list` (exact 7 evaluation tools & valid JSON Schemas).
  - JSON-RPC 2.0 error codes: `-32700` parse error, `-32600` invalid request,
    `-32601` unknown method, `-32602` unknown tool, and `isError: true` on
    runtime tool errors (e.g. missing session) without crashing the server.
  - End-to-end live Chromium dual-mode evaluation over stdio MCP:
    `eval_start(mode='pair', scenario='modal_occlusion')` → `eval_compare` →
    `eval_perceive` → `eval_act(click, auto_perceive=True)` →
    `eval_screenshot` → `eval_metrics` → `eval_stop(all=True)` → EOF cleanup.

Run: .venv/bin/python -m pytest tests/test_eval_connector_mcp.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval_connector.mcp_server import DEFAULT_PROTOCOL_VERSION, TOOL_DEFINITIONS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


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
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            return self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait(timeout=5)


def _find_ref(elements: list[dict], needle: str, kind: str | None = None) -> str:
    low = needle.lower()
    for el in elements:
        if kind and (el.get("kind") or "").lower() != kind:
            continue
        if low in (el.get("text") or "").lower():
            return str(el["id"])
    raise AssertionError(f"Element '{needle}' not found in {[e.get('text') for e in elements]}")


def test_mcp_tool_definitions_exact_seven_tools():
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert names == [
        "eval_start",
        "eval_perceive",
        "eval_act",
        "eval_compare",
        "eval_screenshot",
        "eval_metrics",
        "eval_stop",
    ]
    for tool in TOOL_DEFINITIONS:
        assert tool["inputSchema"]["type"] == "object"
        assert "properties" in tool["inputSchema"]


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
        assert init_resp["result"]["serverInfo"]["name"] == "agent-first-browse-eval"
        assert "tools" in init_resp["result"]["capabilities"]

        # Notification must NOT produce a response (verified by following ping matching its own id)
        client.notify("notifications/initialized")
        ping_resp = client.request("ping")
        assert ping_resp["result"] == {}

        # 2. tools/list returns exact 7 tools
        list_resp = client.request("tools/list")
        listed_names = [t["name"] for t in list_resp["result"]["tools"]]
        assert len(listed_names) == 7
        assert "eval_start" in listed_names and "eval_compare" in listed_names

        # 3. JSON-RPC 2.0 error codes
        parse_err = client.send_raw("{not-valid-json")
        assert parse_err["error"]["code"] == -32700

        invalid_req = client.send_raw(json.dumps({"hello": "world"}))
        assert invalid_req["error"]["code"] == -32600

        method_err = client.request("resources/unknown")
        assert method_err["error"]["code"] == -32601

        tool_err = client.request("tools/call", {"name": "eval_nonexistent", "arguments": {}})
        assert tool_err["error"]["code"] == -32602

        # Runtime tool failure sets isError=True without crashing the server
        is_err, missing_data = client.call_tool("eval_perceive", {"session": "no_such_session"})
        assert is_err is True and missing_data["ok"] is False

        # 4. Live Chromium paired workflow over stdio MCP
        is_err, started = client.call_tool("eval_start", {
            "name": "mcp",
            "mode": "pair",
            "scenario": "modal_occlusion",
            "headless": True,
        })
        assert is_err is False and started["ok"] is True

        is_err, cmp_data = client.call_tool("eval_compare", {
            "legacy_session": "mcp_legacy",
            "v2_session": "mcp_v2",
            "settle": True,
        })
        assert is_err is False
        assert cmp_data["step_comparison"]["v2_occluded_filtered"] >= 30
        assert cmp_data["step_comparison"]["token_savings_pct"] >= 70.0

        is_err, perc = client.call_tool("eval_perceive", {"session": "mcp_v2", "settle": False})
        assert is_err is False
        accept_ref = _find_ref(perc["elements"], "Accept Essential Cookies", kind="button")

        is_err, acted = client.call_tool("eval_act", {
            "session": "mcp_v2",
            "verb": "click",
            "element_id": accept_ref,
            "auto_perceive": True,
        })
        assert is_err is False and acted["ok"] is True
        assert acted["perception"]["element_count"] >= 20

        shot_file = tmp_path / "mcp_v2.png"
        is_err, shot = client.call_tool("eval_screenshot", {
            "session": "mcp_v2",
            "out_path": str(shot_file),
        })
        assert is_err is False and shot["ok"] is True
        assert shot_file.exists() and shot["bytes"] > 500

        is_err, metrics = client.call_tool("eval_metrics", {"include_scenarios": True})
        assert is_err is False and metrics["ok"] is True
        assert "| `mcp_v2` | v2 |" in metrics["markdown_table"]
        assert len(metrics["scenarios"]) >= 5

        is_err, stopped = client.call_tool("eval_stop", {"all": True})
        assert is_err is False and stopped["ok"] is True
        assert set(stopped["closed"]) == {"mcp_legacy", "mcp_v2"}
    finally:
        rc = client.close()
        assert rc == 0
