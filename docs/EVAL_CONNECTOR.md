# Agent-first-browse — Zero-API "Agent-as-LLM" Evaluation Connector

The **Evaluation Connector** (`src/eval_connector/`) lets any AI coding agent (Antigravity, Claude Code, Cursor, VS Code, Gemini CLI) act directly as the LLM brain to live-test and benchmark `Agent-first-browse` in **`--mode legacy`** vs **`--mode v2`** side-by-side — with **zero external LLM API keys** and **zero extra pip dependencies**.

Both interfaces share one core session controller (`DualModeSessionManager`) that runs isolated Chromium contexts through the **exact real `legacy` and `v2` perception & action pipelines** while keeping all frozen legacy modules untouched.

---

## Architecture

- **Core Controller (`src/eval_connector/session.py`)** — Manages named/paired Chromium sessions (`--headless` or `--headed`/Xvfb), serializes legacy global `mcp_tools` binding under an `asyncio.Lock`, records per-step latency/token/action telemetry, and saves screenshots + Markdown/JSON reports.
- **Built-in Deep Scenarios (`src/eval_connector/scenarios.py`)** — Five reproducible stress fixtures:
  1. `modal_occlusion` — blocking consent modal covering 40+ background links/buttons.
  2. `spa_rerender` — detaches and shifts targeted button mid-workflow to verify stale-ref rebinding.
  3. `sparse_aria` — custom ARIA components (`treeitem`, `menuitemcheckbox`, `spinbutton`) triggering V2 Tier-2 A11y escalation.
  4. `multistep_form` — same-URL multi-step form testing delta compression and V34 secret redaction.
  5. `a11y_visibility` — CSS-hidden ghost exclusion, duplicate-row hints, and off-screen actionable recall.
- **Transport A — Stateful CLI Daemon (`agent-eval` / `scripts/agent_eval.py`)** — Persistent `0600` Unix-domain-socket daemon (`src/eval_connector/daemon.py`) with idle auto-shutdown.
- **Transport B — Native MCP stdio Server (`agent-eval-mcp` / `python -m eval_connector.mcp_server`)** — Pure-Python JSON-RPC 2.0 MCP server exposing 7 lean tools.

---

## Quickstart A: Stateful CLI Daemon (Zero MCP Config Needed)

```bash
# List built-in deep-test scenarios
.venv/bin/python scripts/agent_eval.py scenarios

# Start twin legacy + v2 Chromium sessions on a built-in scenario or live URL
.venv/bin/python scripts/agent_eval.py start-pair modal --scenario modal_occlusion
.venv/bin/python scripts/agent_eval.py start-pair hn --url https://news.ycombinator.com

# Add --headed to watch in a visible/Xvfb window
.venv/bin/python scripts/agent_eval.py start-pair live --url https://the-internet.herokuapp.com/login --headed

# Compare legacy vs v2 side-by-side (latency, tokens, occluded elements, visibility diff)
.venv/bin/python scripts/agent_eval.py compare modal_legacy modal_v2

# Act step-by-step as the LLM brain (positional shorthand or explicit flags)
.venv/bin/python scripts/agent_eval.py perceive modal_v2
.venv/bin/python scripts/agent_eval.py act modal_v2 click e1 --auto-perceive
.venv/bin/python scripts/agent_eval.py act modal_v2 type e3 "Sandeep"
.venv/bin/python scripts/agent_eval.py act modal_v2 goto https://example.com --auto-perceive

# Save visual screenshot & export Markdown + JSON benchmark report
.venv/bin/python scripts/agent_eval.py screenshot modal_v2
.venv/bin/python scripts/agent_eval.py report --json-out logs/eval_report.json

# Shut down all sessions and the background daemon
.venv/bin/python scripts/agent_eval.py stop --daemon
```

After `pip install -e .`, the console scripts `agent-eval` and `agent-eval-mcp` are available directly on `PATH`.

---

## Quickstart B: Native MCP stdio Server

Copy `.mcp.json.example` into your MCP client configuration (updating paths if needed):

```json
{
  "mcpServers": {
    "agent-first-browse-eval": {
      "command": ".venv/bin/python",
      "args": ["-m", "eval_connector.mcp_server"],
      "env": {
        "PYTHONPATH": "src"
      }
    }
  }
}
```

### Exposed MCP Tools (7)

| Tool | Purpose |
|---|---|
| `eval_start` | Start a `legacy`, `v2`, or twin `pair` session on a URL or built-in scenario (`headless: true/false`). |
| `eval_perceive` | Run real `legacy` or `v2` perception and return rendered text, ref elements, and step telemetry. |
| `eval_act` | Execute `goto`, `click`, `type`, `hover`, `select_option`, `press_enter`, `press_key`, `scroll`, `scroll_directional`, or `wait` (with optional `auto_perceive`). |
| `eval_compare` | Compare active `legacy` and `v2` sessions side-by-side (speedup, token savings %, occlusion pruning, Tier-2 escalation, visibility diff). |
| `eval_screenshot` | Capture a viewport or full-page PNG screenshot to disk. |
| `eval_metrics` | Return cumulative Markdown comparison table, transcripts, active sessions, and built-in scenario catalog. |
| `eval_stop` | Close a single session or all active sessions cleanly. |
---

## Verified Live Headed Side-by-Side Results (Legacy vs V2)

Measured on headed Chromium (`headless=False`, zero external LLM API keys, AI agent acting directly as the decision-making brain across live websites and all 5 diagnostic scenarios):

| Test | Legacy (`--mode legacy`) | V2 (`--mode v2`) | Gain |
|---|---|---|---|
| `the-internet.herokuapp.com/login` (full → delta) | `1,527 ms` / `~113–122 tok` | `140–148 ms` / `~55–71 tok` | **10.3–10.8x faster**, **42–51% fewer tokens** |
| `news.ycombinator.com` (Step 1 full / Step 2 delta) | `1,524–1,531 ms` / `~2,194 tok` | `151–155 ms` / `~1,419 → 482 tok` | **9.9–10.1x faster**, **35–78% fewer tokens** |
| `modal_occlusion` (blocking pop-up) | `1,520 ms` / `~621 tok` (46 els, 44 blocked) | `143 ms` / `~31 tok` (2 dialog btns) | **10.7x faster**, **95.0% fewer tokens**, **44 occluded pruned** |
| `spa_rerender` (shifted button click) | **FAIL** (`3,080 ms`, clicked stale coords) | **PASS** (`1,676 ms`, rebound to moved btn) | **Stale-ref semantic recovery** |
| `sparse_aria` (custom ARIA roles) | `0` elements found (blind) | `4` elements found (`tier=2` A11y) | **Automatic Tier-2 A11y rescue** |
| `multistep_form` (Step 2 review reveal) | `1,517 ms` / `~540 tok` (secret masked) | `141 ms` / `~256 tok` (`delta`, secret masked) | **10.8x faster**, **52.6% fewer tokens** |
| `a11y_visibility` (hidden ghosts & hints) | `1,515 ms` / `~91 tok` | `138 ms` / `~60 tok` | **11.0x faster**, **34.1% fewer tokens** |
| **Cumulative session average** | **`1,164.4 ms`** / **`754.2 tok/step`** | **`96.8 ms`** / **`341.1 tok/step`** | **12.0x faster**, **54.8% fewer tokens/step** |
