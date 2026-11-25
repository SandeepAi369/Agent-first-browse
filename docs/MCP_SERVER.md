# Agent-first-browse — Production Browser MCP Server

**Agent-first-browse** ships a pure-Python, zero-API **Production Browser MCP Server** (`agent-browse-mcp`) over JSON-RPC 2.0 `stdio`. Any MCP host (**Claude Desktop**, **Claude Code**, **Cursor**, **VS Code**, **Antigravity**, **Gemini CLI**) can drive real Chromium browsers directly with **zero external LLM API keys** and **zero extra MCP SDK dependencies**.

It uses the exact real V2 browser engine (`MutationObserver` adaptive DOM settle, paint-order modal occlusion pruning, incremental delta snapshots, Tier-2 Accessibility-Tree rescue, stale-ref semantic rebinding, and secret redaction) with built-in side-by-side `legacy` vs `v2` diagnostics.

---

## Quickstart: Connect via MCP (`stdio`)

Install Playwright's Chromium binary once on the host machine:

```bash
pip install agent-first-browse && python -m playwright install chromium
```

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "agent-first-browse": {
      "command": "uvx",
      "args": ["--from", "agent-first-browse", "agent-browse-mcp"]
    }
  }
}
```

### Claude Code

Register via the CLI in one command:

```bash
claude mcp add agent-first-browse -- uvx --from agent-first-browse agent-browse-mcp
```

Or copy [`.mcp.json.example`](../.mcp.json.example) to `.mcp.json` in your project root.

### Cursor (`.cursor/mcp.json` or `~/.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "agent-first-browse": {
      "command": "uvx",
      "args": ["--from", "agent-first-browse", "agent-browse-mcp"]
    }
  }
}
```

### Local Source Checkout (Without `uvx`)

```json
{
  "mcpServers": {
    "agent-first-browse": {
      "command": ".venv/bin/python",
      "args": ["-m", "eval_connector.mcp_server"],
      "env": {
        "PYTHONPATH": "src"
      }
    }
  }
}
```

> **Backwards compatibility:** The legacy CLI alias `agent-eval-mcp` and legacy `eval_*` tool names (`eval_start`, `eval_perceive`, `eval_act`, `eval_screenshot`, `eval_stop`, `eval_compare`, `eval_metrics`) remain fully supported and route to the exact same handlers.

---

## Exposed MCP Tools (7)

### Core Browser Control Tools

| Tool | Legacy Alias | Purpose |
|---|---|---|
| `browser_start` | `eval_start` | Launch an isolated Chromium session (`mode: "v2"` default, `"legacy"`, or twin `"pair"`) on any live `url` or built-in `scenario` (`headless: true/false`). |
| `browser_perceive` | `eval_perceive` | Capture a compact ref-indexed snapshot (`[e1]`, `[e2]`, ...) with latency, token count, and occlusion-pruning telemetry. |
| `browser_act` | `eval_act` | Execute `goto`, `click`, `type`, `hover`, `select_option`, `press_enter`, `press_key`, `scroll`, `scroll_directional`, or `wait` (pass `auto_perceive: true` to inspect the updated DOM in one round trip). |
| `browser_screenshot` | `eval_screenshot` | Capture a viewport or full-page PNG screenshot to disk and return its path, dimensions, and byte size. |
| `browser_stop` | `eval_stop` | Close a single named browser session (`session: "<name>"`) or shut down all active sessions (`all: true`). |

### Built-in Diagnostics & Comparison Tools

| Tool | Legacy Alias | Purpose |
|---|---|---|
| `browser_compare` | `eval_compare` | Compare active `legacy` and `v2` sessions side-by-side (speedup, prompt token savings %, occluded elements pruned, Tier-2 escalation, visibility diff). |
| `browser_metrics` | `eval_metrics` | Return cumulative latency/token telemetry, Markdown comparison table, active sessions, and built-in scenario catalog. |

### Recommended Production Agent Loop

1. Call `browser_start(name="work", url="https://example.com", headless=true)` (defaults to fast `mode="v2"`; set `headless=false` to display the browser window).
2. Call `browser_perceive(session="work", objective="<goal>")` to inspect compact interactive ref IDs (`e1`, `e2`, ...).
3. Call `browser_act(session="work", verb="click"|"type"|"goto"|..., element_id="eN", text="...", auto_perceive=true)` step-by-step until the goal is verified on the live page.
4. Call `browser_screenshot(session="work")` whenever visual confirmation is needed.
5. Call `browser_stop(all=true)` when finished so Chromium shuts down cleanly.

---

## Architecture & Built-in Stress Scenarios

- **MCP Server (`src/eval_connector/mcp_server.py`)** — Pure-Python JSON-RPC 2.0 `stdio` server (`BrowserMCPServer`, aliased as `EvalMCPServer`) with strict `stdout` protocol isolation (all logs redirected to `stderr`).
- **Session Controller (`src/eval_connector/session.py`)** — Manages isolated Chromium contexts (`--headless` or `--headed`/Xvfb), serializes legacy global `mcp_tools` binding under an `asyncio.Lock`, records per-step latency/token/action telemetry, and exports screenshots + Markdown/JSON reports.
- **Built-in Diagnostic Scenarios (`src/eval_connector/scenarios.py`)** — Five reproducible local stress fixtures runnable via `browser_start(scenario=...)`:
  1. `modal_occlusion` — blocking consent modal covering 40+ background links/buttons.
  2. `spa_rerender` — detaches and shifts targeted button mid-workflow to verify stale-ref rebinding.
  3. `sparse_aria` — custom ARIA components (`treeitem`, `menuitemcheckbox`, `spinbutton`) triggering V2 Tier-2 A11y escalation.
  4. `multistep_form` — same-URL multi-step form testing delta compression and V34 secret redaction.
  5. `a11y_visibility` — CSS-hidden ghost exclusion, duplicate-row hints, and off-screen actionable recall.
- **Stateful CLI Daemon (`agent-eval` / `scripts/agent_eval.py`)** — Persistent `0600` Unix-domain-socket daemon (`src/eval_connector/daemon.py`) with idle auto-shutdown for shell-driven browser control without MCP config.

---

## Alternative: Stateful CLI Daemon (`agent-eval`)

```bash
# List built-in stress scenarios
agent-eval scenarios

# Start a fast V2 session or paired legacy + v2 sessions on any live URL
agent-eval start work --mode v2 --url https://news.ycombinator.com
agent-eval start-pair modal --scenario modal_occlusion

# Step-by-step perception and actions
agent-eval perceive work
agent-eval act work click e1 --auto-perceive
agent-eval screenshot work

# Side-by-side comparison and cumulative benchmark report
agent-eval compare modal_legacy modal_v2
agent-eval report --json-out logs/eval_report.json

# Shut down all sessions and the background daemon
agent-eval stop --daemon
```

*(Without `pip install -e .`, replace `agent-eval` with `.venv/bin/python scripts/agent_eval.py`.)*

---

## Verified Live Headed Benchmarks (Legacy vs V2)

Measured on headed Chromium (`headless=False`, zero external LLM API keys, AI agent acting directly as the decision-making brain across live websites and all 5 diagnostic scenarios):

| Test | Legacy (`--mode legacy`) | V2 (`--mode v2`) | Gain |
|---|---|---|---|
| `the-internet.herokuapp.com/login` (full → delta) | `1,527 ms` / `~113–122 tok` | `140–148 ms` / `~55–71 tok` | **10.3–10.8× faster**, **42–51% fewer tokens** |
| `news.ycombinator.com` (Step 1 full / Step 2 delta) | `1,524–1,531 ms` / `~2,194 tok` | `151–155 ms` / `~1,419 → 482 tok` | **9.9–10.1× faster**, **35–78% fewer tokens** |
| `modal_occlusion` (blocking pop-up) | `1,520 ms` / `~621 tok` (46 els, 44 blocked) | `143 ms` / `~31 tok` (2 dialog btns) | **10.7× faster**, **95.0% fewer tokens**, **44 occluded pruned** |
| `spa_rerender` (shifted button click) | **FAIL** (`3,080 ms`, clicked stale coords) | **PASS** (`1,676 ms`, rebound to moved btn) | **Stale-ref semantic recovery** |
| `sparse_aria` (custom ARIA roles) | `0` elements found (blind) | `4` elements found (`tier=2` A11y) | **Automatic Tier-2 A11y rescue** |
| `multistep_form` (Step 2 review reveal) | `1,517 ms` / `~540 tok` (secret masked) | `141 ms` / `~256 tok` (`delta`, secret masked) | **10.8× faster**, **52.6% fewer tokens** |
| `a11y_visibility` (hidden ghosts & hints) | `1,515 ms` / `~91 tok` | `138 ms` / `~60 tok` | **11.0× faster**, **34.1% fewer tokens** |
| **Cumulative session average** | **`1,164.4 ms`** / **`754.2 tok/step`** | **`96.8 ms`** / **`341.1 tok/step`** | **12.0× faster**, **54.8% fewer tokens/step** |
