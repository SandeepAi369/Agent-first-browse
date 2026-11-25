---
name: agent-first-browse
description: Drive real Chromium browsers via Agent First Browse's token-efficient accessibility-tree perception and action engine over MCP (`uvx --from agent-first-browse agent-eval-mcp`) or CLI (`agent-first-browse` / `agent-eval`). Use when an AI coding agent needs to open live web pages, fill forms, click SPA controls, handle modals/overlays, capture screenshots, or run end-to-end browser workflows.
---

# Agent First Browse — AI Coding Agent Skill

Use **Agent First Browse** to control real Chromium browsers cleanly with fast `MutationObserver` DOM settle, paint-order modal occlusion pruning, incremental delta snapshots, Tier-2 Accessibility-Tree fallback, and SPA stale-ref semantic recovery.

## Prerequisites

Install Playwright's Chromium binary once on the host machine:

```bash
pip install agent-first-browse && python -m playwright install chromium
```

Or run on demand with `uvx` (no global package install required, provided Playwright Chromium is installed):

```bash
uvx --from agent-first-browse agent-eval-mcp
```

## 1. Connecting via Native MCP (`stdio`)

Configure Claude Code, Cursor, Claude Desktop, or any MCP host to launch:

```json
{
  "mcpServers": {
    "agent-first-browse": {
      "command": "uvx",
      "args": ["--from", "agent-first-browse", "agent-eval-mcp"]
    }
  }
}
```

For Claude Code CLI:

```bash
claude mcp add agent-first-browse -- uvx --from agent-first-browse agent-eval-mcp
```

### MCP Tool Workflow (7 Tools, Zero External LLM API Keys)

| Tool | Purpose |
|---|---|
| `eval_start` | Launch a Chromium session (`mode: "v2"` recommended, `"legacy"`, or `"pair"`) on a `url` or built-in `scenario` (`headless: true/false`). |
| `eval_perceive` | Capture a compact ref-indexed snapshot (`[e1]`, `[e2]`, ...) with latency, token count, and occlusion-pruning telemetry. |
| `eval_act` | Execute `goto`, `click`, `type`, `hover`, `select_option`, `press_enter`, `press_key`, `scroll`, `scroll_directional`, or `wait` (pass `auto_perceive: true` to receive the updated DOM in one round trip). |
| `eval_compare` | Diff paired `legacy` vs `v2` sessions side-by-side (speedup, token savings, occluded elements pruned). |
| `eval_screenshot` | Save a viewport or full-page PNG screenshot to disk for visual verification. |
| `eval_metrics` | Inspect cumulative latency/token telemetry, active sessions, and built-in scenarios. |
| `eval_stop` | Close one session (`session: "<name>"`) or all active sessions (`all: true`). |

### Recommended Agent Loop

1. Call `eval_start(name="work", mode="v2", url="https://example.com", headless=true)` (set `headless=false` to display the browser window on screen).
2. Call `eval_perceive(session="work", objective="<goal>")` to inspect live interactive ref IDs (`e1`, `e2`, ...).
3. Call `eval_act(session="work", verb="click"|"type"|"goto"|..., element_id="eN", text="...", auto_perceive=true)` step-by-step until the goal is verified on the live page.
4. Call `eval_screenshot(session="work")` whenever visual confirmation is needed.
5. Always finish by calling `eval_stop(all=true)` so Chromium shuts down cleanly.

## 2. Running via Standalone CLI

### Autonomous LangGraph Mode (`agent-first-browse`)

When LLM provider API keys are configured in `.env`, run plain-English objectives directly:

```bash
agent-first-browse run --mode v2 "Open https://news.ycombinator.com and find the top 3 stories"
```

Add `--headless` for servers without a display.

### Stateful Zero-API CLI Daemon (`agent-eval`)

Drive persistent Chromium sessions directly from shell commands without MCP config:

```bash
agent-eval start work --mode v2 --url https://example.com
agent-eval perceive work
agent-eval act work click e1 --auto-perceive
agent-eval screenshot work
agent-eval stop --daemon
```

## Safety & Guardrails

- Never print or log passwords, OTPs, or API tokens (password fields are automatically masked as `[filled: value hidden]`).
- Never attempt to solve or bypass CAPTCHAs or anti-bot challenges; pause and report the block to the user.
- Always shut down active sessions (`eval_stop` / `agent-eval stop --daemon`) when finished.
