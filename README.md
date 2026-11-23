# 🌐 Agent First Browse

> **A fast, token-efficient Autonomous Browser Control Engine for full web perception, navigation, and real-browser interaction.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/langgraph-1.x-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Playwright](https://img.shields.io/badge/playwright-1.55%2B-green.svg)](https://playwright.dev/python/)
[![MCP Ready](https://img.shields.io/badge/MCP-stdio%20%2B%20CLI-purple.svg)](docs/EVAL_CONNECTOR.md)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Agent First Browse** gives AI agents reliable, human-grade control over real Chromium browsers—combining **accessibility-tree-first perception**, a **LangGraph verification loop**, and a **zero-API MCP/CLI connector** so any external AI coding agent (Antigravity, Claude Code, Cursor, VS Code, Gemini CLI) can drive the browser directly.

---

## Core V2 Capabilities

| Capability | What It Does |
|---|---|
| **Adaptive DOM Settle** | `MutationObserver` quiet-window settle replaces fixed 1.5s sleeps (**10–12x faster** per step). |
| **Modal & Occlusion Pruning** | Paint-order hit-testing (`elementFromPoint` + Shadow DOM) filters background controls trapped behind dialogs or overlays (up to **95% fewer tokens** when modals are active). |
| **Incremental Delta Snapshots** | On consecutive same-URL steps, sends only `+ADDED` / `~CHANGED` / `-REMOVED` controls plus a compact ref roster (**35–79% token reduction**). |
| **Tier-2 Accessibility-Tree Fallback** | Automatically escalates to CDP/Playwright Accessibility Tree extraction when Tier-1 DOM scans are sparse (`< 2` elements) on custom web components. |
| **SPA Stale-Ref Recovery** | Verifies live `window.__aid` handles, semantically rebinds buttons detached/shifted by SPA re-renders, and validates `<label>`-wrapped checkbox/radio transitions cleanly. |
| **Secret & Privacy Masking** | Masks password, OTP, PIN, and card-input values (`[filled: value hidden]`) before they ever reach prompts, logs, or state. |
| **Headed Stealth & LangGraph Safety** | Real Chromium under Xvfb/visible display, Bézier mouse movement, trusted CDP input, and LangGraph `RetryPolicy` on transient browser/LLM failures. |

---

## Quick Installation

Requires **Python 3.11+** on Linux, WSL, or macOS.

```bash
git clone https://github.com/SandeepAi369/Agent-first-browse.git
cd Agent-first-browse

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
python -m playwright install chromium

# Optional (recommended on headless Linux/WSL for headed stealth display):
sudo apt-get install -y xvfb
```

---

## 1. Run Autonomous Browser Tasks Directly

Configure your LLM provider key(s) once (`cp .env.example .env`), then run any plain-English browser objective:

```bash
# Run with the upgraded V2 engine
./agent.sh --mode v2 "Open https://news.ycombinator.com and find the top 3 stories"

# Run with the baseline legacy engine for comparison
./agent.sh --mode legacy "Open https://news.ycombinator.com and find the top 3 stories"

# Or invoke Python directly (add --headless to run without display)
.venv/bin/python src/run_v16.py run --mode v2 "Log in to https://the-internet.herokuapp.com/login and verify the secure area"
```

---

## 2. Connect Any AI Coding Agent (Zero API Keys Required)

Let **Antigravity, Claude Code, Cursor, VS Code, or Gemini CLI** act directly as the browser's decision-making brain—no external LLM API keys needed.

### Option A: Permanent Native MCP Server (`agent-eval-mcp`)

Copy [`.mcp.json.example`](.mcp.json.example) into your MCP client config:

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

Exposes 7 lean stdio tools: `eval_start`, `eval_perceive`, `eval_act`, `eval_compare`, `eval_screenshot`, `eval_metrics`, and `eval_stop` (also installed as `agent-eval-mcp`).

### Option B: Stateful CLI Daemon (`agent-eval`)

Drive live persistent Chromium sessions (headless or `--headed`) straight from any terminal:

```bash
# Start twin legacy + V2 sessions on any live URL (add --headed to watch live)
agent-eval start-pair live --url https://news.ycombinator.com --headed

# Compare speed, token count, occlusion pruning, and visibility side-by-side
agent-eval compare live_legacy live_v2

# Perceive & act step-by-step
agent-eval perceive live_v2
agent-eval act live_v2 click e1 --auto-perceive
agent-eval screenshot live_v2

# Export cumulative Markdown + JSON report and shut down cleanly
agent-eval report --json-out eval_report.json
agent-eval stop --daemon
```

*(Without `pip install -e .`, replace `agent-eval` with `.venv/bin/python scripts/agent_eval.py`.)*

---

## Benchmarks & Tests

```bash
# Run the full automated test suite
.venv/bin/pytest -q

# Run the built-in side-by-side Legacy vs V2 local stress benchmark (no API keys needed)
.venv/bin/python scripts/benchmark_v2_vs_legacy.py --suite local
```

See [`docs/EVAL_CONNECTOR.md`](docs/EVAL_CONNECTOR.md) for the verified headed benchmark table and diagnostic scenarios (`modal_occlusion`, `spa_rerender`, `sparse_aria`, `multistep_form`, `a11y_visibility`), and [`docs/V2_ECOSYSTEM_AUDIT_2026.md`](docs/V2_ECOSYSTEM_AUDIT_2026.md) for architecture notes.

---

## License & Support

- Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
- If you find this project useful, consider supporting ongoing open-source work via [GitHub Sponsors](https://github.com/sponsors/SandeepAi369).
