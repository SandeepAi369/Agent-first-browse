# AGENTS.md — Agent-first-browse

Engineering handbook, module boundaries, and non-negotiable guardrails for human contributors and coding agents working on **Agent-first-browse**.

---

## 1. Project Overview

**Agent-first-browse** is a local, autonomous browser control engine built on **Python 3.11+**, **Playwright**, **LangGraph 1.x**, **Pydantic v2**, and **MCP (Model Context Protocol)**.

It supports two side-by-side execution engines controlled via `--mode {legacy,v2}` or `AGENT_ENGINE_MODE`:
- **`legacy`**: The proven V16–V29 baseline pipeline backed by frozen stealth/CDP modules.
- **`v2`** (`src/v2_engine/`): Adaptive `MutationObserver` DOM settle, paint-order occlusion pruning, Tier-2 A11y escalation, compact RefID delta compression, live `window.__aid` stale-ref semantic rebinding, and LangGraph `RetryPolicy`.

It also ships a zero-API **"Agent-as-LLM" Evaluation Connector** (`src/eval_connector/`) exposing identical live `legacy` and `v2` sessions over a `0600` Unix-socket CLI daemon (`agent-eval`) and a pure-Python JSON-RPC 2.0 MCP stdio server (`agent-eval-mcp`).

---

## 2. Repository Structure

```text
src/
  v2_engine/            # Dual-mode V2 perception, resilient actions, LangGraph policy, metrics
  eval_connector/       # Zero-API DualModeSessionManager, Unix-socket daemon, CLI, MCP stdio server
  core/                 # (Planned Chunks 2–7) Shared contracts: errors, retries, URL policy, DI, observability
  workers/              # Specialized worker prompts & action selection (base_worker.py)
  skills/               # Browser skill wrappers (navigate, interact, extract)
  orchestrator/         # Legacy/auxiliary critic (critic_v12.py) & orchestrator models
  app/                  # Shared logger/config/observability + legacy browser_promoter/ subsystem
  brain_graph.py        # Main LangGraph StateGraph orchestration spine
  brain_state.py        # Typed Pydantic BrainState (checkpointed per super-step)
  perception_engine.py  # Universal PerceptionStrategy Protocol & strict viewport filtering
  a11y_parser.py        # Hardened ARIA/CDP accessibility tree extractor + V34 secret redaction
  execution_safety.py   # Domain & file-upload safety guardrails
  feature_flags.py      # Live-read V29_* & dual-mode feature switchboard
  model_registry.py     # Multi-provider LLM failover chain, health tracker & circuit breaker
  overwatch.py          # Multi-layer action verification & recovery routing
  mcp_tools.py          # Playwright action wrappers (click, type, scroll, select, upload)
  dom_parser.py         # [FROZEN §6] Stage-1 DOM extraction & shadow piercer
  cdp_click.py          # [FROZEN §6] 4-tier CDP click waterfall
  cdp_input.py          # [FROZEN §6] CDP typing & React-reversion detection
  ghost_input.py        # [FROZEN §6] Bézier cursor & humanized scroll/input
  virtual_display.py    # [FROZEN §6] Xvfb headed display lifecycle
  advanced_agent.py     # Legacy monolithic loop (being decoupled from brain_graph.py)

scripts/
  check_guardrails.py   # Automated guardrail & contract verifier (frozen files, universal-only, exports, secrets, test hygiene)
  agent_eval.py         # CLI wrapper for eval_connector.cli:main
  benchmark_v2_vs_legacy.py # Headed/headless side-by-side V2 vs legacy benchmark
  diagnostics/          # Standalone environment & health checks
  runners/              # Task runner shell scripts

tests/
  test_*.py             # Offline unit & local-Chromium browser tests (default pytest target)
  browser_promoter/     # Legacy browser_promoter state/db/graph tests
  live/                 # Real-network / real-API tests (excluded by pyproject.toml norecursedirs)

docs/
  AUDIT_REPORT_agent_first_browse.md      # §6 preservation rules & V27 audit
  CLOUDFLARE_ARCHITECTURE_COMPARISON.md   # Architectural comparison & chunked upgrade roadmap
  EVAL_CONNECTOR.md                       # Zero-API CLI + MCP evaluation connector guide
  TEST_COVERAGE_MATRIX.md                 # Feature × layer test coverage & tracked debt
  V29_OVERHAUL.md                         # V29 cognitive reliability layers
  V2_ECOSYSTEM_AUDIT_2026.md              # V2 engine benchmark & ecosystem design
```

---

## 3. Public Entry Points & Package Boundaries

Every modern package exposes an explicit `__all__` barrel in `__init__.py`. Keep internal helpers private and verify exports with `scripts/check_guardrails.py`.

| Surface | Path / Entry Point | Stability |
|---|---|---|
| V2 Engine API | `src/v2_engine/__init__.py` | Public (`__all__` enforced) |
| Eval Connector API | `src/eval_connector/__init__.py` | Public (`__all__` enforced) |
| CLI Script: `agent-eval` | `eval_connector.cli:main` (`scripts/agent_eval.py`) | Public (`pyproject.toml`) |
| MCP Script: `agent-eval-mcp` | `eval_connector.mcp_server:main` | Public (`pyproject.toml`) |
| LangGraph Brain CLI | `src/run_v16.py` | Active entrypoint |
| Legacy Runners | `src/run.py`, `src/run_objective.py`, `src/run_prompt.py` | Legacy / maintenance-only |

---

## 4. Development & Verification Commands (Run Inside WSL)

All commands run inside WSL from `/home/sandeep/agent first IDE` using `.venv/bin/python`:

| Command | Purpose |
|---|---|
| `.venv/bin/python scripts/check_guardrails.py` | Run all automated architectural & safety guardrail checks |
| `.venv/bin/python scripts/check_guardrails.py --base-ref main` | Also assert zero edits to §6 frozen files relative to `main` |
| `.venv/bin/pytest -q` | Full offline unit + local-Chromium test suite (excludes `tests/live/`) |
| `.venv/bin/pytest tests/test_v2_engine.py tests/test_eval_connector_mcp.py -q` | Fast V2 + MCP smoke tests |
| `.venv/bin/flake8 src scripts tests --count --select=E9,F63,F7,F82 --show-source --statistics` | Hard CI syntax / undefined-name lint gate (excludes local venv/scratch dirs) |
| `.venv/bin/python scripts/agent_eval.py status` | Inspect zero-API eval daemon status |
| `.venv/bin/python -m eval_connector.mcp_server` | Run pure-Python JSON-RPC 2.0 MCP stdio server |

For headed desktop Chromium runs under WSLg, ensure `DISPLAY=:0` is set.

---

## 5. Non-Negotiable Boundaries

### Always
- Run `scripts/check_guardrails.py`, `flake8` (CI gate), and `pytest -q` before finishing any task.
- Keep new shared modules modular, typed (Python 3.11+ syntax, `from __future__ import annotations`), and backed by focused unit tests.
- Maintain dual-mode compatibility (`--mode legacy` and `--mode v2`) and keep all 350+ baseline tests green.
- Use feature branches for architectural work (never develop experimental changes directly on `main`).
- Include **exactly one blank line** at the end of every commit message followed by:
  ```text
  Co-authored-by: nishaguptha8-design <nishaguptha8@gmail.com>
  ```

### Ask First
- Adding new third-party runtime dependencies to `pyproject.toml` or `requirements.txt`.
- Changing CI workflows (`.github/workflows/python-package.yml`).
- Deleting or rewriting historical/safety branches (especially `backup/pre-history-cleanup`).
- Changing any contract that alters frozen §6 module boundaries.

### Never
- **NEVER modify §6 frozen legacy modules** (see `docs/AUDIT_REPORT_agent_first_browse.md` §6):
  - `src/dom_parser.py`
  - `src/cdp_click.py`
  - `src/cdp_input.py`
  - `src/ghost_input.py`
  - `src/app/browser_promoter/cdp_stealth_launcher.py`
  - `src/virtual_display.py`
- **NEVER add site-, domain-, or brand-specific literals or heuristics** to `src/perception_engine.py` or `src/v2_engine/perception.py` (**UNIVERSAL-ONLY rule**).
- **NEVER print, log, echo, or commit raw API keys, tokens, or credentials** (e.g., from `.env` or `~/.git-credentials`).
- **NEVER add new `tests/test_*.py` files that rely on import-time bare `assert` statements** instead of proper `def test_*()` / `async def test_*()` pytest functions.
- **NEVER bypass stdout protocol isolation** in `src/eval_connector/mcp_server.py` (all logging must go to `stderr`, never `stdout`).

---

## 6. Testing Strategy & Coverage Matrix

See [`docs/TEST_COVERAGE_MATRIX.md`](docs/TEST_COVERAGE_MATRIX.md) for the complete feature-by-layer test rollup, CI vs. local execution boundaries, and tracked test-hygiene debt.
