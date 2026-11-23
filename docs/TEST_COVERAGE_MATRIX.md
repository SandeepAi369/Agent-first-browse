# Test Coverage Matrix — Agent-first-browse

Repo-wide rollup of **what proves each subsystem works, at which test layer, which CI job guards it, and what test debt is explicitly tracked**.

> **Baseline Verified:** 350 collected pytest tests across 28 test files (`pytest --collect-only -q`, commit `e1015c3`), plus new guardrail tests added in Chunk 1 (`tests/test_guardrails.py`).

---

## 1. Test Layers & CI Mapping

| Layer | Location | Execution Command | Guarded in CI (`.github/workflows/python-package.yml`)? | Requires API Keys / Internet? |
|---|---|---|---|---|
| **Static Lint Gate** | Repo-wide (`src/`, `scripts/`, `tests/`) | `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` | ✅ Hard fail (Python 3.11 & 3.12) | No |
| **Guardrail Checks** | Frozen files, UNIVERSAL-ONLY, `__all__`/CLI exports, secret scan, test hygiene | `.venv/bin/python scripts/check_guardrails.py` | ⏳ Local + `tests/test_guardrails.py` (CI step planned in Chunk 9) | No |
| **Pure Unit Tests** | `tests/test_*.py`, `tests/browser_promoter/` | `.venv/bin/pytest -q` | ✅ Runs on every push/PR | No |
| **Local Browser-Backed Tests** | 9 files using headless Playwright Chromium (`page.set_content()` / local HTML) | `.venv/bin/pytest -q` | ✅ Runs on every push/PR (Chromium installed in CI) | No |
| **Zero-API Side-by-Side E2E / Benchmark** | `scripts/benchmark_v2_vs_legacy.py`, `scripts/agent_eval.py` (5 built-in scenarios) | `.venv/bin/python scripts/benchmark_v2_vs_legacy.py` | ⏳ Manual / local headed (`DISPLAY=:0`) or headless | No |
| **Live External Tests** | `tests/live/` | `.venv/bin/pytest tests/live/<file>` | ❌ Excluded via `pyproject.toml` `norecursedirs = ["live", ...]` | Yes |

---

## 2. Feature × Layer Coverage Matrix

Legend: ✅ Covered & passing in default suite | ⚠️ Partial / indirect coverage | ⏳ Planned | ❌ Not applicable or missing

| Subsystem / Feature | Pure Unit | Local Chromium (Playwright) | Protocol / IPC (CLI & MCP) | Full Graph E2E (Offline) | Live External | Primary Test Files |
|---|---|---|---|---|---|---|
| **Accessibility-Tree Parser & V34 Secret Redaction** (`src/a11y_parser.py`) | ✅ | ✅ | — | — | — | `tests/test_a11y_parser.py` (13), `tests/test_security_v34.py` (37) |
| **V2 Browser Engine** (`src/v2_engine/`: settle, occlusion, Tier-2 A11y, delta, `v2_click`/`v2_type`, `RetryPolicy`) | ✅ | ✅ | ✅ | ⏳ (Chunk 8) | Manual | `tests/test_v2_engine.py` (9), `tests/test_eval_connector_core.py` (5) |
| **Zero-API Eval Connector** (`src/eval_connector/`: session manager, `0600` Unix daemon, CLI, JSON-RPC 2.0 MCP stdio) | ✅ | ✅ | ✅ | — | Manual | `tests/test_eval_connector_core.py` (5), `tests/test_eval_connector_cli.py` (3), `tests/test_eval_connector_mcp.py` (2) |
| **Perception Engine & Viewport Recall** (`src/perception_engine.py`, `src/dom_parser.py`) | ✅ | ✅ | — | — | — | `tests/test_perception_engine_v29.py` (15), `tests/test_perception_v19.py` (10) |
| **Execution Safety & Upload Guards** (`src/execution_safety.py`) | ✅ | ✅ | — | — | — | `tests/test_security_v34.py` (37) |
| **Session Lifecycle & Signal Shutdown** (`SessionGuard`, `virtual_display.py`) | ✅ | ✅ | — | — | — | `tests/test_session_guard.py` (4) |
| **Stealth & Anti-Bot Launcher** (`cdp_stealth_launcher.py`) | ✅ | ✅ | — | — | Manual | `tests/test_antibot_v22.py` (7) |
| **Model Failover, Health Tracker & Circuit Breaker** (`src/model_registry.py`) | ✅ | — | — | — | ✅ (`tests/live/test_nvidia.py`) | `tests/test_failover_v17.py` (11), `tests/test_capability_v24.py` (16) |
| **Cognition Core, Ladder & Guidance Bus** (`src/cognition.py`, `src/consensus.py`, `src/action_classifier.py`) | ✅ | — | — | — | — | `tests/test_cognition_v18.py` (22), `tests/test_consensus_v27.py` (16), `tests/test_guidance_v27.py` (9) |
| **Verification, Outcome Judge & Vision Consult** (`src/verification_engine.py`, `src/outcome_judge.py`, `src/vision_consult.py`) | ✅ | — | — | — | — | `tests/test_verification_v20.py` (13), `tests/test_vision_v21.py` (14), `tests/test_handoff_v26.py` (9) |
| **V29 Cognitive Overhaul** (Reality Monitor, Intent Journal, Sub-Goal Lock, Stagnation, Phase A & B, Self-Healing) | ✅ | — | — | — | — | `tests/test_reality_v29.py` (18), `tests/test_intent_journal_v29.py` (17), `tests/test_subgoal_lock_v29.py` (7), `tests/test_phase_a_v29.py` (18), `tests/test_phase_b_v29.py` (16), `tests/test_phase23_v29.py` (19), `tests/test_selfhealing.py` (6) |
| **LangGraph Brain Spine** (`src/brain_graph.py`, `src/brain_state.py`) | ⚠️ (import smoke only) | ❌ | — | ⏳ (Chunk 8 `FakeLLM`) | ✅ (`tests/live/`) | `tests/test_brain_v16.py` (0 pytest fns — debt), planned `tests/test_brain_graph_integration.py` |
| **Legacy Browser Promoter** (`src/app/browser_promoter/`) | ✅ | — | — | — | ✅ (`tests/live/test_auth_graph.py`) | `tests/browser_promoter/test_state.py` (21), `test_database.py` (8), `test_graph.py` (5) |
| **Architectural Guardrails** (`scripts/check_guardrails.py`) | ✅ | — | ✅ (CLI `--json` + exit codes) | — | — | `tests/test_guardrails.py` |

---

## 3. Tracked Test-Hygiene Debt (Quarantined Legacy Files)

The 4 files below live under `tests/` with **0 collected `test_*` functions**. They are explicitly allowlisted as **tracked debt** in `scripts/check_guardrails.py` (`KNOWN_LEGACY_SCRIPT_TESTS`) so they stay visible as warnings in normal runs, fail under `--strict-test-hygiene`, and cannot silently mask any *new* empty/script-style test file.

| File | Current Behavior | Why It Is Tracked Debt | Planned Resolution |
|---|---|---|---|
| `tests/test_brain_v16.py` | Runs top-level import/print checks at module import time during pytest collection (also emits Pydantic `model_fields` deprecation warning) | Zero `def test_*()` functions; executes side effects on import | Convert to proper `def test_*()` functions (Chunk 8) |
| `tests/test_v151_blacklist.py` | Runs bare top-level `assert` statements at import time | Zero `def test_*()` functions; not individually reported/isolated by pytest | Convert to proper `def test_*()` functions (Chunk 8) |
| `tests/test_form_fill.py` | Legacy `browser_promoter` standalone async script | Zero `def test_*()` functions; collected as 0 tests | Quarantine or convert/archive (Chunk 8) |
| `tests/test_read_only.py` | Live Reddit browsing script (`if __name__ == "__main__"`) sitting outside `tests/live/` | Zero `def test_*()` functions; misfiled outside `tests/live/` | Move into `tests/live/test_read_only.py` (Chunk 8) |

---

## 4. Tracked Static & CI Gaps

1. **Offline Full-Graph E2E Testing:** `src/brain_graph.py` currently has no deterministic zero-API multi-step graph integration test (`plan → perceive → route → worker → overwatch → retry/rollback → commit → finalize`). Planned in **Chunk 8** using a content-addressed `ScriptedFakeLLM`.
2. **CI Static Analysis Alignment:** `pyproject.toml` declares `ruff` and `mypy`, while `.github/workflows/python-package.yml` currently only runs `flake8` (hard-failing on syntax/undefined names, `--exit-zero` on complexity) and `pytest`. Integrating `scripts/check_guardrails.py`, `ruff`, and progressive `mypy` into CI is planned in **Chunk 9**.
3. **Pydantic V2 Deprecation Warnings:** `src/orchestrator/{actions,event_log,state}.py` still use legacy class-based `class Config:` (4 pytest collection warnings).
