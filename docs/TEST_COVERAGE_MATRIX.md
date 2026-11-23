# Test Coverage Matrix — Agent-first-browse

Repo-wide rollup of **what proves each subsystem works, at which test layer, which CI job guards it, and what test debt is explicitly tracked**.

> **Baseline & Upgraded Coverage:** Started from 350 collected pytest tests across 28 test files (`main` commit `e1015c3`), expanded through Chunks 1–9 to 528+ offline unit, guardrail, core-primitive, and local-Chromium integration tests (`pytest -q`).

---

## 1. Test Layers & CI Mapping

| Layer | Location | Execution Command | Guarded in CI (`.github/workflows/python-package.yml`)? | Requires API Keys / Internet? |
|---|---|---|---|---|
| **Guardrail Checks** | Frozen files, UNIVERSAL-ONLY, `__all__`/CLI exports, secret scan, test hygiene | `.venv/bin/python scripts/check_guardrails.py --strict-test-hygiene` | ✅ Hard fail (Python 3.11 & 3.12) | No |
| **Modern Lint (`ruff`)** | `src/core`, `src/v2_engine`, `src/eval_connector`, `scripts/check_guardrails.py` | `.venv/bin/ruff check src/core src/v2_engine src/eval_connector scripts/check_guardrails.py` | ✅ Hard fail (Python 3.11 & 3.12) | No |
| **Progressive Typecheck (`mypy`)** | `src/core`, `src/v2_engine`, `src/eval_connector`, `scripts/check_guardrails.py` | `.venv/bin/mypy src/core src/v2_engine src/eval_connector scripts/check_guardrails.py` | ✅ Hard fail (Python 3.11 & 3.12) | No |
| **Static Lint Gate (`flake8`)** | Repo-wide (`src/`, `scripts/`, `tests/`) | `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` | ✅ Hard fail (Python 3.11 & 3.12) | No |
| **Pure Unit Tests** | `tests/test_*.py`, `tests/browser_promoter/` | `.venv/bin/pytest -q` | ✅ Runs on every push/PR | No |
| **Local Browser-Backed Tests** | Local headless Playwright Chromium (`page.set_content()` / local HTML) | `.venv/bin/pytest -q` | ✅ Runs on every push/PR (Chromium installed in CI) | No |
| **Zero-API Side-by-Side E2E / Benchmark** | `scripts/benchmark_v2_vs_legacy.py`, `scripts/agent_eval.py` (5 built-in scenarios) | `.venv/bin/python scripts/benchmark_v2_vs_legacy.py` | ⏳ Manual / local headed (`DISPLAY=:0`) or headless | No |
| **Live External Tests** | `tests/live/` | `.venv/bin/pytest tests/live/<file>` | ❌ Excluded via `pyproject.toml` `norecursedirs = ["live", ...]` | Yes |

---

## 2. Feature × Layer Coverage Matrix

Legend: ✅ Covered & passing in default suite | ⚠️ Partial / indirect coverage | ⏳ Planned | ❌ Not applicable or missing

| Subsystem / Feature | Pure Unit | Local Chromium (Playwright) | Protocol / IPC (CLI & MCP) | Full Graph E2E (Offline) | Live External | Primary Test Files |
|---|---|---|---|---|---|---|
| **Accessibility-Tree Parser & V34 Secret Redaction** (`src/a11y_parser.py`) | ✅ | ✅ | — | — | — | `tests/test_a11y_parser.py` (13), `tests/test_security_v34.py` (37) |
| **V2 Browser Engine** (`src/v2_engine/`: settle, occlusion, Tier-2 A11y, delta, `v2_click`/`v2_type`, `RetryPolicy`) | ✅ | ✅ | ✅ | ✅ (`ScriptedFakeLLM`) | Manual | `tests/test_v2_engine.py` (9), `tests/test_eval_connector_core.py` (5), `tests/test_brain_graph_integration.py` (5) |
| **Zero-API Eval Connector** (`src/eval_connector/`: session manager, `0600` Unix daemon, CLI, JSON-RPC 2.0 MCP stdio) | ✅ | ✅ | ✅ | — | Manual | `tests/test_eval_connector_core.py` (5), `tests/test_eval_connector_cli.py` (3), `tests/test_eval_connector_mcp.py` (2) |
| **Perception Engine & Viewport Recall** (`src/perception_engine.py`, `src/dom_parser.py`) | ✅ | ✅ | — | — | — | `tests/test_perception_engine_v29.py` (15), `tests/test_perception_v19.py` (10) |
| **Execution Safety & Upload Guards** (`src/execution_safety.py`) | ✅ | ✅ | — | — | — | `tests/test_security_v34.py` (37) |
| **Session Lifecycle & Signal Shutdown** (`core.session_bootstrap.SessionGuard`, `virtual_display.py`) | ✅ | ✅ | — | — | — | `tests/test_session_guard.py` (4), `tests/test_core_session_bootstrap.py` (7) |
| **Stealth & Anti-Bot Launcher** (`cdp_stealth_launcher.py`) | ✅ | ✅ | — | — | Manual | `tests/test_antibot_v22.py` (7) |
| **Model Failover, Health Tracker & Circuit Breaker** (`src/model_registry.py`) | ✅ | — | — | — | ✅ (`tests/live/test_nvidia.py`) | `tests/test_failover_v17.py` (11), `tests/test_capability_v24.py` (16) |
| **Cognition Core, Ladder & Guidance Bus** (`src/cognition.py`, `src/consensus.py`, `src/action_classifier.py`) | ✅ | — | — | — | — | `tests/test_cognition_v18.py` (22), `tests/test_consensus_v27.py` (16), `tests/test_guidance_v27.py` (9) |
| **Verification, Outcome Judge & Vision Consult** (`src/verification_engine.py`, `src/outcome_judge.py`, `src/vision_consult.py`) | ✅ | — | — | — | — | `tests/test_verification_v20.py` (13), `tests/test_vision_v21.py` (14), `tests/test_handoff_v26.py` (9) |
| **V29 Cognitive Overhaul** (Reality Monitor, Intent Journal, Sub-Goal Lock, Stagnation, Phase A & B, Self-Healing) | ✅ | — | — | — | — | `tests/test_reality_v29.py` (18), `tests/test_intent_journal_v29.py` (17), `tests/test_subgoal_lock_v29.py` (7), `tests/test_phase_a_v29.py` (18), `tests/test_phase_b_v29.py` (16), `tests/test_phase23_v29.py` (19), `tests/test_selfhealing.py` (6) |
| **LangGraph Brain Spine** (`src/brain_graph.py`, `src/brain_state.py`) | ✅ | ✅ | — | ✅ (`ScriptedFakeLLM`, v2 + legacy) | ✅ (`tests/live/`) | `tests/test_brain_v16.py` (4), `tests/test_brain_graph_integration.py` (5) |
| **Core Shared Primitives** (`src/core/`: errors, retries, url_policy, truncate, redaction, disposable, runtime_context, browser_provider, session_bootstrap, observability, settings) | ✅ | ✅ (`LocalPlaywrightProvider`) | — | — | — | `tests/test_core_*.py` (148) |
| **Legacy Browser Promoter** (`src/app/browser_promoter/`) | ✅ | — | — | — | ✅ (`tests/live/test_auth_graph.py`) | `tests/browser_promoter/test_state.py` (21), `test_database.py` (8), `test_graph.py` (5) |
| **Architectural Guardrails** (`scripts/check_guardrails.py`) | ✅ | — | ✅ (CLI `--json` + exit codes) | — | — | `tests/test_guardrails.py` |

---

## 3. Tracked Test-Hygiene Debt (Cleared in Chunk 8)

All 4 legacy zero-test script files were converted into proper pytest test suites in **Chunk 8**, reducing `KNOWN_LEGACY_SCRIPT_TESTS` in `scripts/check_guardrails.py` to empty (`0` warnings under `--strict-test-hygiene`):

| File | Resolution Completed (Chunk 8) |
|---|---|
| `tests/test_brain_v16.py` | Converted into 4 proper `def test_*()` smoke tests (fixed `BrainState.model_fields` Pydantic V2.11 warning) |
| `tests/test_v151_blacklist.py` | Converted all 5 Patch A+C assertions into 5 isolated `def test_*()` unit tests |
| `tests/test_form_fill.py` | Converted into 2 offline `def test_*()` tests verifying state serialization & `StateGraph` compilation (manual live runner preserved under `__main__`) |
| `tests/test_read_only.py` | Converted into 2 offline `def test_*()` tests verifying read-only constraints & domain policy (manual live runner preserved under `__main__`) |

---

## 4. Tracked Static & CI Gaps

1. **CI Static Analysis Alignment (Resolved in Chunk 9):** `.github/workflows/python-package.yml` now hard-fails on `scripts/check_guardrails.py --strict-test-hygiene`, `ruff check`, and progressive `mypy` across `src/core`, `src/v2_engine`, `src/eval_connector`, and `scripts/check_guardrails.py`, alongside existing `flake8` and `pytest`.
2. **Pydantic V2 Deprecation Warnings:** `src/orchestrator/{actions,event_log,state}.py` still use legacy class-based `class Config:` (3 pytest collection warnings).
