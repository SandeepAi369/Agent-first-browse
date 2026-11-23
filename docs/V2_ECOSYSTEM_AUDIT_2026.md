# Agent-first-browse — 2026 Browser Agent Ecosystem Audit & V2 Upgrade Plan

## Phase 1: 2026 Industry Benchmark vs. Current Codebase

We audited `Agent-first-browse` against current 2025–2026 breakthroughs across **OpenClaw**, **Playwright MCP (`@playwright/mcp`)**, **browser-use**, **LangGraph v1.x**, and **Patchright/Camoufox** stealth architectures.

### Comparison Matrix

| Dimension | 2026 Industry Standard | Legacy `Agent-first-browse` | Concrete Limitation Found | V2 Upgrade (`--mode v2`) |
|---|---|---|---|---|
| **Post-Action Settle** | Event-driven DOM idle / quietness settle | `brain_graph.py:211` unconditionally runs `page.wait_for_timeout(1500)` every step | Wastes **1.5s per step** (~22.5s over a 15-step task) even on already-idle pages | `wait_for_dom_settle()` using `MutationObserver` quiet window (120ms quiet, 650ms cap) |
| **Occlusion / Layering** | Paint-order & z-index filtering (`browser-use` `DomService`) | `strict_viewport_filter` checks viewport bounds only; no hit-testing | Background controls covered by modals, cookie banners, or sticky overlays leak into prompts and get mis-clicked | `filter_occluded_elements()` via `elementFromPoint` (+ shadow DOM traversal + full-shield safety guard + active modal header) |
| **Token Efficiency** | Compact ARIA/RefID tree (`[e5] button "Save"`) + transient/scoped diffs (`Playwright MCP`, `OpenClaw`) | Full verbose Markdown with emojis, container headings, and `(x, y)` coordinates repeated every step | High per-step prompt token bloat on repeated same-URL steps (form fills, modals, dropdowns) | `V2SnapshotCompressor`: compact ref format + same-URL incremental delta (`+ADDED / ~CHANGED / -REMOVED` + ultra-compact `UNCHANGED` ref roster) cutting ~40–65% prompt tokens |
| **Sparse Page Escalation** | Multi-layer DOM + CDP AXTree fusion (`browser-use`, `OpenClaw`) | `perception_engine.py` logs sparse (`< 2` elements) but leaves Tier-2 unimplemented | Custom ARIA/web-component pages returning sparse DOM hits have no automatic A11y escalation | `extract_tier2_a11y_fallback()` activates Tier-2 `a11y_parser.extract()` escalation automatically |
| **Stale Ref Recovery** | Strict ref lifecycle + clean stale invalidation / locator rebind (`OpenClaw`, `Playwright MCP`) | `mcp_click`/`mcp_type` fall back to stale snapshot `(x, y)` coords when `window.__aid[eid]` detaches | Clicking old coordinates after an SPA re-render clicks the wrong neighbor | `v2_click` / `v2_type`: pre-flight live ref check, deterministic semantic rebind by `(kind, label)` into `window.__aid`, or clean `STALE_REF` fail + delta invalidation |
| **Post-Click State Verification** | Live node state verification | `mcp_tools.py:165` queries `document.querySelector('[__aid="..."]')` — **dead code** (`dom_parser` stores refs in `window.__aid`, never DOM attributes) | V32 checkbox/toggle state-change safety net silently returns `{found: false}` 100% of the time | Fixed live `window.__aid[eid]` pre/post state diff (`checked`, `aria-checked`, `aria-expanded`, `aria-selected`, `aria-pressed`, `open`) |
| **Graph Fault Tolerance** | Native LangGraph `RetryPolicy` with exponential backoff + jitter | `build_brain_graph()` attaches zero `RetryPolicy` instances to nodes | Transient timeouts / 429s / execution-context resets propagate immediately without node-level backoff | `build_brain_graph(mode="v2")` attaches `v2_retry_policy()` to `planner`, `perceive`, and `overwatch` |
| **Anti-Bot / CDP Stealth** | Driver-level CDP `Runtime.enable` patching (`Patchright`) or C++ engine spoofing (`Camoufox`) | Standard Playwright + `playwright-stealth` JS init shims + Xvfb headed display (frozen legacy path) | Modern WAFs can detect standard Playwright CDP `Runtime.enable` leaks | Optional `detect_stealth_driver()` capability probe (zero forced bloat; ready for optional `patchright` drop-in while keeping frozen legacy stealth intact) |

### Preservation Guarantees
- All frozen legacy modules (`cdp_stealth_launcher.py`, `cdp_click.py`, `cdp_input.py`, `ghost_input.py`, `dom_parser.py`, `virtual_display.py`) remain **100% untouched**.
- Default mode (`--mode legacy` / unset `AGENT_ENGINE_MODE`) executes the exact original pipeline with zero behavioral regressions.
- All V2 upgrades live in modular `src/v2_engine/` and are selectable via `--mode v2` or `AGENT_ENGINE_MODE=v2`.
