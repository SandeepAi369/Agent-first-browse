# V29 — Cognitive Overhaul & Anti-Regression Audit

Living record of the V29 overhaul. The horizon goal: stop **blind execution**,
kill **clicking/scrolling loops**, make consensus **clarity-triggered (pre-action)**,
give **vision a universal trigger**, and do all of it **without breaking V16–V28**.

Build order (approved): **Phase 0 → 1** now; 2 → 5 next. Additive + feature-flagged.
The legacy monolith `advanced_agent.py` is **not to be touched** until live testing of
the flagged V29 system is signed off.

---

## 0. Feature-flag registry (the anti-regression switchboard)

All flags live in [`feature_flags.py`](feature_flags.py). Defaults **ON**; read live from env.

| Env var | Default | Controls | Phase |
|---|---|---|---|
| `V29_ENABLED` | on | **Master kill-switch** — `0` ⇒ agent behaves exactly like V28 | — |
| `V29_REALITY` | on | Reality Monitor (screen-reality reconciliation) | 1 ✅ |
| `V29_REALITY_LLM` | on | Cheap LLM reconcile for *ambiguous* deltas only (deterministic-first) | 1 ✅ |
| `V29_CLARITY_CONSENSUS` | on | Broaden pre-action consensus to any low-clarity step | 2 ✅ |
| `V29_TARGET_LOCK` | on | Strict goal-binding + distractor resistance + temporal self-check | 2 ✅ |
| `V29_INTENT_JOURNAL` | on | Write-ahead action ledger + handoff hesitation (anti double-toggle) | 2.5 ✅ |
| `V29_SUBGOAL_LOCK` | on | Lock verified sub-goals across a global 'done' rejection (anti amnesia-loop) | 2.7 ✅ |
| `V29_STAGNATION` | on | Revived `same_url_streak` → generalized stagnation detector | 3 ✅ |
| `V29_SMART_SCROLL` | on | Scroll with feedback (delta / at-bottom / new-content) | 3 ✅ |
| `V29_PAGE_CONTEXT` | on | Page archetype + instruction-aware DOM re-rank | 4 ⏳ |
| `V29_ADAPTIVE_PERCEPTION` | on | Strategy-routed perception (P0 = Tier-1 passthrough) | AP-P0 ✅ |
| `V29_STRICT_VIEWPORT` | on | Drop off-screen noise; preserve+tag off-screen actionables | AP-P1 ✅ |
| `V29_DIFFING` | on | CriticV12 page-signal-vector diff + unified `state_change_score` | A ✅ |
| `V29_HYBRID_PRIMITIVES` | on | Clean semantic feedback (FailureClass) + hover/select_option/press_key | A ✅ |
| `V29_WEBDREAMER` | on | Predictive top-K simulation (LLM-imagined), Clarity+cost gated | B ✅ |
| `V29_WEBDREAMER_SITUATIONAL` | on | Situational scoring (reveal/scroll-deadend/desperation-goto) | B+ ✅ |
| `V29_LATS` | off | Tree-search/backtracking over checkpoints — reserved for Phase C | C ⏳ |
| `V29_SKILL_MEMORY_V2` | off | Deepened procedural memory (structural patterns) — reserved for Phase C | C ⏳ |

> To compare against V28 during live testing: `V29_ENABLED=0 ./agent.sh "<task>"`.
> Active flags are logged at the top of every run (`🧬 V29 Cognitive Overhaul ACTIVE …`).

---

## 1. FREEZE-LIST — proven modules, do NOT modify (Mandate 6 / audit §6)

V29 only ever *feeds these better inputs*; their internals stay at **zero diff**.

- **Stealth / anti-bot** — stealth launcher (`STEALTH_INIT_SCRIPT`, launch args, UA,
  WebGL/canvas/audio/font layers), [`virtual_display.py`](virtual_display.py) (Xvfb headed).
- **Click waterfall** — [`cdp_click.py`](cdp_click.py) `resilient_click` (CDP → JS → Playwright + post-click verify).
- **Type waterfall** — [`cdp_input.py`](cdp_input.py) `resilient_type` + React-reversion re-check.
- **Humanization** — [`ghost_input.py`](ghost_input.py) Bézier paths, real-scroll fix, realistic cursor.
  *(V29 smart-scroll, Phase 3, will only read the delta ghost_scroll already computes — not change how it scrolls.)*
- **Perception stage-1** — `_GOD_MODE_JS` salience scoring, shadow-piercer, V19 registry, action-recall.
  *(Phase 4 adds a stage-2 re-rank ON TOP; stage-1 recall is never altered.)*
- **Session persistence** — `user_data_dir`, `SessionGuard`.
- **Legacy monolith** — [`advanced_agent.py`](advanced_agent.py): keep only the shared utilities the brain
  imports (`launch_browser`, `SessionGuard`, `_invoke_with_failover`). **Do not delete or refactor until the
  user signs off live V29.**

---

## 2. Canonical decision path (what actually runs)

`agent.sh` → `run_v16.py run` → [`brain_graph.run_brain`](brain_graph.py) (LangGraph).
The legacy `run.py`/`advanced_agent.py` loop is **off** the decision path (it duplicates loop/streak
logic — a known divergence trap, slated for Phase 5 cleanup, not before live sign-off).

```
goal_compiler → planner → perceive → router(MoE) → worker → OVERWATCH → commit/retry/rollback/finalize
                                                              └─ V29 Reality Monitor lives here (Layer 3)
```

### Single-writer state discipline (prevents the V26-style regression)
Every new field has exactly ONE writer; all transient nudges reach the worker through the **one**
guidance bus ([`cognition.build_guidance`](cognition.py)). No new competing directive channels.

| New field | Writer (only) | Reader(s) |
|---|---|---|
| `reality_status` | `overwatch_node` | logs / diagnostics |
| `reality_note` | `overwatch_node` | `build_guidance` (top priority); cleared by `clear_transient` on commit |
| `bound_target` | `invoke_worker` (worker) | clarity gate, `overwatch` reality note, done-judge |
| `stagnation_level` / `stagnation_note` | `perceive_node` | `build_guidance` (below win); cleared on commit |
| `scroll_stuck_streak` | `overwatch_node` | smart-scroll escalation; cleared on commit |
| `last_attempted_action` | `overwatch_node` (write-ahead, pre-exec) | `invoke_worker` hesitation block + repeat-guard; cleared on verified success |

---

## 3. Phase 0 — Safety net  ✅ DONE

- ✅ Green baseline captured: **93/93** on the modules in scope (cognition, consensus, guidance,
  vision, verification, handoff, perception); the full V-suite holds **127** logic test functions + 34 orchestrator.
- ✅ Flag switchboard [`feature_flags.py`](feature_flags.py) (master kill-switch + per-feature).
- ✅ Freeze-list + dependency map + single-writer registry (this document).
- ✅ Startup flag-logging for run-log auditability.

## 4. Phase 1 — Screen-Reality Reconciliation (Mandate 1)  ✅ DONE

**Problem:** `expected_change` (the worker's forward model) was generated, logged, then discarded.
CriticV12 only checks *"did anything change?"* — so an error toast / out-of-stock / wrong redirect
still read as "progress" and got committed. That is blind execution.

**Fix:** [`reality.py`](reality.py) `classify_reality()` compares the prediction to the live screen →
`CONFIRMED` / `CONTRADICTED` / `UNCLEAR` / `NULL`. Wired into [`overwatch.py`](overwatch.py) Layer 3,
**before** the success branch, so a `CONTRADICTED` screen **overrides a false "progress"**: it blocks
the commit, escalates a distinct ladder tactic, and feeds the discrepancy back to the worker via the
guidance bus ("🚨 SCREEN-REALITY MISMATCH …") so it re-evaluates the REAL state.

Design guarantees:
- **Deterministic-first** — pure logic; one cheap LLM reconcile (`reconcile_with_llm`, reuses the judge
  chain) fires **only** on ambiguous (`UNCLEAR`) deltas, per the approved cost decision.
- **Agreement-bias-safe** — compares against the agent's OWN pre-committed prediction, never a free
  re-judgement (Self-Grounded Verification, arXiv 2507.11662).
- **Conservative** — `CONTRADICTED` only *escalates* (one extra think), never terminates; a rare
  false-positive costs latency, not a failed task.
- **Fully reversible** — `V29_REALITY=0` ⇒ Overwatch path is byte-for-byte V28.

**Tests:** [`test_reality_v29.py`](test_reality_v29.py) — **18/18** (classification across error/out-of-stock/
auth-redirect/success, guidance priority, clear-on-commit, flag switchboard, wiring guards).

---

## 5. Phase 2 — Clarity Gate + Target Lock (Mandates 4, 5, Contextual Focus)  ✅ DONE

**Pre-action consensus, broadened.** [`clarity.py`](clarity.py) computes one `ClaritySignal` from
signals already on the state (confidence, `needs_vision`, last-step contradiction, hesitation,
stagnation, target ambiguity). [`base_worker`](workers/base_worker.py) now enters the consensus
cascade when the action is **low-clarity OR irreversible** (was irreversible-only) — and the cascade
runs at **proposal time, before execution**, so the vote happens BEFORE the click. Cost stays bounded:
Tier-1 still short-circuits a confident, sound, unambiguous primary. Reversible abstentions proceed on
the primary (a retry recovers); irreversible abstentions wait/vision. The same signal also opens
**vision** to disambiguate look-alikes.

**Target Lock (anti context-drift).** [`target_lock.py`](target_lock.py) binds each step to the
target item's **semantic identity** (not the identical button label): `extract_target`,
`count_lookalikes`, `off_target_risk`, and a persistent prompt block carrying the explicit **TEMPORAL
SELF-CHECK** ("I acted on my target and it failed → clicking a neighbor's identical control abandons
the goal → re-examine my target or vote, never drift"). Wired into: the **worker prompt** (the ensemble
sees it too), the **clarity gate** (≥2 identical primary actions, or an about-to-click-wrong-item →
forces consensus), the **Overwatch reality note** (a contradiction recovery is bound to the target),
and the **done-judge** (`bound_target` → "verify completion for THIS item, not a look-alike").

## 6. Phase 3 — Progress-Aware Loops + Smart Scroll (Mandate 3)  ✅ DONE

**Stagnation detector.** [`stagnation.py`](stagnation.py) (PABU-style) combines the **revived**
`same_url_streak` (previously computed-but-unused in the live graph), a flat goal-score window, and a
short action-cycle detector (A,A,A / A,B,A,B). ≥2 signals ⇒ "busy but not progressing" → a guidance-bus
directive to change approach. Computed in `perceive_node`.

**Smart scroll.** [`mcp_tools.mcp_scroll`](mcp_tools.py) now measures the viewport before/after the
(frozen) `ghost_scroll` and returns `{scrolled_px, at_bottom, …}`. Overwatch tracks
`scroll_stuck_streak`; two unproductive scrolls (no movement / at bottom) escalate a *different* tactic
instead of scrolling into a wall.

**Tests:** [`test_phase23_v29.py`](test_phase23_v29.py) — **19/19**. Full V29 + regression: **130 passed**.

## 6.5 Atomic Intent Journaling — handoff-amnesia / double-toggle fix  ✅ DONE

**Diagnosis correction (important).** The LLM `_invoke_with_failover` is the *decision*
layer — Model A→B failover happens while CHOOSING an action; **no browser action runs there**.
The only side-effect point is Overwatch `_execute_action`. `mcp_click` wraps the click in a
`asyncio.wait_for(..., 30s)`; on timeout it returns `success:False` even though the CDP click may
have already dispatched. Recorded as "ineffective", the next worker re-clicks → **double-toggle on an
irreversible action**. So the journal lives at the execute point, fed to the whole chain via the prompt.

**Fix** ([`intent_journal.py`](intent_journal.py)): a write-ahead log (WAL/idempotency pattern).
1. Overwatch writes the Intent Payload `{ts, verb, element_id, target, risk, signature, status}`
   **before** executing — into BrainState (`last_attempted_action`, rides the node's atomic super-step
   commit) **and** a durable, atomically-written file (`persistence/intent_journal.json`, tmp+`os.replace`,
   written *before* the action — the "no action executes without a pre-commit record" guarantee).
2. The action runs; status is stamped (`confirmed`/`uncertain`/`executed`). A **verified success
   resolves/clears** the journal; any non-confirmed outcome (timeout/crash/ineffective) **persists** it.
3. The next decision is fed a strict **HESITATION** block, seen by *every* failover model. It applies to
   **any** unconfirmed side-effecting action (click/type/press_enter) and is **UNIVERSAL by design**: a
   single situational rule — "from the live DOM, decide if the effect already happened; reason about what
   repeating THIS specific control would do (duplicate? invert a state? advance a counter? re-submit?);
   repeat only if the DOM proves it did not" — that adapts to toggles, steppers, multi-state controls,
   foreign-language UIs, anything. `hazard_class()` (toggle/irreversible/state-change) is only a **soft,
   explicitly-non-authoritative hint** appended to it, never the scope.

> **Universality principle (governs the whole V29 layer).** Keyword lists (reality contradiction/affirm
> terms, toggle hints, primary-action patterns) are **non-exhaustive fast-path accelerators only** — never
> the source of truth. The *universal* reasoning lives in (a) the LLM at every gate (worker, consensus
> voters, the reality reconciler on ambiguous deltas, the done-judge) which sees the actual situation, and
> (b) signal/verb/structure-based protections that don't depend on labels at all: journaling fires on the
> *verb* (any side-effecting action), the clarity gate on *confidence/contradiction/hesitation*, stagnation
> on *numeric progress*, reality on *state-change*. So novel controls, unseen phrasings, and other languages
> are covered by the reasoning, with heuristics only making the common cases cheaper.
> (Known perception limit, logged for Phase 4: `dom_parser` flattens role=switch/checkbox/radio → `kind:button`
> and drops aria-checked/pressed — so structural toggle detection can't fire today; the universal LLM rule
> is what actually carries it. Surfacing ARIA state is a clean additive Phase-4 perception upgrade.)
4. If the worker re-proposes the SAME unconfirmed action, the **Clarity Gate forces a PRE-action
   consensus** (deterministic backstop, not just the prompt). Single-model/premium still gets the
   prompt-level protection.

The durable ledger is best-effort (never raises into the step) and cleared at run start so a prior
crashed run can't contaminate a fresh task. **Tests:** [`test_intent_journal_v29.py`](test_intent_journal_v29.py)
— **10/10**. Full V29 + regression: **140 passed**.

## 6.7 Adaptive Perception Engine — P0 (router scaffold)  ✅ DONE

**Mandate: UNIVERSAL-ONLY** (no site/domain/commerce logic in any new perception code) +
**no duplication** (audited: `a11y_parser.py` is dead legacy; `dom_parser` is the single live
pipeline with one caller, `perceive_node`).

[`perception_engine.py`](perception_engine.py) — a `PerceptionStrategy` interface + a router.
P0 wires **Tier-1 as a thin passthrough** over the existing `mcp_snapshot` (→ `dom_parser.extract`,
left **verbatim** per H3) — behavior-identical. The router exposes the seams later phases need:
an ordered, pluggable strategy list (modularity) and a universal `is_sufficient` element-count check
(computed + **logged** now; it gates escalation to the Tier-2 CDP deep-sweep and Tier-3 vision in
later phases). `perceive_node` routes through it behind `V29_ADAPTIVE_PERCEPTION` with a **hard
fallback** to the direct snapshot (perception can never break). No new `BrainState` fields (lean —
they arrive in AP-P3 when the tier actually varies).

The Universal-Only rule is enforced **mechanically**: `test_perception_engine_v29` greps the engine
source and fails on any site/commerce literal. **Tests: 10/10; full suite 181 passed.**

Researched from `browser-use` (studied, not copied) for later phases: CDP `getEventListeners`
(`has_js_click_listener`) recovers React/`<div>` buttons our `page.evaluate` scan can't see (the real
"finds no button" cause) → the **AP-P2 escalation-only** deep sweep; plus AX-tree roles and
propagating-bounds/occlusion filtering for **AP-P1 strict viewport** (recall-preserving, universal).

### Adaptive Perception build order
- **AP-P0** ✅ interface + router scaffold (Tier-1 passthrough).
- **AP-P1** ✅ strict viewport filter ON by default — universal post-pass in `perception_engine`
  (dom_parser untouched per H3). Drops elements whose centre is off-viewport (incl. `Y=-34`); the
  universal recall exemption preserves an off-screen element only when contextually critical —
  `kind∈{button,input}` OR its text overlaps the current-goal tokens (reuses Target-Lock `_tokens`,
  no commerce keywords) — tagging it `offscreen:true` + `(offscreen — scroll to reach)` in the markdown.
  selector_map + markdown filtered in lockstep. No-coords/on-screen elements pass through byte-identical.
  **Occlusion note:** true paint-order occlusion (an element covered by an overlay) needs an
  `elementFromPoint` DOM pass → deferred to **AP-P4**; AP-P1 is geometry-only and, by always preserving
  off-screen buttons/inputs + goal-relevant elements, cannot regress a real target. Tests: 5 new (15 total).
- **AP-P2** Tier-2 deep CDP sweep — escalation-only (protect the ~80 ms Tier-1); lazy-settle + CDP
  listeners + AX roles, same registry/contract.
- **AP-P3** routing/sufficiency triggers + worker `needs_deeper_dom` flag + tier state fields.
- **AP-P4** occlusion/paint-order drop; live tuning.

## 6.8 Phase A — Stabilizers (Diffing + Hybrid Primitives)  ✅ DONE

**Audit-first finding:** both proposed concepts were already ~70–80% present — `CriticV12` is already a
DOM-diff engine, and the 4-tier click waterfall is already internal (LLM issues clean verbs). So Phase A
is **enhance + consolidate**, not new pipelines.

**1. DOM Diffing (`V29_DIFFING`).** New [`dom_diff.py`](dom_diff.py): a cheap ~8-number universal
"page-signal vector" (one `page.evaluate`, layout-light DOM/ARIA selectors — no innerText/bbox) +
`signal_vector_diff` + unified `state_change_score`. Wired into [CriticV12](orchestrator/critic_v12.py):
captures the vector pre/post and **folds subtle overlay/`aria-expanded`/route/focus changes into the
progress signals** — so a click that opens an invisible overlay now reads as success instead of triggering
the stagnation loop. The diff stays in code; only a one-line phrase reaches the LLM. `state_change_score`
(0..1) is written to `BrainState` by Overwatch (single-writer) for Stagnation/Reality/world-model use.

**2. Hybrid Primitives & clean feedback (`V29_HYBRID_PRIMITIVES`).** New [`action_feedback.py`](action_feedback.py):
`FailureClass` (timeout/blocked/not_found/obscured/no_effect/input_failed) + **asymmetric verbosity** —
terse on success (observable effect only, **strategy name removed**), rich+semantic+recourse on failure.
The raw error is preserved in-line so the Reality/Intent-Journal "timed out"/"crashed" detectors are
unaffected. Expanded primitives in [mcp_tools.py](mcp_tools.py) reusing existing backends:
`hover` (ghost_move_to), `select_option` (native `<select>` via `__aid` registry), `press_key` (keyboard).
Wired through the schema, MoE router, action_classifier (`hover`→reversible), `ProposedAction` Literal,
and the Intent Journal (`select_option` journaled). Advertised in the worker prompt only when the flag is on.

**Guardrails honored:** universal-only (failure classes are execution semantics, not site rules), every piece
flag-gated, `window.__aid` registry + single state contract preserved, Overwatch/Stagnation **enhanced not
broken**. **Tests:** [test_phase_a_v29.py](test_phase_a_v29.py) — 18 (11 diffing + 7 primitives). **Full suite: 204 passed.**

## 6.9 Phase B — WebDreamer revival (predictive simulation)  ✅ DONE

The dormant [web_dreamer.py](web_dreamer.py) (was init'd but only `clear_cache()` called) is now wired
into [base_worker](workers/base_worker.py) as a **look-before-you-leap** layer: on a high-stakes ambiguous
step it IMAGINES (LLM world-model — **no real browser action**) the outcomes of the top-K candidates and
picks the best. Double-gated so it never burns compute on obvious steps: **(1) the Clarity Gate**
(`clarity_sig.uncertain`) **AND (2) the cost gate** `should_invoke_dreamer` (irreversible / stuck /
confused). Runs after consensus+vision on the refined action; overrides only when the imagined best is
**confident (≥0.6) AND genuinely different** (`should_override_with_dreamer`), else confirms — 45s timeout,
fully non-fatal. `_DREAMER` threaded through all three worker nodes; diagnostics `webdreamer_runs` /
`webdreamer_overrides` in `BrainState`. Flag `V29_WEBDREAMER`.

**Situational scoring (`V29_WEBDREAMER_SITUATIONAL`).** Fixes WebDreamer's "play-it-too-safe" vacuum scoring
(rated `scroll` 1.0, feared `goto`). A CONSERVATIVE (±0.15) UNIVERSAL additive delta on top of the LLM value,
keyed ONLY to existing state signals (no new computation, no registry/Overwatch touch) and applied solely in
candidate SELECTION (`select_best_evaluation`) so flag-off is byte-identical to baseline. Rules: **(reveal)**
big DOM change + static URL → reward engaging the revealed content, penalize scroll/goto-away; **(scroll)**
penalize ONLY unproductive scroll (`scroll_stuck_streak`) so infinite feeds are untouched; **(goto)** elevate by
`stuckness` (`same_url_streak`/`stagnation_level`) — desperation-goto when interacting fails. base_worker change
= one kwarg (`situation=state`); Overwatch/registry untouched. **Tests:** [test_phase_b_v29.py](test_phase_b_v29.py)
— 16, incl. the 4 universal situations (reveal / infinite-scroll / SPA / desperation-goto) as real selection
flips + flag-off parity. **Full suite: 220 passed.**

## 6.95 Sub-Goal Lock — the "amnesia loop" fix  ✅ DONE

**Bug:** on "Do X and Do Y", the agent finishes Y, prematurely says 'done', the Outcome Judge rejects
globally ("missing X"), and — because the rejection was binary/missing-only and the worker is blind to the
PRM ledger — the agent **re-does Y** (its button is still visible). **Audit finding:** the sticky PRM ledger
(verified+evidence, V26) already LOCKS Y and survives the rejection — it just never *drives* the worker. So
the fix activates existing memory rather than building a new one (no competing decomposition → no V26 regression).

[`subgoal_lock.py`](subgoal_lock.py) (pure, reuses the Target-Lock tokenizer), 4 layers:
- **L1 ledger** = the existing sticky `prm_checklist` (verified ⇒ locked/immutable).
- **L2 partial-success rejection** ([overwatch.py](overwatch.py) L4): `compose_rejection` re-affirms locked-done
  (Y) + names only what remains (X) — instead of a global "False" — and `reconcile_plan_with_ledger` marks the
  locked sub-goal's plan step done + activates the remainder.
- **L3 worker lock-list** ([base_worker.py](workers/base_worker.py)): a FORBID block ("already done — never repeat")
  from the ledger injected into the worker prompt — connects the ledger to the decision (the missing link);
  it forbids, never adds pending focus, so it's complementary to `plan_steps`.
- **L4 deterministic backstop**: fires ONLY in the post-rejection danger zone (`done_blocked > 0`); a proposed
  RE-DO of a locked sub-goal (≥2 shared identity tokens, state-changing verb) is held (wait) + redirected to the
  remaining work. Distinct remaining actions are never falsely blocked.

Flag `V29_SUBGOAL_LOCK`; termination guarantee (`MAX_DONE_BLOCKS`) intact. **Tests:**
[test_subgoal_lock_v29.py](test_subgoal_lock_v29.py) — 7. **Full suite: 227 passed.**

## 7. Next (not yet built)

- **Phase C** — LATS tree-search/backtracking (`V29_LATS`) + procedural skill memory v2 (`V29_SKILL_MEMORY_V2`).
- **AP-P2** — Tier-2 CDP deep sweep (escalation-only).
- **Phase 4** — Page-subject understanding + instruction-aware DOM re-rank + schema hardening.
- **Phase 5** — Prompt-budget allocator; retire duplicated legacy logic incl. dead `a11y_parser.py` (**after** sign-off).
