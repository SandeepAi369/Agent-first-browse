# Agent First Browse — Architectural Audit & Technical Plan
### Horizon: V27 — "LLM-Agnostic Near-Perfect Execution"
*Read-only audit. No execution code changed. Mathematical breakdowns included per mandate.*

---

## 0. Thesis & scope

**Goal:** maximize autonomous accuracy while *minimizing dependence on raw model size*, so that even a small/cheap text model (e.g. `gpt-oss-20b`) executes complex web tasks with near-perfect reliability.

The central claim of this audit, proven mathematically in §3.2: **task reliability is not a property of the model — it is a property of the scaffold around the model.** A weak per-step model wrapped in deterministic verification + bounded retry mathematically converges to near-perfect task success. Everything below serves that claim while honoring the non-regression clause (§6).

The system is mature (V17→V26.1, 10 shipped plans). This report audits the **current** tree (HEAD `9211e21` + 8 uncommitted files) — not an older mental model — and targets only the *weak* subsystems.

---

## 1. State reconciliation — recent custom modifications (Mandate #1)

### 1a. Shipped layers (committed)
| Layer | What it added | Relevance to the thesis |
|---|---|---|
| V17 | Model-first failover, probe-prune, JSON-mode rescue | Delivery reliability regardless of which model answers |
| V18 | Cognition: strategy + confidence + escalation ladder + PRM | Deterministic "how to think" wrapper |
| V19/.1/.2 | Element registry, disambiguation, off-screen action recall, div-button capture, real-scroll | Perception precision (right element, every time) |
| V20 | Evidence-grounded done-judge (`outcome_judge.py`) | Outcome truth independent of the worker |
| V21 | Vision-on-demand (`vision_consult.py`) | Fallback sense when DOM is ambiguous |
| V22 | Anti-bot realism (Xvfb, stealth script fix) | **PROTECTED — see §6** |
| V23 | Vision-chain prune, launcher | Latency |
| V24 | **Capability gate + dual-mode + role separation** | **The existing LLM-agnostic core** |
| V25 | Realistic cursor, viewport-bound vision | Grounding realism |
| V26/.1 | Monotonic sticky verification, single source of truth (`plan_steps`), forward-modeling, ledger-aware judge | Anti re-do-loop |

**V24 is the existing backbone of LLM-agnosticism** and must be *globalized*, not rebuilt: capability probe (drop models that can't structure even via JSON rescue), `AGENTIC_TEXT_ALLOWLIST` floor (pipeline never empty), and **role separation** (`get_worker_chain()` → critical-path worker uses tier ≤ 1 only; planner/PRM/judge use the full chain).

### 1b. In-flight work (uncommitted — 8 files) — audited
1. **`ask_user` + `missing_data`** (base_worker, moe_router) — agent stops and asks instead of fabricating PII. **Sound; globalize (§5).**
2. **Win-State Recognizer** (commit_node) — IRREVERSIBLE action (from `action_classifier`) + page reaction ⇒ strong "verify & finish" nudge. **Sound and task-agnostic.**
3. **Weighted `ChecklistItem`** (prm_critic) — first item w=0.5, last w=2.0. **Sound; but it silently changes the stall-detection scale — see §4 finding F6.**
4. **Semantic plan advancement** (commit_node) — advance only on step↔action keyword overlap. **Better than `"OK" in outcome`, but keyword-overlap is brittle — §4 F1.**
5. **Objective sandwich** (base_worker) — repeat objective at prompt bottom. **Strong, cheap anti-burial. Keep.**
6. **`expected_change` excluded from history** (brain_state) — avoids reconcile-loops. **But the worker still *emits* it — §4 F3.**
7. **URL trailing-punctuation strip** (browser_warmup) — fixes `amazon.in).` → `ERR_NAME_NOT_RESOLVED`. **Pure bugfix; commit it.**
8. **`critical_action_hint`** (base_worker) — injects the first non-done **PRM checklist** item as "CRITICAL REMAINING ACTION." **⚠️ This re-introduces the exact V26 regression (second competing sub-goal from the abstract PRM list vs. `plan_steps`). HIGH-priority latent bug — §4 F2.**

---

## 2. LLM-agnostic intelligence & cognitive wrappers (Mandate #2)

Research consensus (sources below): small models are *not* self-consistent on simple tasks, but become production-reliable through **(a) grammar/schema enforcement, (b) deterministic scaffolding, (c) multi-sample confidence-weighted voting with abstention, (d) an independent verifier.** The system already has (a) and partial (b),(d). The gap is **(c)** and a *deterministic decision-confidence gate*.

**Five wrappers proposed (all deterministic, model-size-independent):**

- **W1 — Critical-Action Consensus (CISC).** For IRREVERSIBLE actions only (cost-gated by `action_classifier`), sample the worker `n=3` times and take a **confidence-weighted majority vote** over `(verb, element_id)`. Confidence-Informed Self-Consistency matches plain self-consistency accuracy with **~46% fewer samples**. Math in §3.4.
- **W2 — Decision-Confidence Gate (abstention).** If the top voted action's weight < threshold `θ`, **abstain → escalate to vision (V21) or `ask_user`** rather than act on a coin-flip. Ensembles that may abstain "dramatically increase trustworthiness of the remaining answers."
- **W3 — Instruction-Aware DOM re-rank.** Add a relevance term to element scoring so the *right* element surfaces near the top even for a weak model that only reads the first few candidates (Prune4Web: instruction-relevance scoring **doubles grounding accuracy**, 25–50× context reduction). Math in §3.1.
- **W4 — Expanded deterministic action primitives.** A weak model fails not because it can't reason but because it lacks the *primitive* (no `select_option`, `hover`, `press_key`, `switch_tab`, `upload`). Give the hands more fingers; the brain needn't be bigger. §4 F7.
- **W5 — Verifier-gated retry as the reliability amplifier.** Already present (Overwatch retry) — §3.2 proves it is the single highest-leverage reliability multiplier and should be tuned, not replaced.

---

## 3. Mathematical formulations (Mandate #3)

### 3.1 DOM element scoring — current heuristic → objective-weighted utility

**Current (in `_GOD_MODE_JS`)** — additive salience with hand-tuned constants:
```
s(e) = ‖(cx,cy) − (vw/2, vh/2)‖₂            (base: distance from viewport centre)
       + 10000 · 𝟙[e off-screen]
       − 16000 · 𝟙[text(e) matches ACTION_RE]
       − 20000 · 𝟙[position(e) ∈ {fixed, sticky}]
rank = argsort_ascending s ; keep top-K (K = 60, or 80 if form)
```
**Weaknesses (mathematically):** (i) terms are *unnormalized* and on incommensurable scales — the distance term (≤ ~1200) is swamped by the magic constants, so the heuristic is effectively a **discrete priority bucketing** (action/fixed > on-screen > off-screen) with distance only as an intra-bucket tiebreak; (ii) **no dependence on the current sub-goal `g`** — two identical "Add to cart" buttons, or a search box vs. a nav link, are ranked purely by geometry, never by *what the task needs now*. This is the root of "weak model clicks a plausible-but-wrong element."

**Proposed (W3): a normalized, instruction-conditioned utility, applied as a STAGE-2 re-rank over the stage-1 survivors (non-regression: stage-1 salience still guarantees recall of action/fixed elements):**
```
U(e | g) = w_R·Rel(e,g) + w_A·Act(e) + w_V·Vis(e) + w_P·Prox(e) − w_D·Dup(e)
where each component ∈ [0,1]:
  Rel(e,g) = |tok(text(e)∪hint(e)) ∩ tok(g)| / |tok(g)|        ← NEW: instruction relevance (token-overlap; cosine if embeddings cheap)
  Act(e)   = 𝟙[role∈interactive ∨ ACTION_RE]                    ← affordance (was the −16000 boost)
  Vis(e)   = 1 if in viewport else max(0, 1 − d_off/vh)         ← visibility (was the +10000 penalty, now bounded)
  Prox(e)  = 1 − ‖(cx,cy)−c_ref‖₂ / D_max,  D_max = ‖(vw,vh)‖₂  ← centre/cursor proximity, normalized
  Dup(e)   = 1 if a higher-U element shares text+role within 22px ← de-dupe (V19.2 backstop, formalized)
rank = argsort_descending U ; keep top-K
```
Default weights (tunable, sum-normalized): `w_R=0.45, w_A=0.25, w_V=0.15, w_P=0.15, w_D=0.30`. The **`Rel` term is the entire point** — it lifts the goal-relevant element into the top handful the small model actually attends to. Stage-1 unchanged ⇒ recall preserved; stage-2 only *reorders*. Complexity O(N·|tok(g)|), negligible (N≤~150, |g|≤~30 tokens).

### 3.2 The reliability amplifier — why scaffolding beats model size (the headline)

Let `p` = probability the worker picks the correct action on a step (a property of model strength: a strong model p≈0.92, a weak one p≈0.7). Let a task require `k` correct *critical* steps.

- **Naïve (no verification):** task success `S₀ = pᵏ`.
- **With verifier-gated retry** (Overwatch detects "no real progress" and retries up to `r` times; treat retries as independent draws, verifier recall ρ≈1 for gross failures since it checks the live DOM): per-step success
```
p_eff = 1 − (1 − p)^(r+1)
S_r   = p_eff^k = [1 − (1−p)^(r+1)]^k
```

**Worked numbers** (weak model `p=0.70`, task `k=6` critical steps):
| | per-step | task success |
|---|---|---|
| Naïve `S₀` | 0.70 | **0.118 (12%)** |
| Retry r=2 | 0.973 | **0.847 (85%)** |
| Retry r=3 | 0.9919 | **0.953 (95%)** |

A 70% model becomes a **95% task-executor** purely through the deterministic retry wrapper. This is the formal justification of the thesis: **invest in the verifier+retry loop, not the model.** Two caveats the math also exposes: (1) it requires verifier false-pass rate `α→0` on *irreversible* steps (you cannot retry a placed order) — hence W1/W2 consensus is reserved precisely for IRREVERSIBLE actions; (2) it requires the retry to be *decorrelated* (a fresh tactic), which is exactly what the V18 escalation ladder guarantees (distinct rung each retry).

### 3.3 Strategy-confidence dynamics (formalizing the existing V18 rule)
`c ← c + γ(1−c)` on progress, `c ← (1−γ)c` on no-progress is the EWMA of a Bernoulli progress signal — an online estimate of `P(strategy correct)`. Strikes-to-threshold:
```
(1−γ)^n < τ  ⟹  n > ln τ / ln(1−γ)
γ=0.3, τ=0.4 ⟹ n > ln0.4/ln0.7 = 2.57 ⟹ 3 consecutive no-progress steps trigger re-strategize.
```
**Recommendation:** make `γ` *risk-adaptive* — `γ_eff = γ·(1+½·𝟙[last action IRREVERSIBLE])` so confidence falls faster after a high-stakes miss (re-strategize sooner where it matters).

### 3.4 Critical-action consensus (W1) — Condorcet / CISC
For `n` independent worker samples each correct w.p. `p>½`, plain majority vote correctness:
```
P_maj(n,p) = Σ_{i=⌈n/2⌉+1}^{n} C(n,i) p^i (1−p)^{n−i}   → 1 as n→∞   (Condorcet)
p=0.70: P_maj(3)=0.784, P_maj(5)=0.837
```
**Confidence-weighted (CISC)** replaces the count with `Σ_i conf_i·𝟙[action_i=a]` and picks `argmax_a`; empirically reaches the same accuracy as plain self-consistency with **~46% fewer samples** → `n=3` weighted ≈ `n≈5` unweighted. **Cost control:** gate W1 to IRREVERSIBLE actions only (typically 1–2 per task), so the consensus tax is ~2 extra calls per *task*, not per step.

### 3.5 Weighted PRM goal-score (formalize the in-flight change + fix the stall interaction)
In-flight aggregate: `G = (Σ_i wᵢ·scoreᵢ) / (Σ_i wᵢ)`, `scoreᵢ = wᵢ·(1·[done] + 0.5·conf·[in_progress])` ← **note the double-weight bug risk:** `item.score` already multiplies by `wᵢ`, and the aggregate divides by `Σwᵢ` — but if any future caller sums raw `item.score` and divides by `len`, weighting breaks. Pin the invariant: `G = Σ wᵢsᵢ / Σ wᵢ ∈ [0,1]`. **Stall interaction (F6):** with `w_last=2.0`, flipping the final item moves `G` by `2.0/Σw` — for a 4-item plan `Σw≈4`, that's `ΔG≈0.5 ≫ STALL_EPS=0.05`, while early-item flips move `G` by `0.5/4=0.125`. The stall detector (spread < 0.05 over window) must therefore use **per-item-normalized** progress, not raw `G`, or it will (a) never fire while early low-weight items dawdle and (b) over-react when the final item flips. Fix in §4 F6.

---

## 4. Hidden friction & adaptation bottlenecks (Mandate #4)

- **F1 — Router & plan-advance are keyword-brittle.** `_classify_plan_step` matches `startswith("navigate to"/"go to"/…)`; the in-flight semantic-advance matches step↔action token overlap (`len(w)>3`). Novel phrasings ("head over to", "pull up", non-English) silently miss → default interactor (safe) but plan-advance stalls. *Low severity (safe defaults) but caps adaptation.* Fix: replace `startswith`/overlap with a tiny normalized intent score `Rel(step, action) > θ` (same token-overlap primitive as §3.1).
- **F2 — ⚠️ Two competing sub-goal sources reintroduced (HIGH).** V26.1 made `plan_steps` the single worker-facing source of truth and *deleted* the PRM-checklist injection. The **in-flight `critical_action_hint` re-injects the first non-done PRM checklist item** into the worker prompt — the precise pattern that caused the V26 goal-amnesia regression. With Win-State hint + goal_complete_hint + correction_context + strategy block + critical_action_hint, the worker can receive **5 stacked, possibly-conflicting directives**. *This is the most important finding.* Fix: a **single "guidance bus"** — one prioritized slot (Win-State > ask_user > escalation > strategy), never the abstract PRM list, with a hard token budget.
- **F3 — `expected_change` is half-removed.** The worker still *generates* `expected_change` (tokens spent), but it's excluded from history (V27) to avoid reconcile-loops. Either *use* it as a one-shot forward check inside the same step (compare predicted vs. observed delta → cheap self-verification, feeds W2) or *drop the field* to save tokens. Half-state wastes both.
- **F4 — Registry-miss falls back to STALE coordinates, not re-perception.** `resolve_element` returns `{ok:false}` when a React re-render replaces the node (`isConnected=false`) → callers use the *snapshot* coords (the original V19 failure mode, just rarer). On dynamic SPAs this silently degrades. Fix: on registry miss, **re-extract once** (cheap, ~16ms) and re-resolve before falling to coordinates.
- **F5 — Prompt-budget unmanaged under context switching.** strategy block + beliefs + plan render + history + DOM markdown + objective-sandwich + hints have no global token ceiling; on dense pages the DOM markdown (the part the model must read) competes with accumulated cognitive scaffolding. Fix: a `prompt_budget` allocator that caps each section and elides oldest beliefs/history first (the LEAN-memory principle, globalized).
- **F6 — Weighted PRM vs. stall detector scale mismatch** (math in §3.5). Fix: stall detector should track the **count of items whose status changed**, or normalize `ΔG` by `max_i wᵢ/Σwᵢ`, not raw `G`.
- **F7 — Action vocabulary too small for novel environments.** No `select_option`, `hover`, `press_key(combo)`, `switch_tab`, `switch_iframe`, `upload`, `drag`. OAuth popups, native `<select>`, date pickers, file inputs, multi-tab flows are *inexpressible* → the agent improvises clicks and fails. **A weak model with the right primitive beats a strong model without it.** Highest-leverage adaptation fix.
- **F8 — `done`/win-state nudges can conflict with the judge.** Win-State says "finish NOW"; the V20 judge may still block (evidence not yet on screen) → potential 1-step oscillation. Fix: Win-State should *lower the judge's confidence bar*, not bypass it, and feed its reason as the judge's `next_hint`.

---

## 5. Inter-component & agent↔tool communication audit

- **Node↔node** is via typed `BrainState` (good, checkpointed). The smell is **directive stacking** (F2/F5): cognition writes to `correction_context`, `goal_complete_hint`, `recovery_advice`, `critical_action_hint` independently with no arbitration. **Recommendation: a single `guidance` channel** with a priority enum, so exactly one coherent instruction reaches the worker per step (mirrors how MoE routing already picks exactly one specialist).
- **Agent↔tool (`mcp_tools`)** returns structured dicts (good) but is **fire-and-forget within a step** — the worker decides, the tool acts, Overwatch observes *next* node. Tightening to an **act→micro-observe** (tool returns the immediate DOM delta + the resolved element identity) lets W2's confidence gate verify *before* committing the step, cheaply. Pairs with F3 (`expected_change` becomes a real forward check).
- **Tool surface (F7)** is the adaptation ceiling: expand primitives with deterministic handlers so capability comes from *code*, not model IQ.

---

## 6. Preservation constraints — DO NOT TOUCH (Mandate #5)

These are proven, high-class, and explicitly out of scope. Any V27 change must treat them as immutable dependencies:
- **Stealth/anti-bot engine** — `cdp_stealth_launcher.py` (`STEALTH_INIT_SCRIPT`, launch args, UA, viewport, WebGL/canvas/audio/font layers), V22 raw-string fix, `virtual_display.py` (Xvfb headed). **Frozen.**
- **4-tier click waterfall** — `cdp_click.resilient_click` (CDP `Input.dispatchMouseEvent` → CDP focus+JS click → Playwright fallback, with post-click verification). V27 only ever *feeds it better coordinates* (the V19 contract); never alters its internals.
- **CDP typing / React-reversion detection** — `cdp_input.resilient_type` waterfall + post-type verify + React-stable re-check. **Frozen.**
- **Humanization** — `ghost_input` Bézier paths, V19.2 real-scroll fix, V25 realistic cursor. **Frozen.**
- **Perception extraction core** — `_GOD_MODE_JS` scoring stage-1, shadow-piercer, V19 registry, V19.1/.2 action recall. §3.1 adds a **stage-2 re-rank on top**; it never modifies stage-1 recall.
- **Session persistence** — native `user_data_dir`, `SessionGuard`. **Frozen.**
The non-regression test: the full suite (currently 102/102 per V26.1) must stay green, and the protected modules' files must show **zero diff** except where a documented contract (coordinates in, struct out) is preserved.

---

## 7. Proposed V27 roadmap (prioritized by leverage ÷ risk)

| # | Item | Mandate | Leverage | Risk |
|---|---|---|---|---|
| P1 | **Fix F2 guidance-bus** (single arbitrated directive; remove PRM-checklist re-injection) | 1,4,5 | Very high | Low (prompt-only) |
| P2 | **W5 retry tuning + W1/W2 consensus & abstention for IRREVERSIBLE actions** | 2,3 | Very high (the 12%→95% lever) | Low–Med |
| P3 | **W3 instruction-aware DOM stage-2 re-rank** | 2,3 | High | Low (additive) |
| P4 | **F7 expanded action primitives** (`select_option`, `hover`, `press_key`, `switch_tab`, `upload`) | 2,4 | High | Med (new handlers; reuse waterfalls) |
| P5 | **F4 re-extract-on-registry-miss + F6 stall/weight fix + F3 expected_change decision** | 3,4 | Med | Low |
| P6 | **Commit & globalize in-flight `ask_user`, URL-strip, objective-sandwich, win-state** | 1 | Med | Low |
| P7 | **F5 prompt-budget allocator** | 2,4 | Med | Low |

**Recommended first execution target: P1 (guidance-bus) + P2 (consensus/abstention/retry).** P1 removes a live regression risk introduced by the parallel sessions; P2 is the mathematically-proven reliability multiplier that most directly delivers "small models, near-perfect tasks." Both are low-risk and touch none of the §6 protected core.

---

## Sources
- [Confidence Improves Self-Consistency in LLMs (CISC)](https://aclanthology.org/2025.findings-acl.1030.pdf) — confidence-weighted voting, ~46% fewer samples.
- [Existing LLMs Are Not Self-Consistent For Simple Tasks](https://arxiv.org/pdf/2506.18781) — small models' inconsistency; motivates W1/W2.
- [Increasing LLM trustworthiness using voting ensembles (with abstention)](https://arxiv.org/pdf/2510.04048) — abstention raises trust of remaining answers (W2).
- [The Six Sigma Agent: enterprise reliability via consensus-driven decomposed execution](https://arxiv.org/pdf/2601.22290) — consensus for critical steps.
- [VerifiAgent: a Unified Verification Agent](https://arxiv.org/pdf/2504.00406) — independent verifier pattern (V20 judge, §3.2).
- [Prune4Web: programmatic DOM tree pruning/scoring](https://www.emergentmind.com/topics/prune4web) — instruction-relevance scoring doubles grounding accuracy, 25–50× context reduction (W3/§3.1).
- [An Illusion of Progress? Assessing Web Agents](https://arxiv.org/html/2504.01382v4) — realistic web-agent reliability framing.
- [Top Small Language Models for Agentic AI](https://thirdeyedata.ai/data-ai-industry-insights/top-small-language-models-for-agentic-ai-solutions-development) — SLM agentic viability with scaffolding.
