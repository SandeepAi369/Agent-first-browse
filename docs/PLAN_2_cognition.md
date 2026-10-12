# Plan 2 — Adaptive Cognition Core (V18)

> The primary "creativity / adaptive thinking" work, built on Plan 1's now-reliable
> reasoning layer. Lands in the canonical V16 LangGraph brain only. **Local only —
> nothing pushed.** Not started — awaiting your green light.

---

## Why this, and why now

Plan 1 made model calls fast and reliable. Now we make the agent *think* better.

Today the V16 brain is a **stateless step-reactor**: every step the worker LLM
re-reads (plan + facts + history + DOM) and re-derives an action from scratch.
Three concrete, evidence-backed gaps cap how creative it can be:

1. **No persistent strategy/hypothesis.** Nothing anchors reasoning across steps,
   so the agent can silently change its approach every step (the seed of loops). It
   never says "my theory is X, let me test it" and never *revises* that theory from
   evidence.

2. **No goal-aware progress signal during execution.** `PRMCritic.score_step()`
   exists but is **never called in the brain** — `brain_graph.py` only calls
   `generate_checklist` once at the start. The only live signal is CriticV12's
   change-based "did anything change?", which says *yes* even when the agent churns
   the DOM without getting closer to the goal. The agent literally cannot tell
   "busy" from "progressing."

3. **Crude, stateless unsticking.** When stuck, `overwatch.py` injects a fixed
   string ("try scroll / dismiss overlays / wait") with no memory of which tactic
   was already tried — so it can suggest scroll, fail, suggest scroll again.
   Reflexion fires only *after* a failure streak; there's no proactive
   "am I on the right track?"

**Outcome:** the agent forms an explicit strategy, reasons *with* it, updates its
confidence from evidence, escalates through *distinct* tactics when blocked (never
repeating one), and proactively catches "local change without goal progress" and
re-strategizes.

---

## What changes (files)

| File | Change |
|---|---|
| **NEW `cognition.py`** | `StuckLadder` (deterministic escalation), `update_confidence()`, `detect_stall()`, `restrategize()` (LLM). Pure logic, unit-testable. |
| `brain_state.py` | Add cognitive fields (below), reusing the `Annotated[list, add]` reducer already used by `reflections`. |
| `brain_graph.py` | `planner_node` → **strategic planner** (one call returns strategy+assumptions+steps, no extra LLM call); `commit_node` reinforces confidence + goal-aware plan advance + throttled PRM audit; `rollback_node` drives ladder/restrategize. |
| `workers/base_worker.py` | Inject `strategy` + `confidence` + `beliefs` into the worker prompt (the core change). Reuses the existing `state_override` slot. |
| `overwatch.py` | Replace generic `correction_context` strings with the `StuckLadder` directive; emit the confidence-decay signal. |
| `prm_critic.py` | **Reused as-is** (revived) — `score_step()` wired into the audit. No schema change. |
| **NEW `test_cognition_v18.py`** | Unit tests, mirroring `test_failover_v17.py`. |

**New BrainState fields:** `strategy`, `strategy_confidence` (1.0), `beliefs`
(add-reducer list), `current_obstacle`, `ladder_rung`, `tried_tactics`,
`goal_score_window`, `restrategize_count`.

---

## The five mechanisms (with the math)

**1. Strategic planner — 0 extra LLM calls.** `planner_node` already makes one
decompose call. Change its output to a typed model
`StrategicPlan{ strategy, assumptions[], steps[] }` (strict-safe thanks to Plan 1).
`steps` → `plan_steps` (downstream unchanged); `strategy` → `state.strategy`;
`assumptions` → `state.beliefs`. The agent starts with an explicit approach +
testable assumptions, not just a step list.

**2. Reason WITH the strategy (the heart).** Inject `strategy`, its confidence, and
top `beliefs` into every worker prompt. The policy becomes
`π(action | observation, strategy, beliefs)` instead of `π(action | observation)`.
The strategy is a slowly-varying anchor that gives temporal coherence and lowers
decision entropy — it stops per-step approach oscillation **without** constraining
the action space (full creativity preserved).

**3. Confidence = evidence accumulation.** After each Overwatch verdict:
```
progress:    c ← c + γ(1 − c)        (reinforce toward 1)
no-progress: c ← (1 − γ)·c           (geometric decay)        γ = 0.3
```
Restrategize when `c < τ = 0.4`. Then **3 consecutive** no-progress steps cross
the threshold (1 → .70 → .49 → .34) — matching CriticV12's "few strikes" feel; a
single failure after successes barely dents `c`, so we never thrash. This is the
formal "form *and revise* a hypothesis" mechanism.

**4. Escalation ladder — distinct tactics, provably no repeats.** Fixed ordered
rungs:
```
reperceive → dismiss_overlay → scroll → wait → alternate_element
           → renavigate → vision → restrategize
```
On a stuck signal (CriticV12 circuit-breaker OR loop OR `correction_failures ≥ 2`),
key the obstacle by `<url>|<plan_step>`; same obstacle → advance to next *untried*
rung and write its specific directive into `correction_context`; obstacle changed →
reset to rung 0. **Guarantee:** for a fixed obstacle the rung advances monotonically
through a finite distinct list → at most `|LADDER|` attempts, each a *different*
tactic → "scroll, fail, scroll again" is provably impossible. Only the last rung
(`restrategize`) spends an LLM call, bounded by `MAX_RESTRATEGIZE = 3`; after that
the existing `error_count → recovery → finalize` path guarantees termination.

**5. Proactive reflexion via revived PRM (closes the blind spot).** Throttled
(every 3 steps, or when CriticV12 reports progress) call the **existing**
`PRMCritic.score_step()` for a goal-aware score `g`; push into `goal_score_window`
(len ≤ 4). **Stall test:** if `max(window) − min(window) < 0.05` while CriticV12
keeps reporting change → "busy but not progressing" → confidence penalty + strategy
review. CriticV12 answers "did anything change?" (necessary); PRM-stall answers
"did we get closer?" (sufficient) — both together remove the blind spot. PRM item
completion also makes `commit_node` plan advancement **goal-aware** instead of the
current brittle "OK"-substring match.

---

## Explicitly NOT touched
- MoE routing logic, Overwatch's verification *layers* (only the stuck-directive
  strings change), the Plan-1 model layer, all browser execution primitives.
- The V14 monolith — Plan 2 lands in the V16 brain only.
- WebDreamer's algorithm (schema already fixed in Plan 1).

Any piece that risks destabilizing the working brain is deferred and called out.

---

## Verification
1. **Unit (`test_cognition_v18.py`):** ladder advances monotonically, never repeats
   a tactic per obstacle, resets on obstacle change, terminates at `restrategize`;
   `update_confidence` converges (3 no-progress → below τ; reinforce recovers);
   `detect_stall` fires on a flat window, not a rising one.
2. **Schema:** `StrategicPlan.model_json_schema()` is strict-safe; live planner call
   returns strategy + assumptions + steps.
3. **Integration:** rerun headless Wikipedia (regression — finish ≤ baseline steps)
   AND a harder objective (Amazon flow), capturing `brain_v18_*.log`. Expect:
   strategy + beliefs visible in logs; on an induced stuck state the ladder emits
   *distinct* successive directives (no per-obstacle duplicates); PRM goal-score
   logged per audit; a restrategize occurs and changes the strategy when confidence
   crosses τ; steps-to-success ≤ the pre-cognition run.
