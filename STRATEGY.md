# Model Strategy — Agentic Capability Gate + Dual-Mode (Free / Premium)

> How the agent chooses, validates, and routes LLMs so that **only models capable
> of agentic structured reasoning ever drive the agent** — while serving two very
> different users equally well: the **free-tier scrapper** (juggling Groq + NVIDIA
> NIM keys against rate limits) and the **premium power user** (one paid, top-tier
> multimodal key, no juggling).

---

## 0. The two users (why one policy isn't enough)

| | **Free-tier scrapper** | **Premium power user** |
|---|---|---|
| Keys | Several free keys (Groq, NVIDIA NIM, Gemini, Cerebras) | One paid key (OpenAI / OpenRouter / Anthropic / Gemini-paid) |
| Pain | Rate limits, dead models, weak models that break JSON | None — the model is reliable and multimodal |
| Needs | Deep fallback, capability gating, probing, role separation | **Get out of the way.** Use my model for everything. |

A single routing policy optimized for the free tier would *punish* a premium
user (needless probes, capability gates, fallback complexity). So the registry
runs in one of two **modes**, chosen once at startup.

```
AGENT_MODE = auto (default) | premium | free
  auto    → premium if a PREMIUM_API_KEY is configured, else free
  premium → force single-key premium path (errors if no premium key)
  free    → force the free-tier juggling path (even if a premium key exists)
```

---

## 1. PREMIUM mode — one key, one model, zero juggling

**Goal:** a premium user brings a single top-tier key whose model natively does
both high-tier agentic text reasoning *and* vision. We trust it and route
everything to it — no capability probe, no fallback hacks, no gates.

**Config (.env):**
```
PREMIUM_API_KEY   = sk-...                      # the paid key
PREMIUM_MODEL     = gpt-5 / claude-opus-4 / gemini-2.5-pro / openrouter/...
PREMIUM_BASE_URL  = https://api.openai.com/v1   # any OpenAI-compatible endpoint
PREMIUM_VISION_MODEL =                          # optional; defaults to PREMIUM_MODEL
PREMIUM_PROVIDER  = openai                       # client family hint (openai|google)
```
*OpenRouter is the canonical "one key for everything" path: one key →
GPT/Claude/Gemini, OpenAI-compatible, vision-capable. Direct OpenAI / Together /
Fireworks / any OpenAI-compatible gateway works identically.*

**Behavior in premium mode:**
- **Text pipeline = Vision pipeline = the premium model.** Multimodal premium
  models serve both; if `PREMIUM_VISION_MODEL` is set it overrides vision.
- **Skip `probe_and_prune` entirely** — no startup round-trip, no risk of
  benching the user's paid model.
- **Skip the capability gate** — a premium model is trusted by definition.
- **Keep** the resilience primitives (circuit breaker, 429 cooldown, adaptive
  timeout, JSON-mode rescue) — they cost nothing when the model is healthy and
  still protect against a transient blip. Multiple keys (comma-separated) are
  allowed for headroom but never required.
- **No role separation** — there's one model; it does every role.

**Auto-detection:** in `AGENT_MODE=auto`, the mere presence of `PREMIUM_API_KEY`
flips the system to premium. A premium user does nothing but paste their key.

---

## 2. FREE mode — the Agentic Capability Gate (six mechanisms)

The free tier's core problem: **most text models can't reliably do agentic
structured reasoning.** They emit malformed JSON, ignore the schema, or reason
poorly — and a model that returns *valid JSON but a bad decision* slips through
every existing check. The fix is a gate where **a model earns its slot by
proving capability**, not by being configured.

### A. Evidence-based allowlist
A curated set of models *proven* (by live probe) to do agentic structured
output. It is the shipped default **and** the safety net (if probing can't run,
fall back to allowlisted-and-alive models so the agent is never bricked).

Proven set (probed 2026-06):
- **Tier 0 (primary):** `gpt-oss-120b` (Groq + NVIDIA), `gpt-oss-20b` (NVIDIA, fast 1.1s)
- **Tier 1 (native API only):** `gemma-4-31b-it` **via Gemini** (fails on NVIDIA — see C)
- **Tier 2 (works but slow, last resort):** `llama-3.3-70b-instruct`,
  `llama-3.3-nemotron-super-49b` (NVIDIA)

### B. Capability probe at startup (upgrade liveness → capability)
`probe_and_prune` currently sends `"pong"` (liveness only) — a model can pass
that and fail every real structured call. **Upgrade it:** send one tiny *real*
structured-output task (a 3-field decision schema + a mock "what's your next
action?" prompt), one rep per (provider, base_model, pipeline). Classify:
- **CAPABLE** — valid structured output (strict OR JSON-mode rescue) → include + seed latency
- **DEAD** — 404 / 410 / not-found → exclude all instances
- **TIMEOUT** — keep (transient) but seed a high latency so it sinks
- **INCAPABLE** — responded but produced no valid structured output → **EXCLUDE**

The INCAPABLE case is the new gate. It is exactly what catches
`gemma-4-31b-it` on NVIDIA (alive, but can't structure) without touching it on
Gemini, where it passes.

**Safety:** if a pipeline ends up empty after gating, restore its
allowlisted-and-alive combos. The agent must always have *some* model.

### C. Capability-derived ordering
The probe's result feeds the existing `(tier, expected_cost)` ordering: CAPABLE
fast models sort first, JSON-mode-only models sink, INCAPABLE/ DEAD are gone.
The hand-tuned tier dict remains the prior; the probe is the evidence that
overrides it per (provider, model) — so `gemma-on-NVIDIA` is dropped while
`gemma-on-Gemini` stays.

### D. Runtime demotion on structured-failure rate
The health tracker already records failures and blacklists schema-400 combos.
Extend it so **malformed JSON in JSON-mode** also counts toward a per-model
structured-failure streak; a model that crosses the threshold mid-session is
quarantined like a 429. Capability is enforced continuously, not just at startup.

### E. Role separation (highest leverage)
The **worker** (action decisions) is where capability matters most; a bad
decision derails the whole task. So the worker draws from a **restricted chain
of tier-0/1 proven models only** (`get_worker_chain()`), while auxiliary calls
(planner, PRM checklist, done-judge) use the full chain. Best models guard the
critical path; cheaper models still do low-stakes work. *(No-op in premium mode —
one model does everything.)*

### F. Schema hygiene
Keep action schemas flat, with clear field descriptions and minimal exotic
constraints (deep nesting / unusual `additionalProperties` combos trip weak
models and even some providers' strict mode). Raises the strict-mode pass rate
before the JSON-mode rescue is needed. The JSON-mode rescue (already present)
remains the backstop.

---

## 3. Where text vs vision keys come from (free mode)

One NVIDIA NIM key unlocks **all** models — text *and* vision (verified: each key
ran `gpt-oss-120b` and `llama-3.2-11b-vision`). The registry keeps text and
vision key pools **separate** (`NVIDIA_NIM_API_KEY` vs `NVIDIA_VISION_API_KEY`)
so their rate budgets are isolated — a burst of vision consults can't starve
text. If only one NVIDIA key exists, vision transparently falls back to it;
clashes stay rare because NVIDIA-text is a *secondary* (behind Groq) and vision
is *on-demand*.

---

## 4. Implementation map

| File | Change |
|---|---|
| `model_registry.py` | Mode detection (`agent_mode`, premium config); `_build_premium_pipeline()`; `AGENTIC_TEXT_ALLOWLIST`; capability probe inside `probe_and_prune` (+ empty-pipeline safety); `get_worker_chain()` (role separation); premium skips probe + gate. |
| `brain_graph.py` | Log the active mode; worker nodes use `get_worker_chain()`, auxiliary keep the full chain. |
| `NEW test_capability_v24.py` | Mode detection + premium bypass; capability classification (capable/dead/incapable); allowlist safety net; role-separation filter; premium skips probe. |
| `STRATEGY.md` | This document. |

**Invariants (must always hold):**
1. Premium mode never probes and never gates — the paid model is trusted.
2. Free mode never ships an empty pipeline — allowlist is the floor.
3. The worker only ever sees capable (tier-0/1) models.
4. Text and vision rate budgets stay isolated when separate keys exist.
5. Existing resilience (breaker, 429 cooldown, adaptive timeout, JSON rescue) is
   preserved in both modes.
