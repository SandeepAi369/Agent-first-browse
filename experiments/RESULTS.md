# Model Brain Benchmark — Results (Agent First Browse)

Standalone agentic-decision benchmark (`experiments/model_bench.py`). 8 real browser-decision
cases (search, element disambiguation, login, add-to-cart among distractors, done-recognition,
real-vs-ad button, cookie wall, scroll-for-hidden) × the production `WorkerAction` schema.
Metrics: **brain_score = accuracy × JSON-reliability**, plus p50/p95 latency. Live API calls,
no brain code touched. Date: 2026-06-15.

## Full ranking (both batches)

| Model | brain | acc | JSON | p50 | p95 | verdict |
|---|---|---|---|---|---|---|
| **groq : gpt-oss-120b** | **100%** | 100% | 100% | **1.5s** | 6.4s | 🏆 best overall — fast + perfect |
| **gemini : gemma-4-31b-it** | **100%** | 100% | 100% | 6.1s | 7.3s | excellent free fallback |
| **gemini : gemini-flash-latest** | **100%** | 100% | 100% | 2.9s | **21.1s** | perfect but latency-spiky (thinking mode) |
| nvidia : gpt-oss-20b | 88% | 88% | 100% | **1.0s** | 3.4s | fastest reliable — great cheap voter |
| nvidia : gpt-oss-120b | 88% | 88% | 100% | 3.0s | 3.9s | solid |
| cerebras : gpt-oss-120b | 56% | 75% | **75%** | 0.7s | 1.5s | ⚠️ fastest but drops JSON 1-in-4 |
| nvidia : gemma-4-31b-it | — | — | — | — | — | ⚠️ SLOW >80s/8 (use Gemma via Gemini) |
| nvidia : nemotron-3-nano-30b | 19% | 25% | 75% | 2.7s | 6.0s | ❌ poor accuracy despite agentic marketing |
| nvidia : nemotron-3-super-120b | — | — | — | — | — | ❌ SLOW >10s/call (thinking mode) |
| nvidia : mistral-nemotron | — | — | — | — | — | ❌ SLOW >10s/call |
| nvidia : llama-3.3-70b-instruct | — | — | — | — | — | ❌ SLOW >10s/call |
| nvidia : qwen3-235b-a22b | — | — | — | — | — | ❌ 404 dead |
| gemini : gemini-2.0-flash | — | — | — | — | — | ❌ SLOW (use gemini-flash-latest instead) |

## Key findings (multiple angles)

1. **The current primary is already optimal.** `groq:gpt-oss-120b` scored a perfect 8/8 at 1.5s —
   it correctly disambiguated the top story (clicked the 540-point comments link, not a nav link),
   avoided the ad "Download", and recognized the done-state. No change needed.

2. **Gemma 4 is validated — but only via Gemini.** `gemini:gemma-4-31b-it` is a perfect 100%
   fallback; `nvidia:google/gemma-4-31b-it` is unusably slow (>80s). Lesson: route Gemma 4 through
   the Gemini API, never NVIDIA NIM.

3. **The NVIDIA Nemotron agentic models did NOT beat the incumbents — the opposite.** Despite
   NVIDIA's "purpose-built for agentic AI" marketing, `nemotron-3-nano` scored **19%** (25% accuracy)
   and `nemotron-3-super` / `mistral-nemotron` were too slow (>10s/call) for a real-time browser
   agent. Marketing ≠ task fit. Switching to Nemotron would REDUCE accuracy.

4. **Cerebras has a structured-output reliability problem.** `cerebras:gpt-oss-120b` is the fastest
   (0.7s) but failed JSON on 2/8 cases (75% reliability) → 56% brain. It sits tier-0 in the chain;
   it leans on the V17 JSON-mode rescue. Consider deprioritizing it below Groq/NVIDIA for the worker.

5. **`gpt-oss-20b` is a hidden gem for speed.** 1.0s p50 (fastest *reliable* model), 88% accuracy,
   100% JSON. Ideal as a fast tier and as a cheap, diverse consensus voter.

6. **Confidence calibration differs (matters for P2/CISC consensus).** `gpt-oss-120b` reported varied,
   calibrated confidence (0.85–0.95); `gemini-flash` reported 1.00 on nearly everything (overconfident).
   For confidence-weighted consensus voting, the calibrated `gpt-oss` family is the better signal.

## Recommendation for the agent's brain chain (no code change made — proposal only)

- **Worker (critical path):** keep `groq:gpt-oss-120b` primary → `gemini:gemma-4-31b-it` →
  `nvidia:gpt-oss-20b` (fast). These three are distinct base-models, all ≥88%, all 100% JSON —
  which **also makes them ideal P2 consensus voters** (fixes the 2-voter-tie issue with a strong 3rd).
- **Deprioritize** `cerebras:gpt-oss-120b` for worker decisions (JSON reliability); fine for non-critical.
- **Do NOT adopt** Nemotron/Mistral-Nemotron/Llama-3.3-70B as the worker brain — slower and/or less
  accurate for this task. (They may still suit non-real-time auxiliary roles.)
- **Drop dead entries** from any candidate list: `qwen3-235b` (404), `nvidia:gemma-4-31b-it` (slow).

## Caveats
Small benchmark (8 cases) → directional, not statistically tight; latency reflects this network/time
window (NVIDIA "thinking" models inflate p95). The benchmark is re-runnable any time:
`.venv/bin/python experiments/model_bench.py [--models prov:model,...]`.
