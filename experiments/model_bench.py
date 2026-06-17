"""model_bench.py — STANDALONE agentic-decision benchmark for Agent First Browse.

Purpose: discover which text models make the BEST "brain" for the worker — i.e.
given (objective + DOM map + context), emit the correct structured WorkerAction.
This is research/experimentation ONLY: it imports the real WorkerAction schema
(so the test matches production exactly) but it NEVER touches or mutates any brain
code or state. It builds its own LLM clients directly from .env keys.

Metrics per model:
  - struct_ok %  : fraction of cases that returned a schema-valid WorkerAction
  - accuracy %   : fraction whose action matches the case's acceptance rule
  - p50 / p95 latency (s)
  - a composite "brain score" = accuracy · struct_ok (what we actually care about)

Run:  .venv/bin/python experiments/model_bench.py
      .venv/bin/python experiments/model_bench.py --models groq:openai/gpt-oss-120b,nvidia:openai/gpt-oss-20b
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python-orchestrator"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langchain_core.messages import SystemMessage, HumanMessage
from workers.base_worker import WorkerAction  # the EXACT production decision schema


# ═══════════════════════════════════════════════════════════════════════════════
#  Candidate models — provider:model. Dead ones are reported UNAVAILABLE, not fatal.
# ═══════════════════════════════════════════════════════════════════════════════

NVIDIA_BASE = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
GROQ_BASE = "https://api.groq.com/openai/v1"
CEREBRAS_BASE = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")

CANDIDATES: list[tuple[str, str]] = [
    # ── current chain (the incumbents to beat) ──
    ("groq", "openai/gpt-oss-120b"),
    ("nvidia", "openai/gpt-oss-120b"),
    ("nvidia", "openai/gpt-oss-20b"),
    ("gemini", "gemma-4-31b-it"),
    ("nvidia", "google/gemma-4-31b-it"),
    ("cerebras", "gpt-oss-120b"),
    # ── NVIDIA agentic candidates (research: Nemotron 3 purpose-built for agents) ──
    ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
    ("nvidia", "nvidia/nemotron-3-nano-30b-a3b"),
    ("nvidia", "mistralai/mistral-nemotron"),
    ("nvidia", "meta/llama-3.3-70b-instruct"),
    ("nvidia", "qwen/qwen3-235b-a22b"),
    # ── Gemini native (fast, strong instruction following) ──
    ("gemini", "gemini-flash-latest"),
    ("gemini", "gemini-2.0-flash"),
]


def build_client(provider: str, model: str, timeout: float = 25.0):
    """Construct a langchain client for a candidate, or None if no key."""
    if provider in ("nvidia", "groq", "cerebras"):
        from langchain_openai import ChatOpenAI
        key_env, base = {
            "nvidia": ("NVIDIA_NIM_API_KEY", NVIDIA_BASE),
            "groq": ("GROQ_API_KEY", GROQ_BASE),
            "cerebras": ("CEREBRAS_API_KEY", CEREBRAS_BASE),
        }[provider]
        key = (os.getenv(key_env, "").split(",")[0]).strip()
        if not key:
            return None
        return ChatOpenAI(model=model, api_key=key, base_url=base,
                          temperature=0.0, timeout=timeout)
    if provider == "gemini":
        key = (os.getenv("GEMINI_API_KEY", "").split(",")[0]).strip()
        if not key:
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, google_api_key=key,
                                          temperature=0.0, timeout=timeout)
        except ImportError:
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark cases — realistic worker decisions with acceptance rules.
#  Each `dom` mirrors dom_parser markdown ([eN] kind: label ⟨hint⟩ → (x,y)).
# ═══════════════════════════════════════════════════════════════════════════════

def acc_type_e3(a):  # search box
    return a.action_type == "type" and a.element_id == "e3" and bool(a.text)

def acc_disambig(a):  # top story's comments (highest points = e7)
    return a.action_type == "click" and a.element_id == "e7"

def acc_login_user(a):  # first login step: type username into e1
    return a.action_type == "type" and a.element_id == "e1"

def acc_add_backpack(a):  # the Backpack's add button is e5
    return a.action_type == "click" and a.element_id == "e5"

def acc_done(a):  # goal already satisfied
    return a.action_type == "done"

def acc_real_download(a):  # the real one is e7 (ad is e2)
    return a.action_type == "click" and a.element_id == "e7"

def acc_dismiss_or_scroll(a):  # cookie wall e1 must go first
    return (a.action_type == "click" and a.element_id == "e1") or a.action_type == "scroll"

def acc_scroll(a):  # target below the fold
    return a.action_type in ("scroll", "wait")


CASES: list[dict] = [
    {
        "name": "search_box",
        "objective": "Search for 'wireless mouse'.",
        "dom": "## header\n- 🔗 **[e1]** link: Home → (40,20)\n- 🔗 **[e2]** link: Deals → (120,20)\n"
               "- 📝 **[e3]** input: Search products [empty] → (600,20)\n- 🔘 **[e4]** button: Go → (760,20)",
        "accept": acc_type_e3,
    },
    {
        "name": "disambiguate_comments",
        "objective": "Open the comments page of the TOP story (the one with the most points).",
        "dom": "## main\n- 🔗 **[e5]** link: 40 comments ⟨item?id=11 · 88 points by ann⟩ → (360,300)\n"
               "- 🔗 **[e6]** link: 12 comments ⟨item?id=22 · 31 points by bob⟩ → (360,360)\n"
               "- 🔗 **[e7]** link: 210 comments ⟨item?id=33 · 540 points by cara⟩ → (360,240)\n"
               "- 🔗 **[e8]** link: newest ⟨/newest⟩ → (120,20)",
        "accept": acc_disambig,
    },
    {
        "name": "login_form",
        "objective": "Log in with username 'standard_user' and password 'secret_sauce'.",
        "dom": "## form\n- 📝 **[e1]** input: Username [empty] → (400,200)\n"
               "- 📝 **[e2]** input: Password [empty] → (400,260)\n- 🔘 **[e3]** button: Login → (400,330)",
        "accept": acc_login_user,
    },
    {
        "name": "add_to_cart_distractors",
        "objective": "Add ONLY the 'Sauce Labs Backpack' to the cart.",
        "dom": "## inventory\n- 🔗 **[e4]** link: Sauce Labs Bike Light → (200,300)\n"
               "- 🔘 **[e5]** button: Add to cart ⟨in: Sauce Labs Backpack $29.99⟩ → (260,360)\n"
               "- 🔗 **[e6]** link: Sauce Labs Bolt T-Shirt → (560,300)\n"
               "- 🔘 **[e7]** button: Add to cart ⟨in: Sauce Labs Bolt T-Shirt $15.99⟩ → (620,360)",
        "accept": acc_add_backpack,
    },
    {
        "name": "done_recognition",
        "objective": "Add the Backpack to the cart; you are done when the cart badge shows 1.",
        "dom": "## inventory\n- • **[e1]** other: Cart 1 → (980,20)\n"
               "- 🔘 **[e5]** button: Remove ⟨in: Sauce Labs Backpack $29.99⟩ → (260,360)\n"
               "Page text: 'Swag Labs  1  Products'",
        "accept": acc_done,
    },
    {
        "name": "real_vs_ad_download",
        "objective": "Click the real Download button for the file (avoid ads).",
        "dom": "## page\n- 🔘 **[e2]** button: Download Now ⟨in: Advertisement — sponsored⟩ → (300,180)\n"
               "- 🔗 **[e3]** link: Login → (900,20)\n"
               "- 🔘 **[e7]** button: Download ⟨in: file report.pdf 2.1MB official⟩ → (480,520)",
        "accept": acc_real_download,
    },
    {
        "name": "cookie_wall",
        "objective": "Read the article text.",
        "dom": "## dialog(cookie)\n- 🔘 **[e1]** button: Accept all cookies → (640,540)\n"
               "- 🔘 **[e2]** button: Reject → (760,540)\n## article\n- 🔗 **[e3]** link: (article hidden behind consent) → (400,300)",
        "accept": acc_dismiss_or_scroll,
    },
    {
        "name": "scroll_for_hidden",
        "objective": "Click the 'Subscribe' button (it is further down the page).",
        "dom": "## header\n- 🔗 **[e1]** link: Home → (40,20)\n- 🔗 **[e2]** link: About → (120,20)\n"
               "- • **[e3]** other: (long article body — no Subscribe button visible in this viewport) → (400,400)",
        "accept": acc_scroll,
    },
]


SYSTEM = (
    "You are an autonomous browser agent. Given the OBJECTIVE and the PAGE STRUCTURE "
    "(a map of interactive elements as [eN] kind: label ⟨hint⟩ → (x,y)), choose the "
    "single best next action. Prefer element_id over coordinates. Output the "
    "structured action with an honest confidence."
)


def make_messages(case: dict) -> list:
    user = (
        f"═══ OBJECTIVE ═══\n{case['objective']}\n\n"
        f"═══ PAGE STRUCTURE ═══\n{case['dom']}\n\n"
        "Choose the ONE correct next action."
    )
    return [SystemMessage(content=SYSTEM), HumanMessage(content=user)]


# ═══════════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════════

async def run_case(client, case: dict, timeout: float = 25.0):
    """Returns (struct_ok, correct, latency, note)."""
    t0 = time.monotonic()
    try:
        structured = client.with_structured_output(WorkerAction)
        decision = await asyncio.wait_for(
            structured.ainvoke(make_messages(case)), timeout=timeout)
        dt = time.monotonic() - t0
        if not isinstance(decision, WorkerAction):
            decision = WorkerAction.model_validate(decision)
        ok = case["accept"](decision)
        note = f"{decision.action_type}/{decision.element_id or '-'}@{decision.confidence:.2f}"
        return True, ok, dt, note
    except Exception as e:
        dt = time.monotonic() - t0
        return False, False, dt, type(e).__name__ + ": " + str(e)[:60]


async def bench_model(provider: str, model: str):
    client = build_client(provider, model)
    label = f"{provider}:{model}"
    if client is None:
        return {"label": label, "status": "NO_KEY"}

    # Liveness check on the first case
    struct, correct, lat, note = await run_case(client, CASES[0])
    if not struct and any(x in note.lower() for x in
                          ("notfound", "404", "does not exist", "not_found",
                           "permission", "401", "invalid", "decommission")):
        return {"label": label, "status": "UNAVAILABLE", "note": note}

    results = [(struct, correct, lat, note)]
    for case in CASES[1:]:
        results.append(await run_case(client, case))
        await asyncio.sleep(0.3)  # gentle on rate limits

    struct_ok = sum(1 for r in results if r[0]) / len(results)
    acc = sum(1 for r in results if r[1]) / len(results)
    lats = [r[2] for r in results if r[0]]
    p50 = statistics.median(lats) if lats else float("nan")
    p95 = (statistics.quantiles(lats, n=20)[-1] if len(lats) >= 2 else (lats[0] if lats else float("nan")))
    return {
        "label": label, "status": "OK",
        "struct_ok": struct_ok, "accuracy": acc,
        "brain_score": acc * struct_ok, "p50": p50, "p95": p95,
        "detail": [(c["name"], r[1], round(r[2], 1), r[3]) for c, r in zip(CASES, results)],
    }


def _fmt_row(r: dict) -> str:
    if r.get("status") == "OK":
        return (f"  {r['label']:<42} {r['brain_score']*100:>5.0f}% {r['accuracy']*100:>5.0f}% "
                f"{r['struct_ok']*100:>5.0f}% {r['p50']:>6.1f}s {r['p95']:>6.1f}s")
    return f"  {r['label']:<42} {r['status']:>6}  {r.get('note','')[:40]}"


async def main(models: list[tuple[str, str]]):
    print(f"\n🧪 Agentic-decision benchmark — {len(CASES)} cases × {len(models)} models")
    print(f"   schema = production WorkerAction ({len(WorkerAction.model_fields)} fields)")
    print(f"   {'MODEL':<42} {'BRAIN':>6} {'ACC':>6} {'JSON':>6} {'p50':>7} {'p95':>7}\n", flush=True)
    rows = []
    for prov, mod in models:
        try:
            # Hard per-model budget so one slow/hanging model can't eat the run.
            r = await asyncio.wait_for(bench_model(prov, mod), timeout=80.0)
        except asyncio.TimeoutError:
            r = {"label": f"{prov}:{mod}", "status": "SLOW", "note": ">80s for 8 cases"}
        except Exception as e:
            r = {"label": f"{prov}:{mod}", "status": "ERROR", "note": str(e)[:80]}
        rows.append(r)
        print(_fmt_row(r) + "  ←done", flush=True)  # INCREMENTAL: never lose a result

    ok = [r for r in rows if r.get("status") == "OK"]
    ok.sort(key=lambda r: (r["brain_score"], -r["p50"]), reverse=True)

    print("\n" + "═" * 92)
    print(f"  {'MODEL':<42} {'BRAIN':>6} {'ACC':>6} {'JSON':>6} {'p50':>7} {'p95':>7}")
    print("═" * 92)
    for r in ok:
        print(f"  {r['label']:<42} {r['brain_score']*100:>5.0f}% {r['accuracy']*100:>5.0f}% "
              f"{r['struct_ok']*100:>5.0f}% {r['p50']:>6.1f}s {r['p95']:>6.1f}s")
    for r in rows:
        if r.get("status") != "OK":
            print(f"  {r['label']:<42} {r['status']}  {r.get('note','')[:34]}")
    print("═" * 92)

    if ok:
        best = ok[0]
        print(f"\n🏆 Best brain: {best['label']}  (brain_score={best['brain_score']*100:.0f}%, "
              f"acc={best['accuracy']*100:.0f}%, p50={best['p50']:.1f}s)")
        print("\n   Per-case breakdown of the top model:")
        for name, correct, lat, note in best["detail"]:
            print(f"     {'✅' if correct else '❌'} {name:<26} {lat:>4.1f}s  {note}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma list of provider:model to override the candidate set")
    args = ap.parse_args()
    if args.models.strip():
        models = []
        for tok in args.models.split(","):
            if ":" in tok:
                p, m = tok.split(":", 1)
                models.append((p.strip(), m.strip()))
    else:
        models = CANDIDATES
    asyncio.run(main(models))
