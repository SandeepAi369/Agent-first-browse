# 🌐 Agent First Browse

> **An autonomous, vision-capable browser agent that thinks before it acts, verifies before it moves on, and behaves like a real human user — not a script.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/langgraph-1.x-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Playwright](https://img.shields.io/badge/playwright-1.55%2B-green.svg)](https://playwright.dev/python/)
[![Status](https://img.shields.io/badge/status-active%20preview-orange.svg)](#project-status)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

---

## Overview

**Agent First Browse** turns a plain-English instruction — *“search Flipkart for a water bottle under ₹300 and add it to the cart”*, *“star this GitHub repo”*, *“log in and create an API key”* — into a fully autonomous browser session that completes the task on a **real, headed Chromium browser**.

Unlike screen-scrapers or brittle click-bots, it runs on a **LangGraph cognitive “brain”**: a graph of specialized reasoning nodes that perceives the page through the **accessibility tree first** (fast and cheap), escalates to a **vision model only when genuinely confused**, and **verifies every outcome with evidence** before declaring success. It plans, predicts the consequence of each action, recovers from failure with a non-repeating tactical ladder, and presents a human-grade fingerprint to evade bot detection.

In short: it is engineered to be a **critical thinker**, not an obedient parrot — it reasons about *what* to do, *how little* it needs to confirm success, and *when* it is truly done.

---

## Table of Contents

- [Overview](#overview)
- [Why It’s Different](#why-its-different)
- [Key Features & Architecture](#key-features--architecture)
  - [1. The Cognitive Brain (LangGraph)](#1-the-cognitive-brain-langgraph)
  - [2. Accessibility-DOM-First Perception](#2-accessibility-dom-first-perception)
  - [3. Vision-on-Demand (Thinker, not Parrot)](#3-vision-on-demand-thinker-not-parrot)
  - [4. Critical-Thinker Verification](#4-critical-thinker-verification)
  - [5. Self-Curating Model Layer](#5-self-curating-model-layer)
  - [6. Human-Grade Anti-Bot Realism](#6-human-grade-anti-bot-realism)
  - [7. Resilient Execution & Self-Healing](#7-resilient-execution--self-healing)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Configuration & API Keys](#configuration--api-keys)
  - [Operating Modes](#operating-modes)
  - [Text Models](#text-models)
  - [Vision Models (Optional)](#vision-models-optional)
  - [Graceful Fallback](#graceful-fallback)
- [Usage](#usage)
- [Project Status](#project-status)
- [License](#license)

---

## Why It’s Different

Most browser agents do one of two things: drive everything through an expensive vision model (slow, imprecise clicking), or follow rigid DOM selectors (brittle, breaks on any UI change). Agent First Browse was engineered to avoid both traps — and to solve the failure modes those approaches ignore:

| Common failure in browser agents | How Agent First Browse solves it |
| --- | --- |
| Clicks the *wrong* look-alike element on dense pages | **Stable element registry** resolves the exact node the LLM chose, with fresh, drift-proof coordinates |
| “Completes” a task but can’t confirm it, then **loops re-doing it** | **Sticky Verification Ledger** — once a sub-goal is verified, it can never be silently un-completed |
| Misses a button that’s off-screen or rendered as a styled `<div>` | **Primary-action recall** + real-scroll fix surface the goal button even when the DOM hides it |
| Gets flagged as a bot and has its clicks ignored | **Headed-under-Xvfb** real browser + 12-layer fingerprint stealth + trusted OS-level clicks |
| Burns tokens/latency sending every frame to a vision model | **A11y-DOM by default**, vision **only** when the text view is ambiguous |
| Declares success without checking, or never stops | **Evidence-grounded outcome judge** that cites on-page proof |

---

## Key Features & Architecture

### 1. The Cognitive Brain (LangGraph)

The agent is not a simple `observe → act` loop. It is a **stateful graph of cognitive nodes**, each with a single responsibility:

```
Goal Compiler → Planner → Perceive → Router ─► [ Navigator | Interactor | Extractor ]
                                                          │
                                          Overwatch (multi-layer verification)
                                                          │
                          ┌───────────────┬───────────────┼───────────────┐
                        commit          retry           rollback        finalize
```

- **Goal Compiler / Planner** decompose the objective into a strategy, explicit *success criteria* (“done when…”), and an ordered checklist of sub-goals.
- **Mixture-of-Experts Router** dispatches each step to the right specialist worker (navigation, interaction, or data extraction).
- **Overwatch** is the only node allowed to commit state — every proposed action passes through layered verification first.
- A deterministic **escalation ladder** guarantees that a stuck step never repeats a tactic that already failed, and terminates cleanly instead of looping.

### 2. Accessibility-DOM-First Perception

Perception runs in a **single zero-mutation `page.evaluate()` pass (~80 ms)** that produces a compact, LLM-friendly semantic map of the page:

- **Shadow-DOM piercer** captures elements inside open *and* closed shadow roots.
- **Semantic Markdown compression** cuts perception tokens by 60–80% versus raw DOM/HTML.
- **Stable element registry** stamps every interactive element with a durable handle, so an action resolves to the *exact* node the model chose — re-read at click time, immune to layout drift and “snap to the wrong neighbor” errors.
- **Primary-action recall** guarantees commerce/checkout “goal” buttons (Add to Cart, Buy Now, Star, …) are surfaced even when they’re scrolled off-screen or rendered as non-semantic `<div>`s.

### 3. Vision-on-Demand (Thinker, not Parrot)

The agent works from the accessibility DOM by default and **“opens its eyes” only when it genuinely cannot resolve the page from text** — for example, two visually identical buttons where one is disabled, or a canvas-rendered control with no DOM. A single screenshot is sent to a vision model, the answer is mapped back to a **stable element id** (not fuzzy pixels), and the agent **immediately reverts to the a11y DOM**. Vision is strictly **viewport-bounded** — it never sees or maps coordinates onto the surrounding desktop.

### 4. Critical-Thinker Verification

This is the heart of the system — the mechanism that makes the agent trustworthy on multi-step tasks:

- **Forward Modeling (pre-click anticipation).** Before acting, the worker explicitly predicts the exact observable change (*“the button flips to ‘Starred’ and the count increments”*). The verifier then checks reality against that prediction.
- **Adaptive 3-Tier Verification.** The agent uses the *cheapest sufficient* proof, escalating only when needed:
  1. **A11y DOM** — did the predicted structural change happen (state switched, element appeared/vanished, redirect)?
  2. **Vision** — if the DOM is ambiguous for a small visual change, confirm it visually (e.g., “did the star fill?”).
  3. **Path Proof** — only if still unsure, navigate to where the result *undeniably* lives and confirm there.
- **Sticky Verification Ledger.** Once a sub-goal is verified, it is **permanently locked** — a later glance that can’t re-confirm it can never demote it back to “pending.” This is what eliminates the classic *“did it, couldn’t confirm it, did it again”* loop.
- **Task Serialization.** Given multiple tasks, the agent finishes and verifies **one** before it even considers the next.
- **Evidence-Grounded Outcome Judge.** The final “done” gate is a skeptical, independent verdict over the *fresh* page that must **cite concrete on-page proof**; if the proof isn’t there, it returns actionable feedback instead of blindly stopping or blindly looping.

### 5. Self-Curating Model Layer

- **Model-first failover** orders the chain by capability tier and expected cost, exhausting every instance of the best model (across all keys/providers) before falling to a weaker one.
- **Agentic Capability Gate.** At startup, each model is probed with a real structured-reasoning task; any model that can’t reliably produce agentic structured output is **excluded** — so weak models never derail a run. The pipeline is never left empty (a safety floor always remains).
- **Role Separation.** The *worker* (the critical decision-maker) draws only from top-tier proven models; cheaper models handle low-stakes auxiliary calls.
- **Dual-Mode design.** Runs equally well for the **free-tier user** (juggling several free keys with deep fallback) and the **premium user** (one paid, top-tier multimodal key that bypasses all the juggling — no probing, no gates).

### 6. Human-Grade Anti-Bot Realism

- **Headed under a virtual display (Xvfb).** The browser runs as a *genuinely headed* Chromium — no `HeadlessChrome` tell, real window/compositor — while staying invisible on a server/WSL machine.
- **12-layer fingerprint stealth:** `navigator.webdriver` proxy masking, canvas/WebGL/audio noise seeding, **platform-consistent GPU** spoofing, plugin/`chrome` stubs, WebRTC IP sanitization, and font hardening.
- **Trusted, human-like input:** Bézier-curve mouse paths and OS-level CDP clicks (`isTrusted = true`), a **realistic native arrow cursor** with accurate, continuous coordinate awareness, and automatic suppression of the Chrome “didn’t shut down correctly” crash bubble.

### 7. Resilient Execution & Self-Healing

- **Multi-strategy click waterfall** (CDP native event → JS click → direct navigation) with post-action verification.
- **Overlay penetration** to click through cookie banners and modals.
- **Circuit breaker + provider health tracker** quarantine failing models with exponential backoff and recover automatically.
- **Skill memory** records successful workflows for reuse, and a **per-run log file** captures every session for later analysis.

---

## How It Works

```
                ┌─────────────────────────────────────────────────────────┐
   "star this   │  GOAL COMPILER → strategy + success criteria + checklist │
    repo"  ───► │  PLANNER        → ordered, serialized sub-goals          │
                └───────────────────────────┬─────────────────────────────┘
                                            ▼
        ┌──────────────────────────── PER-STEP LOOP ───────────────────────────────┐
        │  PERCEIVE   a11y DOM (fast)  ──► ambiguous? ──► VISION consult (1 shot)    │
        │  DECIDE     worker predicts the exact expected change (forward modeling)   │
        │  EXECUTE    trusted CDP click / type / scroll on the live browser          │
        │  VERIFY     A11y → Vision → Path proof  (cheapest sufficient tier)         │
        │  LEDGER     verified sub-goal is locked — never re-done                    │
        └───────────────────────────────────────────────────────────────────────────┘
                                            ▼
                 OUTCOME JUDGE → cites on-page proof → ✅ done, or 🔁 actionable retry
```

---

## Installation

**Requirements:** Python **3.11+**, Linux / WSL / macOS, and (for stealth mode) the `xvfb` system package.

```bash
# 1. Clone
git clone https://github.com/SandeepAi369/Agent-first-ide.git
cd Agent-first-ide

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -e .

# 4. Install the Chromium browser engine
python -m playwright install chromium

# 5. (Recommended) Install Xvfb for headed "stealth" mode on a display-less machine
sudo apt-get install -y xvfb

# 6. Configure your API keys (see next section)
cp .env.example .env      # then edit .env with your keys

# 7. Make the launcher executable
chmod +x agent.sh
```

> 💡 If `xvfb` is not installed, the agent automatically falls back to headless mode (more bot-detectable, but fully functional).

---

## Configuration & API Keys

All configuration lives in a `.env` file at the project root. The system is provider-agnostic and reads keys for any combination of providers you have.

### Operating Modes

Set `AGENT_MODE` to choose how the model layer behaves:

| `AGENT_MODE` | Behavior |
| --- | --- |
| `auto` *(default)* | **Premium** if a `PREMIUM_API_KEY` is set, otherwise **Free**. |
| `free` | Multi-key free-tier juggling + the Agentic Capability Gate. |
| `premium` | One trusted paid model for **both text and vision** — skips probing/gating entirely. |

**Premium (single-key) setup** — one paid, multimodal key does everything:

```dotenv
AGENT_MODE=premium
PREMIUM_API_KEY=sk-...
PREMIUM_MODEL=gpt-5                            # or claude-opus-4, gemini-2.5-pro, openrouter/...
PREMIUM_BASE_URL=https://api.openai.com/v1     # any OpenAI-compatible endpoint (OpenRouter, etc.)
# PREMIUM_VISION_MODEL=                        # optional; defaults to PREMIUM_MODEL
# PREMIUM_PROVIDER=openai                      # use "google" for the Gemini client
```

### Text Models

The reasoning chain. `gpt-oss-120b` is the proven default and is fast on Groq / Cerebras / NVIDIA.

```dotenv
# Provider keys (any subset; comma-separate multiple keys of one provider)
GROQ_API_KEY=gsk_...
NVIDIA_NIM_API_KEY=nvapi-...
CEREBRAS_API_KEY=csk-...
GEMINI_API_KEY=...

# Optional model overrides
GROQ_MODEL=openai/gpt-oss-120b
NVIDIA_TEXT_MODELS=openai/gpt-oss-120b,openai/gpt-oss-20b   # comma-separated
CEREBRAS_MODEL=gpt-oss-120b
GEMINI_TEXT_MODEL=gemma-4-31b-it
```

#### ⭐ Recommended setup — two keys per provider (rate-limit resilience)

The agent **rotates across keys** and fails over automatically, so providing **two keys for each provider**
keeps it running smoothly when one hits a free-tier rate limit. A solid, fully-free configuration:

```dotenv
# 2× Groq  — primary worker (gpt-oss-120b: fast + accurate)
GROQ_API_KEY=gsk_key1,gsk_key2

# 2× NVIDIA NIM — secondary (gpt-oss-120b / gpt-oss-20b)
NVIDIA_NIM_API_KEY=nvapi-key1
NVIDIA_NIM_API_KEYS=nvapi-key2

# 2× Google Gemini — fallback (Gemma 4 31B; generous free daily quota)
GEMINI_API_KEY=AIza-key1
GEMINI_API_KEY_FALLBACKS=AIza-key2
GEMINI_TEXT_MODEL=gemma-4-31b-it
```

> 💡 **Get the two keys per provider from different accounts/projects.** Free-tier quotas are billed
> **per project**, so two keys from the same project don't add capacity. `gpt-oss-120b` (Groq/NVIDIA) is the
> proven worker; `gemma-4-31b-it` via Gemini is the validated free fallback.

> ⚠️ **A note on API keys & speed.** If your keys are valid and not rate-limited, the agent runs at
> **100% capability and full speed — the best value.** If a key is missing, invalid, or hitting its rate
> limit, the agent does **not** fail — it transparently fails over to the next key/provider, so a task may
> just run **slightly slower** while it juggles limits. For the smoothest experience, keep at least one
> healthy key per provider (two is better).

### Vision Models (Optional)

Vision is **entirely optional** — used only for on-demand visual confirmation. Defaults to the Llama 4 family (Scout on Groq, Maverick on NVIDIA).

```dotenv
# Groq Vision — Llama 4 Scout (primary)
GROQ_VISION_API_KEY=gsk_...
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# NVIDIA Vision — Llama 4 Maverick (+ optional fallbacks, comma-separated)
NVIDIA_VISION_API_KEY=nvapi-...
NVIDIA_VISION_MODELS=meta/llama-4-maverick-17b-128e-instruct
```

### Graceful Fallback

- **No vision keys?** The agent runs **accessibility-DOM only** — it never crashes or requires vision; it simply doesn’t escalate to a screenshot. (NVIDIA vision transparently falls back to your text NVIDIA key if a dedicated vision key isn’t set.)
- **Dead or incapable model?** The startup Capability Gate prunes it automatically; the chain self-curates down to models that actually work.
- **All top models rate-limited?** Failover walks the full chain across every key/provider before giving up.

---

## Usage

The simplest path — the one-click launcher (runs in stealth headed mode automatically):

```bash
./agent.sh "search Flipkart for a water bottle under ₹300 and add it to the cart"
```

Run it with no argument and it will prompt you for the task:

```bash
./agent.sh
```

Or call the brain directly:

```bash
# Stealth headed mode (default — recommended)
.venv/bin/python run_v16.py run "Go to github.com/torvalds/linux and star the repository"

# Force true headless (more bot-detectable)
.venv/bin/python run_v16.py run "your task here" --headless
```

**Persisting a login session** — open a browser to sign in manually once; the session is reused on future runs:

```bash
.venv/bin/python run_v16.py login
```

Every run is saved to `logs/run_<timestamp>.log` for later inspection.

---

## Project Status

Agent First Browse is in **active private preview**. The core architecture — cognitive brain, perception, verification, model layer, and anti-bot stealth — is implemented and covered by an automated test suite. APIs and configuration may still evolve.

---

## Acknowledgements & Credits

Agent First Browse stands on the shoulders of outstanding open-source work and research. Deep thanks to:

- **[LangGraph](https://github.com/langchain-ai/langgraph)** & **[LangChain](https://github.com/langchain-ai/langchain)** — the stateful graph that forms the orchestration spine of the cognitive brain.
- **[Playwright](https://playwright.dev/python/)** — the browser-automation engine (CDP, trusted OS-level input).
- **[Pydantic](https://github.com/pydantic/pydantic)** — typed global state and strict structured-output schemas.
- **Browser-agent projects we studied and learned from** (concepts assimilated and re-implemented cleanly, never copied): **[browser-use](https://github.com/browser-use/browser-use)** (DOM pruning, viewport filtering, CDP event-listener detection), **[Stagehand](https://github.com/browserbase/stagehand)** (act/observe/extract action abstraction), **Skyvern** (multi-signal element identification), **Crawl4AI** (markdown compression), **BrowserGym** (stable element IDs), and **Agent-E** (text-DOM-first navigation).
- **Research that shaped the cognitive layers** (cited inline across the modules): WebDreamer (model-based planning), LATS — Language Agent Tree Search, Reflexion, CISC / self-consistency, Self-Grounded Verification, PABU, Prune4Web, and the MAST / Six-Sigma-Agent reliability analyses.
- **Model providers** for fast, accessible inference: **[Groq](https://groq.com)**, **[NVIDIA NIM](https://build.nvidia.com)**, **[Google Gemini](https://ai.google.dev)**, and **Cerebras**.

If your project or work is reflected here and you'd like different or additional attribution, please open an issue — credit is gladly given.

## License

Licensed under the **GNU General Public License v3.0 (GPLv3)** — see [LICENSE](LICENSE) for the full text.

You are free to use, study, share, and modify this software under the terms of the GPLv3; derivative works
and redistributions must remain licensed under GPLv3 and keep this notice. The software is provided **"as is",
without warranty of any kind**, express or implied.

```
Agent First Browse — an autonomous, vision-capable browser agent.
Copyright (C) 2026  SandeepAi369

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed WITHOUT ANY WARRANTY; see the GNU GPL v3 for details.
```
