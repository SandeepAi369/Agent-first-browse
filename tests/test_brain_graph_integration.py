"""Offline zero-API integration tests for brain_graph.py using ScriptedFakeLLM.

Drives the REAL LangGraph StateGraph (`build_brain_graph(mode='v2')` and
`mode='legacy'`) against live local headless Chromium (`page.set_content()`)
and `model_registry.invoke_with_failover` backed by our content-addressed
`ScriptedFakeLLM` — zero API keys and zero external network calls required.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from playwright.async_api import async_playwright

import brain_graph
import dom_parser
import mcp_tools
import overwatch
from brain_state import BrainState
from core import is_transient_error, try_n
from model_registry import CircuitBreaker, ModelClient, ProviderHealthTracker, invoke_with_failover
from orchestrator.critic_v12 import CriticV12
from fakes.fake_model import Script, ScriptedFakeLLM, Step, Turn
from v2_engine import get_engine_mode, set_engine_mode
from v2_engine.metrics import reset_active_metrics
from v2_engine.perception import get_shared_compressor

_CONFIRM_PAGE_HTML = """<!doctype html>
<html>
  <head><title>Order Confirmation</title></head>
  <body style="font-family: sans-serif; padding: 24px;">
    <h1>Checkout Review</h1>
    <div id="status">Pending confirmation</div>
    <button id="confirm-btn" style="padding: 12px 24px; font-size: 16px;"
      onclick="
        document.getElementById('status').textContent = 'Order #42 Confirmed';
        this.textContent = 'Receipt Approved';
      ">
      Confirm Order
    </button>
  </body>
</html>
"""


def _page_section(prompt_text: str) -> str:
    """Extract only the live PAGE STRUCTURE slice (excluding mission/history text)."""
    if "═══ PAGE STRUCTURE ═══" not in prompt_text:
        return prompt_text
    after = prompt_text.split("═══ PAGE STRUCTURE ═══", 1)[1]
    return after.split("═══ REMEMBER YOUR MISSION ═══", 1)[0]


def _extract_ref_for_label(prompt_text: str, label: str) -> str | None:
    """Locate the live RefID (e.g. 'e0', 'e1') assigned to `label` in the page."""
    for line in _page_section(prompt_text).splitlines():
        if label.lower() in line.lower():
            match = re.search(r"\[(e\d+)\]", line)
            if match:
                return match.group(1)
    return None


async def _brain_invoke_adapter(
    chain: list,
    messages: list,
    schema: type | None = None,
    breaker: Any = None,
    **kwargs: Any,
):
    """Bridge brain_graph's positional `breaker` call shape to real invoke_with_failover."""
    return await invoke_with_failover(
        chain,
        messages,
        schema,
        breaker=breaker,
        **kwargs,
    )


@contextlib.contextmanager
def _wired_offline_brain(
    *,
    page: Any,
    context: Any,
    fake_llm: ScriptedFakeLLM,
    mode: str,
    extra_chain: list[ModelClient] | None = None,
):
    """Temporarily wire brain_graph, mcp_tools, and overwatch to local Chromium + FakeLLM."""
    prev_bg = {
        "_PAGE": brain_graph._PAGE,
        "_CONTEXT": brain_graph._CONTEXT,
        "_FAILOVER_CHAIN": brain_graph._FAILOVER_CHAIN,
        "_WORKER_CHAIN": brain_graph._WORKER_CHAIN,
        "_VISION_CHAIN": brain_graph._VISION_CHAIN,
        "_BREAKER": brain_graph._BREAKER,
        "_HEALTH_TRACKER": brain_graph._HEALTH_TRACKER,
        "_INVOKE_FN": brain_graph._INVOKE_FN,
        "_CRITIC": brain_graph._CRITIC,
        "_DREAMER": brain_graph._DREAMER,
        "_PRM_CRITIC": brain_graph._PRM_CRITIC,
        "_SKILL_MEM": brain_graph._SKILL_MEM,
    }
    prev_mcp_page = mcp_tools._PAGE
    prev_mcp_smap = mcp_tools._SELECTOR_MAP
    prev_judge = (
        overwatch._JUDGE_INVOKE_FN,
        overwatch._JUDGE_CHAIN,
        overwatch._JUDGE_BREAKER,
        overwatch._JUDGE_HEALTH,
    )
    prev_mode = get_engine_mode()

    chain = (
        extra_chain
        if extra_chain is not None
        else [ModelClient("fake:scripted:0", fake_llm, "fake", "text")]
    )
    breaker = CircuitBreaker()
    health = ProviderHealthTracker()

    try:
        set_engine_mode(mode)
        get_shared_compressor().invalidate()
        reset_active_metrics(mode)

        brain_graph._PAGE = page
        brain_graph._CONTEXT = context
        brain_graph._FAILOVER_CHAIN = chain
        brain_graph._WORKER_CHAIN = chain
        brain_graph._VISION_CHAIN = []
        brain_graph._BREAKER = breaker
        brain_graph._HEALTH_TRACKER = health
        brain_graph._INVOKE_FN = _brain_invoke_adapter
        brain_graph._CRITIC = CriticV12(page)
        brain_graph._DREAMER = None
        brain_graph._PRM_CRITIC = None
        brain_graph._SKILL_MEM = None

        mcp_tools.set_page(page)
        mcp_tools.set_selector_map({})
        overwatch.configure_outcome_judge(_brain_invoke_adapter, chain, breaker, health)
        yield
    finally:
        for attr, val in prev_bg.items():
            setattr(brain_graph, attr, val)
        mcp_tools.set_page(prev_mcp_page)
        mcp_tools.set_selector_map(prev_mcp_smap)
        overwatch.configure_outcome_judge(*prev_judge)
        set_engine_mode(prev_mode)
        get_shared_compressor().invalidate()


def _make_confirm_order_script() -> Script:
    """Content-addressed script driving click('Confirm Order') -> done."""

    def _pick_click_action(prompt_text: str, _schema: type | None) -> dict[str, Any]:
        eid = _extract_ref_for_label(prompt_text, "Confirm Order")
        assert eid is not None, f"Could not find Confirm Order ref in prompt:\n{prompt_text[:500]}"
        return {
            "screen_state": "Checkout review page with Confirm Order button.",
            "previous_action_result": "Initial page load.",
            "goal_progress": "50%",
            "reasoning": "Click Confirm Order to complete checkout.",
            "expected_change": "Status updates to Order #42 Confirmed.",
            "action_type": "click",
            "element_id": eid,
        }

    return Script(
        rules=(
            Step(
                schema="StrategicPlan",
                output={
                    "strategy": "Click Confirm Order and verify the confirmation message.",
                    "success_criteria": "Page displays Order #42 Confirmed.",
                    "assumptions": ["Confirm Order button is visible."],
                    "steps": ["Click Confirm Order button", "Verify Order #42 Confirmed"],
                },
            ),
            Step(
                schema="WorkerAction",
                match=lambda p: "Receipt Approved" in _page_section(p),
                output={
                    "screen_state": "Button now shows Receipt Approved and order is confirmed.",
                    "previous_action_result": "Click succeeded and confirmed the order.",
                    "goal_progress": "100%",
                    "reasoning": "The confirmation state is visible on the live page.",
                    "expected_change": "",
                    "action_type": "done",
                    "proof_of_completion": "Receipt Approved / Order #42 Confirmed",
                },
            ),
            Step(
                schema="WorkerAction",
                output=_pick_click_action,
            ),
            Step(
                schema="RealityCheck",
                output={
                    "matches_prediction": True,
                    "contradiction": False,
                    "note": "Order #42 Confirmed appeared on page.",
                },
            ),
            Step(
                schema="DoneVerdict",
                output={
                    "achieved": True,
                    "confidence": 0.99,
                    "evidence": "Page displays 'Order #42 Confirmed' and button is Confirmed.",
                    "missing": "",
                    "next_hint": "",
                },
            ),
        )
    )


# ── 1. Content-Addressed FakeModel Determinism & Retry Tests ───────────────────


async def test_scripted_fake_llm_content_addressed_turns_and_retry_determinism():
    script = Script(
        turns=(
            Turn(
                id="t1",
                steps=(
                    Step(output="step-0-reply", fail_times=1, error=TimeoutError("transient 503")),
                    Step(output="step-1-reply"),
                ),
            ),
        )
    )
    fake = ScriptedFakeLLM(script)
    turn_messages = [HumanMessage(content="[t1] Say hello.")]

    # Retry on exact same content recovers after 1 transient failure
    first = await try_n(
        2,
        lambda _attempt: fake.ainvoke(turn_messages),
        base_delay_ms=1.0,
        max_delay_ms=5.0,
        should_retry=lambda exc, _n: is_transient_error(exc),
    )
    assert first.content == "step-0-reply"

    # Re-sending identical messages again (e.g. caller retry) still resolves to step 0
    retry_same = await fake.ainvoke(turn_messages)
    assert retry_same.content == "step-0-reply"

    # Adding 1 AIMessage after [t1] deterministically advances to step 1
    next_step = await fake.ainvoke([*turn_messages, AIMessage(content="step-0-reply")])
    assert next_step.content == "step-1-reply"
    assert [e.outcome for e in fake.request_log] == ["error", "ok", "ok", "ok"]


# ── 2. Offline Real-Browser Brain Graph E2E (V2 and Legacy) ────────────────────


@pytest.mark.parametrize("mode", ["v2", "legacy"])
async def test_brain_graph_end_to_end_real_browser_click_and_complete(mode, monkeypatch):
    # Keep single-model test deterministic (no multi-model consensus/webdreamer fan-out)
    monkeypatch.setenv("V29_CLARITY_CONSENSUS", "0")
    monkeypatch.setenv("V29_WEBDREAMER", "0")

    fake_llm = ScriptedFakeLLM(_make_confirm_order_script())

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        await dom_parser.install_shadow_piercer(context)
        page = await context.new_page()
        await page.set_content(_CONFIRM_PAGE_HTML, wait_until="domcontentloaded")

        try:
            with _wired_offline_brain(page=page, context=context, fake_llm=fake_llm, mode=mode):
                compiled = brain_graph.build_brain_graph(mode=mode).compile(
                    checkpointer=InMemorySaver()
                )
                initial = BrainState(
                    objective="Click Confirm Order and verify Order #42 Confirmed appears",
                    max_steps=6,
                )
                final = await compiled.ainvoke(
                    initial,
                    config={"configurable": {"thread_id": f"offline-{mode}"}},
                )

            assert final["mission_success"] is True
            assert await page.text_content("#status") == "Order #42 Confirmed"
            schemas_called = [e.schema_name for e in fake_llm.request_log]
            assert "StrategicPlan" in schemas_called
            assert "WorkerAction" in schemas_called
            assert "DoneVerdict" in schemas_called
        finally:
            await context.close()
            await browser.close()


# ── 3. Offline Failover Fault Recovery Inside Real Brain Graph ─────────────────


async def test_brain_graph_v2_recovers_from_primary_llm_transient_fault(monkeypatch):
    monkeypatch.setenv("V29_CLARITY_CONSENSUS", "0")
    monkeypatch.setenv("V29_WEBDREAMER", "0")

    # Primary model always times out; secondary scripted model succeeds seamlessly
    failing_primary = ScriptedFakeLLM(
        Script(
            rules=(
                Step(
                    output={},
                    fail_times=99,
                    error=TimeoutError("primary 503 timeout"),
                ),
            )
        )
    )
    healthy_backup = ScriptedFakeLLM(_make_confirm_order_script())

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        await dom_parser.install_shadow_piercer(context)
        page = await context.new_page()
        await page.set_content(_CONFIRM_PAGE_HTML, wait_until="domcontentloaded")

        try:
            chain = [
                ModelClient("fake:primary:0", failing_primary, "fake", "text"),
                ModelClient("fake:backup:0", healthy_backup, "fake", "text"),
            ]
            with _wired_offline_brain(
                page=page,
                context=context,
                fake_llm=healthy_backup,
                mode="v2",
                extra_chain=chain,
            ):
                compiled = brain_graph.build_brain_graph(mode="v2").compile(
                    checkpointer=InMemorySaver()
                )
                final = await compiled.ainvoke(
                    BrainState(
                        objective="Click Confirm Order and confirm order #42",
                        max_steps=6,
                    ),
                    config={"configurable": {"thread_id": "offline-failover"}},
                )

            assert final["mission_success"] is True
            assert await page.text_content("#status") == "Order #42 Confirmed"
            assert any(e.outcome == "error" for e in failing_primary.request_log)
            assert any(e.outcome == "ok" for e in healthy_backup.request_log)
        finally:
            await context.close()
            await browser.close()


# ── 4. Offline Blocked Domain Enforcement & Max-Step Termination ───────────────


async def test_brain_graph_blocks_dangerous_domain_and_terminates_cleanly(monkeypatch):
    monkeypatch.setenv("V29_CLARITY_CONSENSUS", "0")
    monkeypatch.setenv("V29_WEBDREAMER", "0")

    blocked_script = Script(
        rules=(
            Step(
                schema="StrategicPlan",
                output={
                    "strategy": "Attempt navigation.",
                    "success_criteria": "Reached target.",
                    "assumptions": [],
                    "steps": ["Navigate to bank site"],
                },
            ),
            Step(
                schema="WorkerAction",
                output={
                    "screen_state": "Blank page.",
                    "previous_action_result": "None.",
                    "goal_progress": "0%",
                    "reasoning": "Try going to blocked banking domain.",
                    "expected_change": "Navigate to banking login.",
                    "action_type": "goto",
                    "url": "https://chase.com/login",
                },
            ),
        )
    )
    fake_llm = ScriptedFakeLLM(blocked_script)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        await page.set_content("<p>Safe local page</p>", wait_until="domcontentloaded")

        try:
            with _wired_offline_brain(page=page, context=context, fake_llm=fake_llm, mode="v2"):
                compiled = brain_graph.build_brain_graph(mode="v2").compile(
                    checkpointer=InMemorySaver()
                )
                final = await compiled.ainvoke(
                    BrainState(
                        objective="Open safe local page only",
                        max_steps=2,
                    ),
                    config={"configurable": {"thread_id": "offline-blocked"}},
                )

            # Navigation to blocked banking domain must never occur
            assert "chase.com" not in page.url
            assert final["mission_success"] is False
        finally:
            await context.close()
            await browser.close()
