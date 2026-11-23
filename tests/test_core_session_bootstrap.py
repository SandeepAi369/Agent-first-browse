"""Unit tests for Chunk 9: session_bootstrap decoupling and orchestrator helpers."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import advanced_agent
import brain_graph
import core
import core.session_bootstrap as session_bootstrap
import model_registry
import overwatch
import virtual_display
from workers import base_worker

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_brain_graph_has_zero_advanced_agent_imports() -> None:
    """AST-level guarantee that src/brain_graph.py never imports advanced_agent."""
    source = (REPO_ROOT / "src" / "brain_graph.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="src/brain_graph.py")

    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "advanced_agent" or alias.name.startswith("advanced_agent."):
                    offending.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "advanced_agent" or mod.startswith("advanced_agent."):
                offending.append(f"line {node.lineno}: from {mod} import ...")

    assert offending == [], f"Unexpected advanced_agent imports in brain_graph.py: {offending}"


def test_advanced_agent_reexports_session_bootstrap_identities() -> None:
    """Legacy advanced_agent names must alias the exact same objects in core.session_bootstrap."""
    assert advanced_agent.SessionGuard is session_bootstrap.SessionGuard
    assert core.SessionGuard is session_bootstrap.SessionGuard
    assert advanced_agent.launch_browser is session_bootstrap.launch_browser
    assert core.launch_browser is session_bootstrap.launch_browser
    assert advanced_agent._invoke_with_failover is session_bootstrap.invoke_with_failover_compat
    assert core.invoke_with_failover_compat is session_bootstrap.invoke_with_failover_compat


@pytest.mark.asyncio
async def test_session_guard_shutdown_closes_context_and_playwright_once() -> None:
    """SessionGuard.shutdown() closes attached context + playwright and is idempotent."""
    guard = session_bootstrap.SessionGuard()
    fake_ctx = SimpleNamespace(close=AsyncMock())
    fake_pw = SimpleNamespace(stop=AsyncMock())

    guard.attach(fake_ctx, playwright=fake_pw)
    assert guard._context is fake_ctx
    assert guard._playwright is fake_pw

    await guard.shutdown()
    fake_ctx.close.assert_awaited_once()
    fake_pw.stop.assert_awaited_once()
    assert guard._context is None
    assert guard._playwright is None

    # Second call must be a clean no-op
    await guard.shutdown()
    fake_ctx.close.assert_awaited_once()
    fake_pw.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_invoke_with_failover_compat_forwards_breaker_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positional circuit_breaker must map cleanly to keyword-only `breaker`."""
    captured: dict[str, object] = {}

    async def fake_invoke(
        failover_chain,
        messages,
        schema,
        *,
        breaker=None,
        base64_image=None,
        health_tracker=None,
    ):
        captured["chain"] = failover_chain
        captured["messages"] = messages
        captured["schema"] = schema
        captured["breaker"] = breaker
        captured["base64_image"] = base64_image
        captured["health_tracker"] = health_tracker
        return ("ok", "fake-model")

    monkeypatch.setattr(model_registry, "invoke_with_failover", fake_invoke)

    marker_breaker = object()
    marker_health = object()
    result, model_name = await session_bootstrap.invoke_with_failover_compat(
        ["m1"],
        ["msg"],
        dict,
        marker_breaker,
        "img-b64",
        marker_health,
    )

    assert (result, model_name) == ("ok", "fake-model")
    assert captured["breaker"] is marker_breaker
    assert captured["base64_image"] == "img-b64"
    assert captured["health_tracker"] is marker_health


def test_resolve_launch_headless_respects_forced_headless_and_display_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced headless returns True; headed falls back to headless if display unavailable."""
    probe_calls: list[tuple[int, int]] = []

    def fake_display_ok(w: int, h: int) -> bool:
        probe_calls.append((w, h))
        return True

    def fake_display_unavailable(w: int, h: int) -> bool:
        probe_calls.append((w, h))
        return False

    monkeypatch.setattr(virtual_display, "start_virtual_display", fake_display_ok)
    assert brain_graph._resolve_launch_headless(True) is True
    assert probe_calls == []
    assert brain_graph._resolve_launch_headless(False) is False
    assert probe_calls == [(1920, 1080)]

    monkeypatch.setattr(virtual_display, "start_virtual_display", fake_display_unavailable)
    assert brain_graph._resolve_launch_headless(False) is True


@pytest.mark.asyncio
async def test_overwatch_layer_helpers_preserve_verdict_contracts() -> None:
    """Decomposed Overwatch helpers keep exact L1/L2/L3 verdict semantics."""
    # L1 ERROR short-circuit
    err_updates = await overwatch.overwatch_node(
        {"proposed_action": {"verb": "click"}, "page_fsm": "ERROR", "error_count": 2},
        page=None,
        critic=None,
    )
    assert err_updates == {"overwatch_verdict": "rollback", "error_count": 3}

    # L2 skips non-click/type verbs
    l2_updates: dict[str, object] = {}
    assert await overwatch._layer_2_ground_action("wait", {"verb": "wait"}, {}, l2_updates) is True
    assert l2_updates == {}

    # L3 critic verdict: success resets counters; first failure retries
    ok_updates: dict[str, object] = {}
    ok_verdict = SimpleNamespace(success=True, reason="DOM updated", confidence=0.95)
    assert overwatch._evaluate_critic_verdict(ok_verdict, "click", {"critic_progress": 1}, ok_updates) is True
    assert ok_updates["critic_progress"] == 2
    assert ok_updates["correction_failures"] == 0

    fail_updates: dict[str, object] = {}
    fail_verdict = SimpleNamespace(
        success=False,
        reason="no change",
        confidence=0.1,
        circuit_breaker_triggered=False,
    )
    assert overwatch._evaluate_critic_verdict(fail_verdict, "click", {}, fail_updates) is False
    assert fail_updates["overwatch_verdict"] == "retry"
    assert fail_updates["ineffective_streak"] == 1


def test_base_worker_helpers_preserve_target_lock_and_proposal_semantics() -> None:
    """Decomposed base_worker helpers format target-lock blocks and build proposals cleanly."""
    bound, lock_text = base_worker._build_target_lock_section(
        {"bound_target": "Submit Order"},
        "Click Submit Order",
    )
    assert bound is not None
    assert bound.phrase == "Click Submit Order"
    assert {"submit", "order"}.issubset(bound.tokens)
    assert "Submit Order" in lock_text

    decision = base_worker.WorkerAction(
        screen_state="Checkout form visible",
        previous_action_result="Page loaded",
        goal_progress="50%",
        reasoning="Click the primary button",
        expected_change="Confirmation banner appears",
        action_type="click",
        element_id="e1",
    )
    proposed, _risk = base_worker._classify_and_build_proposed(
        decision,
        {"e1": {"name": "Submit", "kind": "button"}},
        "https://example.com/checkout",
    )
    assert proposed["verb"] == "click"
    assert proposed["element_id"] == "e1"
    assert proposed["target_name"] == "Submit"
    assert proposed["expected_change"] == "Confirmation banner appears"
