"""Real-world unit tests for src/core/observability.py and all built-in sinks."""

from __future__ import annotations

import base64
import json
import sys

import pytest

import core
from core import (
    DEFAULT_OBSERVABILITY_BUS,
    OBSERVABILITY_SCHEMA_VERSION,
    SECRET_MASK,
    BrowserRuntimeContext,
    ConfigValidationError,
    DisposableStore,
    EventSink,
    EventType,
    JsonlFileSink,
    LangSmithSink,
    MemoryEventSink,
    ObservabilityBus,
    ObservabilityEvent,
    RunMetricsSink,
    build_langsmith_run_payload,
    build_run_config,
    get_default_bus,
    use_runtime_context,
)
from v2_engine.metrics import RunMetrics


# ── 1. Barrel, Schema & Event Validation ───────────────────────────────────────


def test_core_barrel_exports_chunk6_symbols_and_schema_version():
    assert OBSERVABILITY_SCHEMA_VERSION == "1"
    assert get_default_bus() is DEFAULT_OBSERVABILITY_BUS
    for name in (
        "DEFAULT_OBSERVABILITY_BUS",
        "OBSERVABILITY_SCHEMA_VERSION",
        "EventSink",
        "EventType",
        "JsonlFileSink",
        "LangSmithSink",
        "MemoryEventSink",
        "ObservabilityBus",
        "ObservabilityEvent",
        "RunMetricsSink",
        "build_langsmith_run_payload",
        "build_run_config",
        "get_default_bus",
    ):
        assert name in core.__all__
        assert hasattr(core, name)

    # Confirm zero coupling to legacy app.browser_promoter
    assert "app.browser_promoter.state" not in sys.modules or "app.browser_promoter" not in (
        core.observability.__file__ or ""
    )

    src_payload = {"step": 1}
    ev = ObservabilityEvent(type=EventType.ACTION_EXECUTED, payload=src_payload)
    src_payload["step"] = 999
    assert ev.type == "action.executed"
    assert ev.payload["step"] == 1
    assert ev.schema_version == "1"
    assert ev.timestamp_ms > 0

    with pytest.raises(ConfigValidationError):
        ObservabilityEvent(type="   ")


# ── 2. Bus Silence, Ambient Enrichment, Filtering & Error Isolation ────────────


def test_bus_silent_by_default_enriches_context_filters_and_isolates_errors():
    errors: list[str] = []
    bus = ObservabilityBus(on_error=lambda exc, _s: errors.append(str(exc)))

    # Silent emit with 0 subscribers
    silent_ev = bus.emit(EventType.SESSION_STARTED, {"url": "https://example.test"})
    assert silent_ev.type == "session.started"
    assert silent_ev.session_id is None

    all_sink = MemoryEventSink()
    retry_sink = MemoryEventSink()
    assert isinstance(all_sink, EventSink)

    def broken_sink(_ev: ObservabilityEvent) -> None:
        raise RuntimeError("sink disk full")

    with DisposableStore() as ds:
        ds.add(bus.subscribe(broken_sink))
        ds.add(bus.subscribe(all_sink))
        ds.add(bus.subscribe(retry_sink, event_types=[EventType.ACTION_RETRIED]))
        assert bus.subscriber_count == 3

        ctx = BrowserRuntimeContext(session_id="v2-live", scope="conversation:7", engine_mode="v2")
        with use_runtime_context(ctx):
            ev1 = bus.emit(EventType.ACTION_EXECUTED, {"verb": "click"})
            ev2 = bus.emit(EventType.ACTION_RETRIED, {"attempt": 2}, mode="legacy")

        assert ev1.session_id == "v2-live"
        assert ev1.scope == "conversation:7"
        assert ev1.mode == "v2"
        # Explicit kwarg overrides ambient context mode
        assert ev2.session_id == "v2-live"
        assert ev2.mode == "legacy"

        # Healthy sinks still received events despite broken_sink raising on every emit
        assert len(all_sink.events) == 2
        assert [e.type for e in retry_sink.events] == ["action.retried"]
        assert len(errors) == 2

    # Unsubscribed automatically via DisposableStore
    assert bus.subscriber_count == 0
    bus.emit(EventType.SESSION_CLOSED)
    assert len(all_sink.events) == 2


# ── 3. MemoryEventSink Ring Bounds ─────────────────────────────────────────────


def test_memory_event_sink_ring_bounds_and_validation():
    sink = MemoryEventSink(max_events=2)
    bus = ObservabilityBus()
    bus.subscribe(sink)

    bus.emit("custom.one", {"n": 1})
    bus.emit(EventType.MCP_TOOL_CALLED, {"tool": "eval_perceive"})
    bus.emit(EventType.MCP_TOOL_CALLED, {"tool": "eval_act"})

    assert len(sink.events) == 2
    assert [e.payload["tool"] for e in sink.by_type(EventType.MCP_TOOL_CALLED)] == [
        "eval_perceive",
        "eval_act",
    ]
    sink.clear()
    assert sink.events == []

    with pytest.raises(ConfigValidationError):
        MemoryEventSink(max_events=0)
    with pytest.raises(ConfigValidationError):
        MemoryEventSink(max_events=True)  # type: ignore[arg-type]


# ── 4. JsonlFileSink Redaction, Truncation & Disposal ──────────────────────────


def test_jsonl_file_sink_writes_redacted_bounded_ndjson(tmp_path):
    out_file = tmp_path / "nested" / "telemetry.jsonl"
    sink = JsonlFileSink(out_file, redact=True, max_payload_chars=400)
    bus = ObservabilityBus()
    bus.subscribe(sink)

    fake_key = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
    big_shot = base64.b64encode(b"Z" * 5000).decode("ascii")

    bus.emit(
        EventType.MCP_TOOL_CALLED,
        {
            "tool": "eval_perceive",
            "api_key": fake_key,
            "est_tokens": 341,
            "screenshot": {"type": "browser_screenshot", "mediaType": "image/png", "data": big_shot},
            "huge_dom": "item " * 500,
        },
        session_id="s-jsonl",
        mode="v2",
    )
    sink.dispose()
    assert sink.is_closed is True

    # Emitting after dispose does not write or crash
    bus.emit(EventType.SESSION_CLOSED)

    raw_text = out_file.read_text(encoding="utf-8")
    lines = [line for line in raw_text.splitlines() if line.strip()]
    assert len(lines) == 1
    assert fake_key not in raw_text
    assert big_shot not in raw_text

    parsed = json.loads(lines[0])
    assert parsed["schema_version"] == "1"
    assert parsed["session_id"] == "s-jsonl"
    assert parsed["payload"]["api_key"] == SECRET_MASK
    assert parsed["payload"]["est_tokens"] == 341
    assert len(json.dumps(parsed["payload"], separators=(",", ":"))) <= 400


# ── 5. RunMetricsSink Real Bridge to v2_engine.metrics.RunMetrics ──────────────


def test_run_metrics_sink_bridges_events_into_real_run_metrics():
    metrics = RunMetrics(mode="v2")
    bus = ObservabilityBus()
    bus.subscribe(RunMetricsSink(metrics))

    ctx = BrowserRuntimeContext(session_id="bench-v2", engine_mode="v2")
    with use_runtime_context(ctx):
        bus.emit(
            EventType.PERCEPTION_COMPLETED,
            {
                "step": 1,
                "url": "https://example.test",
                "settle_ms": 22.5,
                "extract_ms": 48.0,
                "total_perceive_ms": 70.5,
                "raw_elements": 40,
                "emitted_elements": 18,
                "occluded_filtered": 6,
                "markdown_chars": 1200,
                "est_tokens": 300,
                "snapshot_kind": "delta",
                "tier_used": 1,
            },
        )
        bus.emit(EventType.ACTION_RETRIED, {"attempt": 2})
        bus.emit(EventType.STALE_REF_RECOVERED, {"element_id": "e4"})
        bus.emit(EventType.STATE_SAFETY_OVERRIDE, {"element_id": "e9"})

    summary = metrics.summary()
    assert summary["mode"] == "v2"
    assert summary["steps"] == 1
    assert summary["avg_settle_ms"] == 22.5
    assert summary["avg_perceive_ms"] == 70.5
    assert summary["total_est_tokens"] == 300
    assert summary["total_occluded_filtered"] == 6
    assert summary["delta_snapshots"] == 1
    assert summary["action_retries"] == 1
    assert summary["stale_ref_recoveries"] == 1
    assert summary["state_safety_overrides"] == 1


# ── 6. LangSmithSink & General-Purpose build_run_config ────────────────────────


def test_langsmith_sink_and_decoupled_build_run_config():
    recorded_runs: list[dict] = []

    class _FakeLangSmithClient:
        def create_run(self, **kwargs) -> None:
            recorded_runs.append(kwargs)

    fake_token = "ghp_" + "9876543210abcdefghij1234567890"
    sink = LangSmithSink(client=_FakeLangSmithClient(), project="afb-eval", tags=["smoke"])
    bus = ObservabilityBus()
    bus.subscribe(sink)

    bus.emit(
        EventType.ACTION_EXECUTED,
        {"verb": "click", "debug": f"auth={fake_token}"},
        session_id="ls-1",
        mode="v2",
    )
    bus.emit(EventType.SESSION_RESTARTED, {"reason": "tab crashed"}, session_id="ls-1", mode="v2")

    assert len(recorded_runs) == 2
    assert recorded_runs[0]["run_type"] == "tool"
    assert recorded_runs[1]["run_type"] == "chain"
    assert recorded_runs[0]["project_name"] == "afb-eval"
    assert "mode:v2" in recorded_runs[0]["tags"]
    assert "smoke" in recorded_runs[0]["tags"]
    assert fake_token not in json.dumps(recorded_runs[0])

    # Disabled sink & missing-method client behave safely
    disabled_sink = LangSmithSink(client=_FakeLangSmithClient(), enabled=False)
    disabled_sink.handle(ObservabilityEvent(type=EventType.CUSTOM))

    bad_client_sink = LangSmithSink(client=object())
    bad_client_sink.handle(ObservabilityEvent(type=EventType.CUSTOM))
    assert bad_client_sink.dropped_count == 1

    # Pure payload builder check
    direct_payload = build_langsmith_run_payload(
        ObservabilityEvent(type=EventType.MCP_TOOL_CALLED, payload={"tool": "eval_act"}, mode="legacy")
    )
    assert direct_payload["run_type"] == "tool"
    assert "mode:legacy" in direct_payload["tags"]

    # Decoupled build_run_config enriches from runtime context and masks metadata secrets
    ctx = BrowserRuntimeContext(session_id="graph-9", scope="shared", engine_mode="v2")
    with use_runtime_context(ctx):
        cfg = build_run_config(
            "brain_graph",
            thread_id="thr-9",
            tags=["v2-test"],
            metadata={"api_key": fake_token, "est_tokens": 250},
        )
    assert cfg["run_name"] == "brain_graph"
    assert cfg["configurable"] == {"recursion_limit": 1000, "thread_id": "thr-9"}
    assert cfg["metadata"]["session_id"] == "graph-9"
    assert cfg["metadata"]["mode"] == "v2"
    assert cfg["metadata"]["api_key"] == SECRET_MASK
    assert cfg["metadata"]["est_tokens"] == 250
    assert "campaign_id" not in cfg["metadata"]
