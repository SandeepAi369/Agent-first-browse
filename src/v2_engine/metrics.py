"""v2_engine.metrics — Structured latency, token, and reliability telemetry.

Tracks per-step measurements so legacy and v2 runs can be compared apples-to-apples
on settle time, extraction latency, prompt token footprint, occlusion pruning,
delta compression hit rate, and action retry / stale-ref recovery counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def estimate_tokens(text: str) -> int:
    """Fast, model-agnostic token estimate (~4 chars/token for English/UI text)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass
class StepMetric:
    step: int
    url: str
    mode: str
    settle_ms: float = 0.0
    extract_ms: float = 0.0
    total_perceive_ms: float = 0.0
    raw_elements: int = 0
    emitted_elements: int = 0
    occluded_filtered: int = 0
    offscreen_dropped: int = 0
    markdown_chars: int = 0
    est_tokens: int = 0
    snapshot_kind: str = "full"  # "full" | "delta" | "tier2_a11y"
    tier_used: int = 1
    action_retries: int = 0
    stale_ref_recoveries: int = 0


@dataclass
class RunMetrics:
    mode: str = "legacy"
    steps: list[StepMetric] = field(default_factory=list)
    total_action_retries: int = 0
    total_stale_ref_recoveries: int = 0
    total_state_safety_overrides: int = 0

    def record_step(self, metric: StepMetric) -> None:
        self.steps.append(metric)

    def record_action_retry(self) -> None:
        self.total_action_retries += 1

    def record_stale_ref_recovery(self) -> None:
        self.total_stale_ref_recoveries += 1

    def record_state_safety_override(self) -> None:
        self.total_state_safety_overrides += 1

    def summary(self) -> dict[str, Any]:
        n = len(self.steps)
        if n == 0:
            return {
                "mode": self.mode,
                "steps": 0,
                "avg_settle_ms": 0.0,
                "avg_extract_ms": 0.0,
                "avg_perceive_ms": 0.0,
                "total_perceive_ms": 0.0,
                "avg_markdown_chars": 0.0,
                "total_est_tokens": 0,
                "avg_est_tokens": 0.0,
                "total_occluded_filtered": 0,
                "delta_snapshots": 0,
                "tier2_escalations": 0,
                "action_retries": self.total_action_retries,
                "stale_ref_recoveries": self.total_stale_ref_recoveries,
                "state_safety_overrides": self.total_state_safety_overrides,
            }

        total_settle = sum(s.settle_ms for s in self.steps)
        total_extract = sum(s.extract_ms for s in self.steps)
        total_perceive = sum(s.total_perceive_ms for s in self.steps)
        total_chars = sum(s.markdown_chars for s in self.steps)
        total_tokens = sum(s.est_tokens for s in self.steps)

        return {
            "mode": self.mode,
            "steps": n,
            "avg_settle_ms": round(total_settle / n, 1),
            "avg_extract_ms": round(total_extract / n, 1),
            "avg_perceive_ms": round(total_perceive / n, 1),
            "total_perceive_ms": round(total_perceive, 1),
            "avg_markdown_chars": round(total_chars / n, 1),
            "total_est_tokens": total_tokens,
            "avg_est_tokens": round(total_tokens / n, 1),
            "total_occluded_filtered": sum(s.occluded_filtered for s in self.steps),
            "delta_snapshots": sum(1 for s in self.steps if s.snapshot_kind == "delta"),
            "tier2_escalations": sum(1 for s in self.steps if s.tier_used >= 2),
            "action_retries": self.total_action_retries,
            "stale_ref_recoveries": self.total_stale_ref_recoveries,
            "state_safety_overrides": self.total_state_safety_overrides,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "steps": [asdict(s) for s in self.steps],
        }


_ACTIVE_METRICS: RunMetrics | None = None


def reset_active_metrics(mode: str) -> RunMetrics:
    global _ACTIVE_METRICS
    _ACTIVE_METRICS = RunMetrics(mode=mode)
    return _ACTIVE_METRICS


def get_active_metrics() -> RunMetrics | None:
    return _ACTIVE_METRICS
