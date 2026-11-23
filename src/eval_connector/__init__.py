"""eval_connector — Zero-API 'Agent-as-LLM' Evaluation Connector for Agent-first-browse.

Provides a shared stateful session controller (`DualModeSessionManager`) that
lets an external AI agent act directly as the LLM brain—reading real `legacy`
and `v2` perception outputs and dispatching every action step-by-step without
requiring external LLM API keys.
"""

from __future__ import annotations

from eval_connector.scenarios import ScenarioSpec, get_scenario, list_scenarios
from eval_connector.session import DualModeSessionManager, EvalSession, TranscriptEntry

__all__ = [
    "DualModeSessionManager",
    "EvalSession",
    "ScenarioSpec",
    "TranscriptEntry",
    "get_scenario",
    "list_scenarios",
]
