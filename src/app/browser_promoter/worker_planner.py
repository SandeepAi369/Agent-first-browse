"""Worker Planner — Vision + Reasoning agents for Agent First IDE v9.0.

Both agents now pull their LLM clients from ModelRegistry (single source of truth).
No more duplicated key-loading logic or missed API keys.

Vision Agent  → uses ModelRegistry.get_vision_chain()  (Gemini×2 → NVIDIA×2)
Reasoning Agent → uses ModelRegistry.get_text_chain()  (Groq×3 → NVIDIA×2)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import zlib
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.browser_promoter.state import BrowserAction

log = logging.getLogger("app.browser_promoter.worker_planner")


class ReasoningDecision(BaseModel):
    """Structured output for the reasoning agent."""

    action: BrowserAction | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confused: bool = False
    confusion_reason: str = ""
    scene_summary: str = ""
    reasoning: str = ""

    model_config = ConfigDict(extra="forbid")


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility Functions (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_screenshot_data_url(encoded: str, encoding: str) -> str:
    """Normalize state screenshot encoding into a standard image data URL."""
    try:
        if encoding == "zlib+base64:jpeg":
            compressed = base64.b64decode(encoded.encode("ascii"), validate=False)
            raw_jpeg = zlib.decompress(compressed)
            plain_base64 = base64.b64encode(raw_jpeg).decode("ascii")
            return f"data:image/jpeg;base64,{plain_base64}"

        if encoding == "base64:jpeg":
            return f"data:image/jpeg;base64,{encoded}"

        if encoding == "base64:png":
            return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""

    return ""


def _extract_response_text(raw_response: Any) -> str:
    """Normalize provider response objects into plain text for JSON parsing."""
    content = getattr(raw_response, "content", raw_response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    return str(content)


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""
    candidate = text.strip()
    if "```json" in candidate:
        candidate = candidate.split("```json", maxsplit=1)[1].split("```", maxsplit=1)[0]
    elif "```" in candidate:
        candidate = candidate.split("```", maxsplit=1)[1].split("```", maxsplit=1)[0]

    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(candidate[start : end + 1])


# ═══════════════════════════════════════════════════════════════════════════════
#  Vision Agent — Uses ModelRegistry VISION chain exclusively
# ═══════════════════════════════════════════════════════════════════════════════

class VisionAgent:
    """Vision model: detect elements from screenshots and return strict JSON mapping.

    Pulls its LLM clients from ModelRegistry.get_vision_chain().
    This ensures ALL vision API keys (Gemini + NVIDIA) are available.
    """

    def __init__(self) -> None:
        from model_registry import ModelRegistry
        registry = ModelRegistry.get_instance()
        self._model_clients = registry.get_vision_chain()
        self._llm_clients = [mc.client for mc in self._model_clients]
        self._client_names = [mc.name for mc in self._model_clients]
        self._cursor = 0
        log.info(
            "VisionAgent: %d vision models loaded — %s",
            len(self._llm_clients),
            " → ".join(self._client_names),
        )

    def _ordered_clients(self) -> list[tuple[int, Any]]:
        if not self._llm_clients:
            return []
        total = len(self._llm_clients)
        start = self._cursor % total
        return [((start + offset) % total, self._llm_clients[(start + offset) % total]) for offset in range(total)]

    async def detect_elements(
        self,
        *,
        screenshot_base64: str,
        screenshot_encoding: str,
        llm_config: dict[str, Any] | None = None,
    ) -> str:
        if not screenshot_base64:
            return "{}"

        data_url = _decode_screenshot_data_url(screenshot_base64, screenshot_encoding)
        if not data_url:
            return "{}"

        if not self._llm_clients:
            log.warning("Vision agent unavailable: no vision models configured in ModelRegistry.")
            return "{}"

        system = (
            "You are a vision-only detector. Return ONLY valid JSON. "
            "Do not include reasoning, explanations, or markdown. "
            "Detect all interactive elements, text labels, and buttons. "
            "Output a JSON object with this schema:\n"
            "{\"elements\":[{\"id\":\"e1\",\"kind\":\"button|input|link|text|other\",\"text\":\"...\",\"x\":123,\"y\":456}],"
            "\"image_size\":{\"width\":1234,\"height\":567}}\n"
            "Use screen pixel coordinates relative to the top-left of the screenshot."
        )
        user = (
            "Return the JSON mapping now. Ensure every element has x and y coordinates."
        )

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=[
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]),
        ]

        for idx, llm in self._ordered_clients():
            client_name = self._client_names[idx] if idx < len(self._client_names) else f"vision:{idx}"
            try:
                raw = await llm.ainvoke(messages, config=llm_config)
                text = _extract_response_text(raw)
                payload = _extract_json_payload(text)
                if isinstance(payload, list):
                    payload = {"elements": payload}
                elif not isinstance(payload, dict):
                    payload = {"elements": []}
                
                # Handle models that return "interactive_elements" instead of "elements"
                if "interactive_elements" in payload and "elements" not in payload:
                    payload["elements"] = payload["interactive_elements"]
                    
                payload.setdefault("elements", [])
                self._cursor = idx
                log.info("Vision map: %d elements detected (by %s)", len(payload["elements"]), client_name)
                return json.dumps(payload, ensure_ascii=True)
            except Exception as exc:
                log.warning("Vision model failed (%s) [%d/%d]: %s", client_name, idx + 1, len(self._llm_clients), exc)

        return "{}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Reasoning Agent — Uses ModelRegistry TEXT chain exclusively
# ═══════════════════════════════════════════════════════════════════════════════

class ReasoningAgent:
    """Large reasoning model: choose action based on goal + vision JSON.

    Pulls its LLM clients from ModelRegistry.get_text_chain().
    This ensures ALL text API keys (Groq + NVIDIA + DeepSeek) are available.
    """

    def __init__(self) -> None:
        from model_registry import ModelRegistry
        registry = ModelRegistry.get_instance()
        self._model_clients = registry.get_text_chain()
        self._clients = self._model_clients  # Keep full ModelClient objects
        self._cursor = 0
        log.info(
            "ReasoningAgent: %d text models loaded — %s",
            len(self._clients),
            " → ".join(mc.name for mc in self._model_clients),
        )

    def get_failover_chain(self) -> list[Any]:
        return list(self._clients)  # Return ModelClient objects

    def get_chain_names(self) -> list[str]:
        return [mc.name for mc in self._clients]

    def _ordered_clients(self) -> list[tuple[int, str, Any]]:
        if not self._clients:
            return []
        total = len(self._clients)
        start = self._cursor % total
        ordered: list[tuple[int, str, Any]] = []
        for offset in range(total):
            idx = (start + offset) % total
            mc = self._clients[idx]
            ordered.append((idx, mc.name, mc.client))
        return ordered

    async def decide_action(
        self,
        *,
        high_level_command: str,
        current_url: str,
        vision_map_json: str,
        action_history: list[str] | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> ReasoningDecision:
        cleaned_command = high_level_command.strip()
        if not cleaned_command:
            return ReasoningDecision(
                confused=True,
                confidence=0.0,
                confusion_reason="Missing high_level_command from supervisor.",
                scene_summary="No command provided.",
            )

        if not self._clients:
            return ReasoningDecision(
                confused=True,
                confidence=0.0,
                confusion_reason="No reasoning model keys configured.",
                scene_summary="Reasoning model unavailable.",
            )

        vision_payload = vision_map_json.strip() or "{}"
        history = action_history or []

        system = (
            "You are the reasoning agent. Use ONLY the provided coordinate map to decide the next step. "
            "Return STRICT JSON for ReasoningDecision with a BrowserAction. "
            "Do not include extra text, markdown, or explanations outside the JSON. "
            "When choosing click/type actions, you MUST include x and y from the map. "
            "Allowed actions: goto, click, type, type_and_enter, scroll, wait, manual_intervention_required, screenshot. "
            "If the target is not present, choose a safe scroll or wait action."
        )

        user_payload = {
            "goal": cleaned_command,
            "current_url": current_url,
            "recent_actions": history[-6:],
            "vision_map": vision_payload,
        }

        messages = [SystemMessage(content=system), HumanMessage(content=json.dumps(user_payload))]

        for idx, name, llm in self._ordered_clients():
            try:
                structured = llm.with_structured_output(ReasoningDecision)
                decision = await structured.ainvoke(messages, config=llm_config)
                if not isinstance(decision, ReasoningDecision):
                    decision = ReasoningDecision.model_validate(decision)
                self._cursor = idx
                return decision
            except Exception as exc:
                log.warning("Reasoning model failed (%s): %s", name, exc)

        return ReasoningDecision(
            confused=True,
            confidence=0.0,
            confusion_reason="All reasoning models failed.",
            scene_summary="Reasoning model failure.",
        )
