from __future__ import annotations

import base64
import json
import os
import zlib
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.browser_promoter.state import BrowserAction, InteractiveRegion


class WorkerDecision(BaseModel):
    """
    Structured decision object returned by the small Worker VLM.

    `action` must conform to BrowserAction when present.
    """

    action: BrowserAction | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confused: bool = False
    confusion_reason: str = ""
    scene_summary: str = ""

    model_config = ConfigDict(extra="forbid")


class WorkerVisionPlanner:
    """LangChain-based planner that maps high-level intent to physical browser action."""

    def __init__(self) -> None:
        self.model_name = os.getenv("WORKER_VLM_MODEL", "gpt-4.1-mini")
        self.api_key = os.getenv("WORKER_VLM_API_KEY") or os.getenv("OPENAI_API_KEY")

        self._llm: ChatOpenAI | None = None
        if self.api_key:
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=0.0,
                timeout=25,
            )

    async def decide_action(
        self,
        *,
        high_level_command: str,
        screenshot_base64: str,
        screenshot_encoding: str,
        interactive_regions: list[InteractiveRegion],
        current_url: str,
        accessibility_tree: dict[str, Any],
    ) -> WorkerDecision:
        """Plan one next physical action from command + screenshot + UI metadata."""
        cleaned_command = high_level_command.strip()
        if not cleaned_command:
            return WorkerDecision(
                confused=True,
                confidence=0.0,
                confusion_reason="Missing high_level_command from supervisor.",
                scene_summary="No command provided.",
            )

        if not screenshot_base64:
            return WorkerDecision(
                confused=True,
                confidence=0.0,
                confusion_reason="No screenshot available for visual grounding.",
                scene_summary="No screenshot available.",
            )

        if self._llm is None:
            return WorkerDecision(
                confused=True,
                confidence=0.0,
                confusion_reason="Worker VLM credentials are not configured.",
                scene_summary="Model unavailable: missing WORKER_VLM_API_KEY or OPENAI_API_KEY.",
            )

        data_url = _decode_screenshot_data_url(screenshot_base64, screenshot_encoding)
        if not data_url:
            return WorkerDecision(
                confused=True,
                confidence=0.0,
                confusion_reason=f"Unsupported screenshot encoding: {screenshot_encoding}",
                scene_summary="Screenshot decode failed.",
            )

        structured_llm = self._llm.with_structured_output(WorkerDecision)
        messages = _build_vlm_messages(
            high_level_command=cleaned_command,
            current_url=current_url,
            data_url=data_url,
            interactive_regions=interactive_regions,
            accessibility_tree=accessibility_tree,
        )

        try:
            raw = await structured_llm.ainvoke(messages)
            decision = raw if isinstance(raw, WorkerDecision) else WorkerDecision.model_validate(raw)
        except Exception as exc:
            return WorkerDecision(
                confused=True,
                confidence=0.0,
                confusion_reason=f"Worker VLM call failed: {type(exc).__name__}: {exc}",
                scene_summary="Model invocation failed.",
            )

        if decision.action is not None:
            try:
                decision = decision.model_copy(update={"action": BrowserAction.model_validate(decision.action.model_dump())})
            except Exception as exc:
                return WorkerDecision(
                    confused=True,
                    confidence=0.0,
                    confusion_reason=f"Action schema validation failed: {type(exc).__name__}: {exc}",
                    scene_summary=decision.scene_summary or "Action schema invalid.",
                )

        if decision.action is None:
            return decision.model_copy(
                update={
                    "confused": True,
                    "confidence": min(decision.confidence, 0.4),
                    "confusion_reason": decision.confusion_reason or "Model returned no action.",
                }
            )

        if decision.confidence < 0.35:
            return decision.model_copy(
                update={
                    "confused": True,
                    "confusion_reason": decision.confusion_reason or "Worker confidence below threshold.",
                }
            )

        return decision


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


def _build_vlm_messages(
    *,
    high_level_command: str,
    current_url: str,
    data_url: str,
    interactive_regions: list[InteractiveRegion],
    accessibility_tree: dict[str, Any],
) -> list[SystemMessage | HumanMessage]:
    """Build a multimodal LangChain message payload for structured planning."""
    compact_regions = []
    for idx, region in enumerate(interactive_regions[:80]):
        compact_regions.append(
            {
                "index": idx,
                "tag": region.tag,
                "role": region.role,
                "text": region.text[:100],
                "selector": region.selector[:140],
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
            }
        )

    a11y_hints = _extract_accessibility_hints(accessibility_tree, max_items=60)

    guidance = (
        "You are a fast tactical browser-worker model. "
        "Map the high-level supervisor command into exactly one immediate BrowserAction. "
        "Use the screenshot and interactive bounding boxes for grounding. "
        "Return a structured WorkerDecision. "
        "If uncertain, set confused=true with low confidence and no action."
    )

    user_text = (
        f"Current URL: {current_url}\n"
        f"High-level command: {high_level_command}\n"
        "Interactive regions JSON:\n"
        f"{json.dumps(compact_regions, ensure_ascii=True)}\n"
        "Accessibility hints JSON:\n"
        f"{json.dumps(a11y_hints, ensure_ascii=True)}\n"
        "Action constraints:\n"
        "- click: use selector if reliable, otherwise use x/y.\n"
        "- type: must include selector and text.\n"
        "- goto: must include url.\n"
        "- scroll/wait/screenshot: provide required fields only.\n"
        "- Keep action human-like and minimal (one tactical step).\n"
        "- scene_summary should be a short plain-English description of what is visible.\n"
    )

    return [
        SystemMessage(content=guidance),
        HumanMessage(
            content=[
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        ),
    ]


def _extract_accessibility_hints(tree: dict[str, Any], max_items: int) -> list[dict[str, str]]:
    """Flatten important accessibility attributes into a bounded hint list."""
    if not tree:
        return []

    hints: list[dict[str, str]] = []
    stack: list[dict[str, Any]] = [tree]

    while stack and len(hints) < max_items:
        node = stack.pop()
        role = str(node.get("role", "")).strip()
        name = str(node.get("name", "")).strip()
        value = str(node.get("value", "")).strip()
        if role or name or value:
            hints.append(
                {
                    "role": role[:40],
                    "name": name[:120],
                    "value": value[:80],
                }
            )

        children = node.get("children") or []
        for child in reversed(children):
            if isinstance(child, dict):
                stack.append(child)

    return hints
