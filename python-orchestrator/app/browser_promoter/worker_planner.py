from __future__ import annotations

import base64
import json
import logging
import zlib
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.call_pacing import AsyncGapLimiter
from app import config
from app.browser_promoter.state import BrowserAction, InteractiveRegion


log = logging.getLogger("app.browser_promoter.worker_planner")
_WORKER_LLM_GAP = AsyncGapLimiter(config.WORKER_LLM_MIN_GAP_SECONDS)


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


def _is_google_model_name(model_name: str) -> bool:
    lowered = model_name.strip().lower()
    return "gemini" in lowered or "gemma" in lowered


def _build_worker_llms() -> tuple[list[Any], str]:
    """Build worker LLM clients with automatic API-key failover.

    Priority:
    1. Google key(s) + Gemma/Gemini model → ChatGoogleGenerativeAI
    2. WORKER_VLM_API_KEY(S) / OPENAI_API_KEY(S) → ChatOpenAI-compatible provider
    """
    model_name = config.WORKER_VLM_MODEL.strip()
    google_keys = config.GOOGLE_API_KEYS
    openai_keys = config.WORKER_VLM_API_KEYS or config.OPENAI_API_KEYS
    wants_google_model = _is_google_model_name(model_name)

    # ── Google path (Gemma/Gemini models) ──
    if google_keys and wants_google_model:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Strip "models/" prefix if present — the SDK adds it
            if model_name.startswith("models/"):
                model_name = model_name[len("models/"):]
            return (
                [
                    ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=google_key,
                        temperature=0.0,
                        timeout=30,
                    )
                    for google_key in google_keys
                ],
                "",
            )
        except Exception as exc:
            error = (
                "Google client bootstrap failed for Gemma/Gemini model. "
                f"Install/verify langchain-google-genai and model name. ({type(exc).__name__}: {exc})"
            )
            log.error(error)
            return [], error

    if wants_google_model and not google_keys:
        error = (
            "WORKER_VLM_MODEL requests Gemini/Gemma but GOOGLE_API_KEY (or GEMINI_API_KEY) is missing. "
            "Set GOOGLE_API_KEY or switch WORKER_VLM_MODEL to an OpenAI-compatible model."
        )
        return [], error

    # ── OpenAI-compatible path with key rotation (OpenAI/Groq/etc.) ──
    if openai_keys:
        from langchain_openai import ChatOpenAI
        resolved_model_name = model_name

        if wants_google_model:
            # Prevent provider mismatch loops (e.g., gemini model name on OpenAI/Groq endpoint).
            resolved_model_name = config.WORKER_VLM_FALLBACK_MODEL.strip() if hasattr(config, 'WORKER_VLM_FALLBACK_MODEL') and config.WORKER_VLM_FALLBACK_MODEL.strip() else "openai/gpt-oss-120b"
            log.warning(
                "Worker model '%s' appears Gemini-specific but Gemini provider is unavailable. "
                "Falling back to '%s'.",
                model_name,
                resolved_model_name,
            )

        base_url = config.OPENAI_BASE_URL.strip() if config.OPENAI_BASE_URL else None
        clients = [
            ChatOpenAI(
                model=resolved_model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                timeout=25,
            )
            for api_key in openai_keys
        ]

        return (
            clients,
            "",
        )

    return [], "Worker model client is not configured. No supported API key variables were found."


class WorkerVisionPlanner:
    """LangChain-based planner that maps high-level intent to physical browser action."""

    def __init__(self) -> None:
        self.model_name = config.WORKER_VLM_MODEL
        self._llm_clients, self._llm_bootstrap_error = _build_worker_llms()
        self._llm_cursor = 0

    def _ordered_llm_clients(self) -> list[tuple[int, Any]]:
        if not self._llm_clients:
            return []

        total = len(self._llm_clients)
        start = self._llm_cursor % total
        ordered: list[tuple[int, Any]] = []
        for offset in range(total):
            idx = (start + offset) % total
            ordered.append((idx, self._llm_clients[idx]))
        return ordered

    async def decide_action(
        self,
        *,
        high_level_command: str,
        screenshot_base64: str,
        screenshot_encoding: str,
        interactive_regions: list[InteractiveRegion],
        current_url: str,
        accessibility_tree: dict[str, Any],
        llm_config: dict[str, Any] | None = None,
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
            import re
            url_match = re.search(r"https?://[^\s\|,\)]+", cleaned_command)
            target_url = url_match.group(0) if url_match else "https://www.google.com"
            return WorkerDecision(
                action=BrowserAction(action="goto", url=target_url),
                confidence=1.0,
                confused=False,
                confusion_reason="",
                scene_summary="No screenshot available. Bootstrapping browser with initial navigation.",
            )

        if not self._llm_clients:
            return WorkerDecision(
                confused=True,
                confidence=0.0,
                confusion_reason=self._llm_bootstrap_error or "Worker VLM credentials are not configured.",
                scene_summary="Model unavailable due to provider/key configuration.",
            )

        data_url = _decode_screenshot_data_url(screenshot_base64, screenshot_encoding)
        if not data_url:
            return WorkerDecision(
                confused=True,
                confidence=0.0,
                confusion_reason=f"Unsupported screenshot encoding: {screenshot_encoding}",
                scene_summary="Screenshot decode failed.",
            )

        messages = _build_vlm_messages(
            high_level_command=cleaned_command,
            current_url=current_url,
            data_url=data_url,
            interactive_regions=interactive_regions,
            accessibility_tree=accessibility_tree,
        )

        wait_seconds = await _WORKER_LLM_GAP.wait_turn()
        if wait_seconds > 0.0:
            log.info("Worker LLM pacing wait applied: %.2fs", wait_seconds)

        decision: WorkerDecision | None = None
        last_exc: Exception | None = None
        total_clients = len(self._llm_clients)

        for idx, llm in self._ordered_llm_clients():
            try:
                # Try structured output first (works with OpenAI/Groq)
                try:
                    structured_llm = llm.with_structured_output(WorkerDecision)
                    raw = await structured_llm.ainvoke(messages, config=llm_config)
                    decision = raw if isinstance(raw, WorkerDecision) else WorkerDecision.model_validate(raw)
                except (NotImplementedError, AttributeError, TypeError):
                    # Fallback for providers that don't expose strict structured output.
                    raw_response = await llm.ainvoke(messages, config=llm_config)
                    text = _extract_response_text(raw_response)
                    parsed = _extract_json_payload(text)
                    decision = WorkerDecision.model_validate(parsed)

                # Stick to whichever key most recently succeeded so mid-run calls stay healthy.
                self._llm_cursor = idx
                break
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "Worker VLM call failed on key %d/%d (%s). Trying next key.",
                    idx + 1,
                    total_clients,
                    type(exc).__name__,
                )

        if decision is None:
            reason = (
                f"Worker VLM call failed across all configured API keys: {type(last_exc).__name__}: {last_exc}"
                if last_exc is not None
                else "Worker VLM call failed across all configured API keys."
            )
            return WorkerDecision(
                confused=True,
                confidence=0.0,
                confusion_reason=reason,
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

        if _looks_like_login_wall(decision.scene_summary, decision.confusion_reason, interactive_regions):
            return decision.model_copy(
                update={
                    "action": BrowserAction(action="manual_intervention_required"),
                    "confidence": max(decision.confidence, 0.7),
                    "confused": False,
                    "confusion_reason": "",
                }
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
        "If uncertain, set confused=true with low confidence and no action.\n\n"
        "You MUST respond with ONLY valid JSON matching this exact schema:\n"
        "{\n"
        '  "action": {"action": "goto|click|type|scroll|wait|screenshot", "url": "", "selector": "", "text": "", "x": null, "y": null, "delta_y": 700} or null,\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "confused": true/false,\n'
        '  "confusion_reason": "",\n'
        '  "scene_summary": "short description of what is visible"\n'
        "}\n"
        "Action types: goto (needs url), click (needs selector or x/y), type (needs selector+text), scroll (needs delta_y), wait, screenshot.\n"
        "Do NOT include any text outside the JSON object."
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
        "- manual_intervention_required: use this for login walls, Gmail auth, captcha, or MFA screens.\n"
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


def _looks_like_login_wall(scene_summary: str, confusion_reason: str, interactive_regions: list[InteractiveRegion]) -> bool:
    """Detect login/captcha wall cues that require manual human intervention."""
    haystack = f"{scene_summary} {confusion_reason}".lower()
    
    # Fast fail on known hard blocks
    hard_blocks = ("captcha", "cloudflare", "hcaptcha", "human verification")
    if any(kw in haystack for kw in hard_blocks):
        return True
        
    # Verify true login intent by searching UI elements for password/OTP fields
    for region in interactive_regions:
        r_text = region.text.lower()
        if "password" in r_text or "otp" in r_text or "verification code" in r_text:
            return True
            
    return False
