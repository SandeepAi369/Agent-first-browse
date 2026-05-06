from __future__ import annotations

import base64
import json
import logging
import os
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


def _split_keys(env_var: str) -> list[str]:
    """Extract multiple API keys from a comma/semicolon-delimited env var."""
    raw = os.getenv(env_var, "")
    if not raw.strip():
        return []
    return [k.strip() for k in raw.replace(";", ",").split(",") if k.strip()]


def _build_worker_llms() -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Build worker LLM clients with multi-key support for every provider.

    Returns: (primary_clients, secondary_clients, deepseek_clients, tertiary_clients)

    Each provider supports multiple API keys via comma-separated env vars:
      NVIDIA_NIM_API_KEY       → one or more NVIDIA keys
      NVIDIA_NIM_API_KEYS      → additional NVIDIA keys (comma-separated)
      OPENAI_API_KEY           → Groq/OpenAI keys (existing multi-key support)
      DEEPSEEK_API_KEY         → one or more DeepSeek keys
      DEEPSEEK_API_KEYS        → additional DeepSeek keys (comma-separated)
      CEREBRAS_API_KEY         → one or more Cerebras keys
      CEREBRAS_API_KEYS        → additional Cerebras keys (comma-separated)
      GOOGLE_API_KEY           → Google keys (existing multi-key support)

    When multiple keys exist for a provider, each key creates a separate
    LLM client in the failover chain. This maximizes throughput because
    rate limits are per-key, so N keys = N× the daily token quota.
    """
    primary_clients = []
    secondary_clients = []
    deepseek_clients = []
    tertiary_clients = []

    # ── FAKE AGENT HEADER FACTORY ──
    # Spoof a modern browser to bypass Cloudflare Error 1010 on API endpoints
    SPOOFED_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # ── Primary: NVIDIA NIM (supports multi-key) ──
    nvidia_keys = _split_keys("NVIDIA_NIM_API_KEYS")
    single_nvidia = os.getenv("NVIDIA_NIM_API_KEY", "")
    if single_nvidia and single_nvidia not in nvidia_keys:
        nvidia_keys.insert(0, single_nvidia)

    if nvidia_keys:
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        model_name = os.getenv("NVIDIA_NIM_MODEL", "z-ai/glm-5.1")
        for key in nvidia_keys:
            primary_clients.append(
                ChatOpenAI(
                    model=model_name,
                    api_key=key,
                    base_url=base_url,
                    temperature=0.0,
                    timeout=12,
                    default_headers=SPOOFED_HEADERS
                )
            )
        log.info("NVIDIA NIM: %d key(s) loaded", len(nvidia_keys))

    # ── Secondary: GROQ (Strict Mapping) ──
    groq_keys = _split_keys("GROQ_API_KEYS")
    single_groq = os.getenv("GROQ_API_KEY", "")
    if single_groq and single_groq not in groq_keys:
        groq_keys.insert(0, single_groq)

    if groq_keys:
        from langchain_openai import ChatOpenAI
        base_url = "https://api.groq.com/openai/v1"
        model_name = "openai/gpt-oss-120b"
        for api_key in groq_keys:
            secondary_clients.append(
                ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.0,
                    timeout=25,
                    default_headers=SPOOFED_HEADERS
                )
            )
        log.info("Groq: %d key(s) loaded", len(groq_keys))

    # ── Tertiary: Cerebras (Strict Mapping) ──
    cerebras_keys = _split_keys("CEREBRAS_API_KEYS")
    single_cerebras = os.getenv("CEREBRAS_API_KEY", "")
    if single_cerebras and single_cerebras not in cerebras_keys:
        cerebras_keys.insert(0, single_cerebras)

    if cerebras_keys:
        # DISABLED: Diagnostic confirmed gpt-oss-120b returns 404 on Cerebras API.
        # Re-enable when Cerebras fixes their model registry.
        log.warning("Cerebras: SKIPPED (gpt-oss-120b returns 404 — model not available)")
        # from langchain_openai import ChatOpenAI
        # for key in cerebras_keys:
        #     tertiary_clients.append(
        #         ChatOpenAI(
        #             model="gpt-oss-120b",
        #             api_key=key,
        #             base_url="https://api.cerebras.ai/v1",
        #             temperature=0.0,
        #             timeout=25,
        #             default_headers=SPOOFED_HEADERS
        #         )
        #     )

    # ── DeepSeek V4 Pro (supports multi-key) ──
    deepseek_keys = _split_keys("DEEPSEEK_API_KEYS")
    single_deepseek = os.getenv("DEEPSEEK_API_KEY", "")
    if single_deepseek and single_deepseek not in deepseek_keys:
        deepseek_keys.insert(0, single_deepseek)

    if deepseek_keys:
        from langchain_openai import ChatOpenAI
        for key in deepseek_keys:
            deepseek_clients.append(
                ChatOpenAI(
                    model="deepseek-v4-pro",
                    api_key=key,
                    base_url="https://api.deepseek.com/v1",
                    temperature=0.0,
                    timeout=30,
                    default_headers=SPOOFED_HEADERS
                )
            )
        log.info("DeepSeek: %d key(s) loaded", len(deepseek_keys))

    # ── Tertiary: Google Gemma (already supports multi-key via config) ──
    google_keys = config.GOOGLE_API_KEYS
    if google_keys:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = "gemma-4-31b-it"
            for google_key in google_keys:
                tertiary_clients.append(
                    ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=google_key,
                        temperature=0.0,
                        timeout=30,
                    )
                )
            log.info("Google Gemma: %d key(s) loaded", len(google_keys))
        except Exception as exc:
            log.error("Google client bootstrap failed: %s", exc)

    total = len(primary_clients) + len(secondary_clients) + len(deepseek_clients) + len(tertiary_clients)
    log.info("Total LLM clients built: %d (NVIDIA=%d, Groq/Cerebras=%d, DeepSeek=%d, Gemma=%d)",
             total, len(primary_clients), len(secondary_clients),
             len(deepseek_clients), len(tertiary_clients))

    return primary_clients, secondary_clients, deepseek_clients, tertiary_clients



class StrategyDecision(BaseModel):
    strategy: str = Field(default="", description="The high-level strategy for the next step.")
    vision_requested: bool = Field(default=False, description="Set to true if visual analysis is required to find an element.")

class StrategicPlanner:
    def __init__(self) -> None:
        self._primary_clients, _, _, self._tertiary_clients = _build_worker_llms()
        self.model_name = getattr(config, 'NVIDIA_NIM_MODEL', 'glm-5.1')

    async def draft_strategy(
        self,
        high_level_command: str,
        current_url: str,
        interactive_regions: list[InteractiveRegion],
        accessibility_tree: dict[str, Any],
        action_history: list[dict[str, Any]] | None = None,
        shared_reasoning_log: list[str] = None,
        llm_config: dict[str, Any] = None,
    ) -> StrategyDecision:
        if not self._primary_clients and not self._secondary_clients:
            return StrategyDecision(strategy="No models configured", vision_requested=False)
            
        compact_regions = [{"tag": r.tag, "text": r.text[:100], "selector": r.selector} for r in interactive_regions[:50]]
        
        system = (
            "You are the Strategic Planner. Your job is to decide WHAT to do next based on the command and DOM state.\\n"
            "Analyze the command, look at the DOM regions, and output a structured strategy.\\n"
            "If the DOM regions do not contain the target element, or if you encounter a complex UI (like a captcha), set vision_requested to true."
        )
        user = f"Command: {high_level_command}\\nURL: {current_url}\\nDOM: {json.dumps(compact_regions)}\\nReasoning Log: {shared_reasoning_log}"
        
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        
        decision = None
        # Try NVIDIA First
        if self._primary_clients:
            try:
                llm = self._primary_clients[0]
                if hasattr(llm, "with_structured_output"):
                    structured_llm = llm.with_structured_output(StrategyDecision)
                    decision = await structured_llm.ainvoke(messages, config=llm_config)
            except Exception as e:
                log.warning(f"Strategy NVIDIA error: {e}. Falling back to Backup Model (Gemma).")
                
        # Fallback to Gemma (Backup)
        if decision is None and self._tertiary_clients:
            try:
                llm = self._tertiary_clients[0]
                if hasattr(llm, "with_structured_output"):
                    structured_llm = llm.with_structured_output(StrategyDecision)
                    decision = await structured_llm.ainvoke(messages, config=llm_config)
            except Exception as e:
                log.error(f"Strategy Gemma Backup error: {e}")

        if decision is None:
            decision = StrategyDecision(strategy="Fallback strategy: attempt to find interactive elements.", vision_requested=False)
            
        if decision.vision_requested and shared_reasoning_log:
            if any("[Gemma 4 Vision]" in log_entry for log_entry in shared_reasoning_log[-2:]):
                decision.vision_requested = False
                decision.strategy += " [Forcing tactical action to break loop.]"
                
        return decision

class TacticalSelector:
    def __init__(self) -> None:
        _, self._secondary_clients, self._tertiary_clients = _build_worker_llms()

    async def select_tactics(
        self,
        strategy: str,
        interactive_regions: list[InteractiveRegion],
        llm_config: dict[str, Any] = None,
    ) -> WorkerDecision:
        if not self._secondary_clients:
            return WorkerDecision(confused=True, confusion_reason="No Groq model configured.")
            
        llm = self._secondary_clients[0]
        
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
                }
            )
            
        system = (
            "You are the Tactical Selector (Groq GPT-OSS-120B). Your job is to convert a Strategy into a precise JSON BrowserAction.\\n"
            "Action must match the schema. Use exact selectors from the DOM whenever possible. Use raw x/y coordinates ONLY as a last resort."
        )
        user = f"Strategy: {strategy}\\nInteractive Regions: {json.dumps(compact_regions)}"
        
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        
        decision = None
        if self._secondary_clients:
            llm = self._secondary_clients[0]
            if hasattr(llm, "with_structured_output"):
                structured_llm = llm.with_structured_output(WorkerDecision)
                try:
                    decision = await structured_llm.ainvoke(messages, config=llm_config)
                except Exception as e:
                    log.error(f"Tactical Groq error: {e}. Falling back to Backup Model (Gemma).")
                    
        # Fallback to Gemma (Backup)
        if decision is None and self._tertiary_clients:
            try:
                llm = self._tertiary_clients[0]
                if hasattr(llm, "with_structured_output"):
                    structured_llm = llm.with_structured_output(WorkerDecision)
                    decision = await structured_llm.ainvoke(messages, config=llm_config)
            except Exception as e:
                log.error(f"Tactical Gemma Backup error: {e}")
                
        if decision is not None:
            return decision

        return WorkerDecision(confused=True, confusion_reason="Tactical parsing failed entirely.")

class VisionTool:
    def __init__(self) -> None:
        _, _, self._tertiary_clients = _build_worker_llms()

    async def analyze(
        self,
        screenshot_base64: str,
        screenshot_encoding: str,
        llm_config: dict[str, Any] = None,
    ) -> str:
        if not self._tertiary_clients:
            return "Vision analysis unavailable. No Gemma configured."
            
        data_url = _decode_screenshot_data_url(screenshot_base64, screenshot_encoding)
        if not data_url: return "Screenshot decode failed."
        
        llm = self._tertiary_clients[0]
        system = "You are a visual analysis tool (Gemma 4). Describe the UI, login walls, popups, and layout briefly to help the text models."
        
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=[{"type": "image_url", "image_url": {"url": data_url}}])
        ]
        
        try:
            res = await llm.ainvoke(messages, config=llm_config)
            if hasattr(res, "content"):
                return res.content
            return str(res)
        except Exception as e:
            return f"Vision API error: {e}"


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
