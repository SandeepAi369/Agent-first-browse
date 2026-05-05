from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field
from playwright.async_api import Page

from app import config
from app.browser_promoter.playwright_human_input import PlaywrightHumanInput
from app.browser_promoter.shadow_dom_piercer import TargetPoint, locate_target_point
from app.browser_promoter.state import BrowserAction, InteractiveRegion
from app.browser_promoter.worker_planner import WorkerDecision
from app.logger import get_logger
from app.observability import build_llm_config

logger = get_logger(__name__)


@dataclass(slots=True)
class ActionExecutionResult:
    details: dict[str, Any]
    vision_calls_used: int = 0


class ActionExecutionError(RuntimeError):
    def __init__(self, message: str, *, vision_calls_used: int = 0) -> None:
        super().__init__(message)
        self.vision_calls_used = vision_calls_used


class DOMSelectorDecision(BaseModel):
    selector: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""

    model_config = ConfigDict(extra="forbid")


class LowTokenDOMSelector:
    """Text-only DOM selector resolver for low-token fallback stage."""

    def __init__(self) -> None:
        self.model_name = config.DOM_SELECTOR_MODEL
        self.max_candidates = config.DOM_SELECTOR_MAX_CANDIDATES
        self._llm_clients = self._build_llms()
        self._llm_cursor = 0

    def _ordered_llms(self) -> list[tuple[int, Any]]:
        if not self._llm_clients:
            return []

        total = len(self._llm_clients)
        start = self._llm_cursor % total
        ordered: list[tuple[int, Any]] = []
        for offset in range(total):
            idx = (start + offset) % total
            ordered.append((idx, self._llm_clients[idx]))
        return ordered

    async def suggest_selector(self, *, page: Page, action: BrowserAction) -> DOMSelectorDecision | None:
        if not config.DOM_SELECTOR_ENABLED:
            return None
        if not self._llm_clients:
            return None

        candidates = await _extract_dom_candidates(page, max_candidates=self.max_candidates)
        if not candidates:
            return None

        compact_candidates = [
            {
                "index": item["index"],
                "selector": item["selector"],
                "tag": item["tag"],
                "role": item["role"],
                "text": item["text"][:80],
            }
            for item in candidates
        ]

        messages = [
            SystemMessage(
                content=(
                    "You are a low-token DOM selector resolver. "
                    "Pick exactly one selector from the provided candidates. "
                    "Return JSON only with keys: selector, confidence, reason."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "action": action.action,
                        "selector_hint": action.selector,
                        "text_hint": action.text,
                        "url": page.url,
                        "candidates": compact_candidates,
                    },
                    ensure_ascii=True,
                )
            ),
        ]

        invoke_config = build_llm_config(
            run_name="executor.low_token_dom_selector",
            tags=["executor", "stage-b", action.action],
            metadata={"candidate_count": len(compact_candidates), "model": self.model_name},
        )

        decision: DOMSelectorDecision | None = None
        total_clients = len(self._llm_clients)
        for idx, llm in self._ordered_llms():
            try:
                chain = llm.with_structured_output(DOMSelectorDecision)
                raw = await chain.ainvoke(messages, config=invoke_config)
                decision = raw if isinstance(raw, DOMSelectorDecision) else DOMSelectorDecision.model_validate(raw)
                self._llm_cursor = idx
                break
            except Exception as exc:
                try:
                    raw = await llm.ainvoke(messages, config=invoke_config)
                    text = _extract_response_text(raw)
                    decision = DOMSelectorDecision.model_validate(_extract_json_payload(text))
                    self._llm_cursor = idx
                    break
                except Exception as nested_exc:
                    logger.warning(
                        "Low-token DOM selector failed on key %d/%d: %s | %s",
                        idx + 1,
                        total_clients,
                        exc,
                        nested_exc,
                    )

        if decision is None:
            return None

        selector = decision.selector.strip()
        if not selector:
            return None
        return decision.model_copy(update={"selector": selector})

    def _build_llms(self) -> list[Any]:
        model_name = self.model_name.strip()
        wants_google = _is_google_model_name(model_name)

        google_keys = config.GOOGLE_API_KEYS
        if wants_google and google_keys:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                if model_name.startswith("models/"):
                    model_name = model_name[len("models/"):]
                return [
                    ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=google_key,
                        temperature=0.0,
                        timeout=25,
                    )
                    for google_key in google_keys
                ]
            except Exception as exc:
                logger.warning("DOM selector Google model bootstrap failed: %s", exc)

        api_keys = config.WORKER_VLM_API_KEYS or config.OPENAI_API_KEYS
        if not api_keys:
            # Fallback check for single keys
            if config.WORKER_VLM_API_KEY:
                api_keys = [config.WORKER_VLM_API_KEY]
            elif hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY:
                api_keys = [config.OPENAI_API_KEY]
            else:
                return []

        from langchain_openai import ChatOpenAI

        base_url = config.OPENAI_BASE_URL.strip() or None
        return [
            ChatOpenAI(
                model=config.DOM_SELECTOR_MODEL if not wants_google else "llama3-70b-8192",
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                timeout=20,
            )
            for api_key in api_keys
        ]


class ZeroTokenActionExecutor:
    """Module-4 executor: A=selector, B=AI text DOM, C=vision fallback.

    Uses PlaywrightHumanInput for all browser interactions.  This driver
    communicates directly with the browser engine via Playwright's native
    API — no OS-level window focus required, no PowerShell subprocess.
    """

    def __init__(self) -> None:
        self._input = PlaywrightHumanInput(
            typing_delay_min_ms=config.HUMAN_TYPING_DELAY_MIN_MS,
            typing_delay_max_ms=config.HUMAN_TYPING_DELAY_MAX_MS,
            enable_bezier_movement=True,
            enable_typo_simulation=True,
        )
        self._dom_selector = LowTokenDOMSelector()
        self._vision = WorkerVisionPlanner()

    async def execute_action(self, *, page: Page, action: BrowserAction) -> ActionExecutionResult:
        if action.action in {"manual_intervention_required", "goto", "wait", "screenshot", "scroll"}:
            details = await self._execute_non_targeted_action(page=page, action=action)
            details["fallback_stage"] = "A"
            return ActionExecutionResult(details=details)

        if action.action not in {"click", "type", "type_and_enter"}:
            raise ValueError(f"Unsupported action for zero-token executor: {action.action}")

        # Step A: selector/coordinates targeting (0 tokens)
        stage_a_target = await self._stage_a_resolve_target(page=page, action=action)
        if stage_a_target is not None:
            details = await self._apply_target_action(page=page, action=action, target=stage_a_target)
            details["fallback_stage"] = "A"
            return ActionExecutionResult(details=details)

        # Step B: AI text-based DOM selection (low tokens)
        decision = await self._dom_selector.suggest_selector(page=page, action=action)
        if decision is not None:
            stage_b_target = await locate_target_point(
                page,
                selector=decision.selector,
                text_hint=action.text,
            )
            if stage_b_target is not None:
                details = await self._apply_target_action(page=page, action=action, target=stage_b_target)
                details.update(
                    {
                        "fallback_stage": "B",
                        "dom_selector_confidence": decision.confidence,
                        "dom_selector_reason": decision.reason,
                    }
                )
                return ActionExecutionResult(details=details)

        # Step C: vision model fallback (only after A+B fail)
        if not config.VISION_FALLBACK_ENABLED:
            raise ActionExecutionError("Target resolution failed in stages A and B; vision fallback is disabled.")

        try:
            vision_action = await self._vision_fallback(page=page, action=action)
        except Exception as exc:
            raise ActionExecutionError(
                f"Vision fallback call failed: {exc}",
                vision_calls_used=1,
            ) from exc

        if vision_action is None:
            raise ActionExecutionError(
                "Vision fallback could not produce an actionable target.",
                vision_calls_used=1,
            )

        if vision_action.action in {"manual_intervention_required", "goto", "wait", "screenshot", "scroll"}:
            details = await self._execute_non_targeted_action(page=page, action=vision_action)
            details.update(
                {
                    "fallback_stage": "C",
                    "vision_action": vision_action.model_dump(mode="json"),
                }
            )
            return ActionExecutionResult(details=details, vision_calls_used=1)

        vision_target = await self._stage_a_resolve_target(page=page, action=vision_action)
        if vision_target is None:
            raise ActionExecutionError(
                "Vision fallback returned action without resolvable target.",
                vision_calls_used=1,
            )

        details = await self._apply_target_action(page=page, action=vision_action, target=vision_target)
        details.update(
            {
                "fallback_stage": "C",
                "vision_action": vision_action.model_dump(mode="json"),
            }
        )
        return ActionExecutionResult(details=details, vision_calls_used=1)

    async def _stage_a_resolve_target(self, *, page: Page, action: BrowserAction) -> TargetPoint | None:
        if action.x is not None and action.y is not None:
            return TargetPoint(
                x=action.x,
                y=action.y,
                method="direct-coordinates",
                selector_used=action.selector,
                text_hint=action.text,
            )

        if action.selector.strip():
            return await locate_target_point(
                page,
                selector=action.selector,
                text_hint=action.text,
            )

        return None

    async def _apply_target_action(
        self,
        *,
        page: Page,
        action: BrowserAction,
        target: TargetPoint,
    ) -> dict[str, Any]:
        if action.action == "click":
            result = await self._input.click(page, x=target.x, y=target.y)
            result.update({
                "target_method": target.method,
                "selector": target.selector_used,
            })
            return result

        if action.action == "type":
            await self._input.click(page, x=target.x, y=target.y)
            if getattr(action, "clear_before_type", False):
                await self._input.clear_field(page)
            typed = await self._input.type_text(page, action.text)
            return {
                "action": "type",
                "x": target.x,
                "y": target.y,
                "target_method": target.method,
                "selector": target.selector_used,
                "typed_characters": typed["typed"],
                "typing_corrections": typed["corrections"],
            }

        if action.action == "type_and_enter":
            await self._input.click(page, x=target.x, y=target.y)
            if getattr(action, "clear_before_type", False):
                await self._input.clear_field(page)
            typed = await self._input.type_text(page, action.text)
            await self._input.press_key(page, "Enter")
            return {
                "action": "type_and_enter",
                "x": target.x,
                "y": target.y,
                "target_method": target.method,
                "selector": target.selector_used,
                "typed_characters": typed["typed"],
                "typing_corrections": typed["corrections"],
                "key_pressed": "Enter",
            }

        raise ValueError(f"Targeted apply is only valid for click/type/type_and_enter, got: {action.action}")

    async def _execute_non_targeted_action(self, *, page: Page, action: BrowserAction) -> dict[str, Any]:
        if action.action == "manual_intervention_required":
            return {
                "action": "manual_intervention_required",
                "message": "Manual step flagged by planner.",
            }

        if action.action == "goto":
            response = await page.goto(
                _normalize_url(action.url),
                timeout=action.timeout_ms,
                wait_until="domcontentloaded",
            )
            return {
                "action": "goto",
                "status_code": response.status if response is not None else None,
                "final_url": page.url,
            }

        if action.action == "wait":
            await page.wait_for_timeout(action.wait_ms)
            return {"action": "wait", "wait_ms": action.wait_ms}

        if action.action == "screenshot":
            return {"action": "screenshot"}

        if action.action == "scroll":
            physics = await self._input.scroll(page, action.delta_y)
            return {"action": "scroll", "physics": physics}

        raise ValueError(f"Unsupported non-targeted action: {action.action}")

    async def _vision_fallback(self, *, page: Page, action: BrowserAction) -> BrowserAction | None:
        screenshot_bytes = await page.screenshot(type="jpeg", quality=60, full_page=False)
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("ascii")

        regions = await _extract_interactive_regions(page, max_regions=90)
        tree: dict[str, Any]
        try:
            tree = await getattr(page, "accessibility").snapshot(interesting_only=False)
            if not isinstance(tree, dict):
                tree = {}
        except Exception:
            tree = {}

        high_level = (
            f"fallback_action={action.action} | selector_hint={action.selector} | "
            f"text_hint={action.text} | url={page.url}"
        )
        decision: WorkerDecision = await self._vision.decide_action(
            high_level_command=high_level,
            screenshot_base64=screenshot_b64,
            screenshot_encoding="base64:jpeg",
            interactive_regions=regions,
            current_url=page.url,
            accessibility_tree=tree,
            llm_config=build_llm_config(
                run_name="executor.vision_fallback",
                tags=["executor", "stage-c", action.action],
                metadata={"region_count": len(regions), "url": page.url},
            ),
        )

        if decision.confused or decision.action is None:
            return None
        return decision.action


async def _extract_interactive_regions(page: Page, max_regions: int) -> list[InteractiveRegion]:
    raw_regions = await page.evaluate(
        """
(limit) => {
  const isVisible = (el) => {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
  };

  const selectorFor = (el) => {
    if (el.id) return `#${el.id}`;
    const testId = el.getAttribute('data-testid');
    if (testId) return `[data-testid="${testId}"]`;
    const name = el.getAttribute('name');
    if (name) return `${el.tagName.toLowerCase()}[name="${name}"]`;

    let idx = 1;
    let sib = el;
    while ((sib = sib.previousElementSibling) != null) {
      if (sib.tagName === el.tagName) idx += 1;
    }
    return `${el.tagName.toLowerCase()}:nth-of-type(${idx})`;
  };

  const out = [];
  const nodes = document.querySelectorAll('a,button,input,textarea,select,[role="button"],[role="link"],[tabindex]');
  for (const el of nodes) {
    if (!isVisible(el)) continue;
    const rect = el.getBoundingClientRect();
    out.push({
      tag: el.tagName.toLowerCase(),
      role: (el.getAttribute('role') || '').slice(0, 40),
      text: ((el.innerText || el.value || el.getAttribute('aria-label') || '').trim()).slice(0, 120),
      selector: selectorFor(el).slice(0, 160),
      x: Math.round(rect.left + window.scrollX),
      y: Math.round(rect.top + window.scrollY),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    });
    if (out.length >= limit) break;
  }
  return out;
}
""",
        max_regions,
    )

    regions: list[InteractiveRegion] = []
    for item in raw_regions:
        try:
            regions.append(InteractiveRegion.model_validate(item))
        except Exception:
            continue
    return regions


async def _extract_dom_candidates(page: Page, max_candidates: int) -> list[dict[str, Any]]:
    return await page.evaluate(
        """
(limit) => {
  const visible = (el) => {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
  };

  const toSelector = (el) => {
    if (el.id) return `#${el.id}`;
    const testId = el.getAttribute('data-testid');
    if (testId) return `[data-testid="${testId}"]`;
    const aria = el.getAttribute('aria-label');
    if (aria) return `${el.tagName.toLowerCase()}[aria-label="${aria.slice(0, 60)}"]`;

    let idx = 1;
    let sib = el;
    while ((sib = sib.previousElementSibling) != null) {
      if (sib.tagName === el.tagName) idx += 1;
    }
    return `${el.tagName.toLowerCase()}:nth-of-type(${idx})`;
  };

  const out = [];
  const nodes = document.querySelectorAll('a,button,input,textarea,select,[role="button"],[role="link"],[tabindex]');
  let i = 0;
  for (const el of nodes) {
    if (!visible(el)) continue;
    out.push({
      index: i,
      selector: toSelector(el),
      tag: el.tagName.toLowerCase(),
      role: (el.getAttribute('role') || '').slice(0, 40),
      text: (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().slice(0, 120),
    });
    i += 1;
    if (out.length >= limit) break;
  }
  return out;
}
""",
        max_candidates,
    )


# NOTE: _clear_with_cdp_shortcut has been removed.
# Field clearing is now handled by PhysicalInputDriver.physical_clear_field()
# which sends real OS-level Ctrl+A → Backspace via PowerShell.


def _extract_response_text(raw_response: Any) -> str:
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


def _is_google_model_name(model_name: str) -> bool:
    lowered = model_name.strip().lower()
    return "gemma" in lowered or "gemini" in lowered


def _normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        return value
    if re.match(r"^https?://", value):
        return value
    return f"https://{value}"
