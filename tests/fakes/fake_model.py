"""Content-addressed scripted fake LLM for deterministic zero-API graph tests.

Adapts Cloudflare's `packages/fake-model/src/script.ts` to Python / LangChain /
LangGraph structured invocations (`ainvoke` + `with_structured_output(schema)`):

  1. Turn/step content addressing (`[<turn_id>]` marker):
     Selects the reply from the newest user/human message carrying `[<turn_id>]`
     and the count of assistant/AI messages after it — NOT a brittle request
     counter. Retrying the exact same request therefore deterministically
     receives the exact same reply.
  2. Schema + prompt-content rules (`Script.rules`):
     Supports stateless structured LangGraph calls (`StrategicPlan`,
     `WorkerAction`, `DoneVerdict`, `RealityCheck`) matched by `(schema_name,
     prompt_content)` so real live DOM state drives which scripted reply fires.
  3. Deterministic transient failure injection (`fail_times` + `error`):
     Fails the first `fail_times` calls for a given content key before returning
     the scripted reply, enabling deterministic retry & failover verification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

_TURN_MARKER_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")


@dataclass(frozen=True)
class FakeAIMessage:
    """Minimal LangChain-compatible response object exposing `.content`."""

    content: str


@dataclass(frozen=True)
class RequestLogEntry:
    """Recorded invocation audited by `ScriptedFakeLLM`."""

    seq: int
    turn_id: str | None
    step_index: int | None
    schema_name: str | None
    prompt_excerpt: str
    outcome: str  # "ok" | "error"


@dataclass(frozen=True)
class Step:
    """Single scripted response rule or turn step."""

    output: Any
    schema: str | None = None
    match: str | Callable[[str], bool] | None = None
    fail_times: int = 0
    error: Exception | None = None


@dataclass(frozen=True)
class Turn:
    """Multi-step turn keyed by `[<id>]` marker in the newest user message."""

    id: str
    steps: tuple[Step, ...]
    prompt: str = ""


@dataclass(frozen=True)
class Script:
    """Collection of content-addressed turns and stateless schema/content rules."""

    turns: tuple[Turn, ...] = ()
    rules: tuple[Step, ...] = ()


def _message_role(msg: Any) -> str:
    role = getattr(msg, "type", None) or getattr(msg, "role", None)
    if isinstance(role, str):
        return role.lower()
    if isinstance(msg, dict):
        return str(msg.get("role") or msg.get("type") or "").lower()
    return type(msg).__name__.lower()


def _message_text(msg: Any) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _render_prompt(messages: Sequence[Any]) -> str:
    return "\n".join(_message_text(m) for m in messages)


def _find_turn_step(
    messages: Sequence[Any],
    turns_by_id: dict[str, Turn],
) -> tuple[str, int, Step] | None:
    """Locate the newest user `[<turn_id>]` marker and count assistant replies after it."""
    if not turns_by_id:
        return None
    marker_idx: int | None = None
    turn_id: str | None = None

    for idx, msg in enumerate(messages):
        role = _message_role(msg)
        if role not in ("human", "user"):
            continue
        match = _TURN_MARKER_RE.search(_message_text(msg))
        if match and match.group(1) in turns_by_id:
            marker_idx = idx
            turn_id = match.group(1)

    if marker_idx is None or turn_id is None:
        return None

    assistant_count = sum(
        1
        for msg in messages[marker_idx + 1:]
        if _message_role(msg) in ("ai", "assistant")
    )
    turn = turns_by_id[turn_id]
    if assistant_count >= len(turn.steps):
        raise LookupError(
            f"ScriptedFakeLLM turn '{turn_id}' has only {len(turn.steps)} step(s), "
            f"requested step index {assistant_count}."
        )
    return turn_id, assistant_count, turn.steps[assistant_count]


def _step_matches(step: Step, prompt_text: str, schema_name: str | None) -> bool:
    if step.schema is not None and step.schema != schema_name:
        return False
    if step.match is None:
        return True
    if isinstance(step.match, str):
        return step.match in prompt_text
    return bool(step.match(prompt_text))


def _coerce_output(raw_output: Any, prompt_text: str, schema: type | None) -> Any:
    value = raw_output(prompt_text, schema) if callable(raw_output) else raw_output
    if schema is None:
        if isinstance(value, FakeAIMessage):
            return value
        if isinstance(value, str):
            return FakeAIMessage(content=value)
        return FakeAIMessage(content=json.dumps(value))
    if isinstance(value, schema):
        return value
    if isinstance(value, dict):
        return schema.model_validate(value)
    if isinstance(value, str):
        return schema.model_validate_json(value)
    return value


class _StructuredFakeRunner:
    """Adapter returned by `ScriptedFakeLLM.with_structured_output(schema)`."""

    def __init__(self, parent: ScriptedFakeLLM, schema: type) -> None:
        self._parent = parent
        self._schema = schema

    async def ainvoke(self, messages: Sequence[Any]) -> Any:
        return await self._parent._dispatch(messages, schema=self._schema)


class ScriptedFakeLLM:
    """LangChain-compatible content-addressed fake LLM with deterministic fault injection."""

    def __init__(self, script: Script) -> None:
        self.script = script
        self._turns_by_id: dict[str, Turn] = {t.id: t for t in script.turns}
        self._failure_counts: dict[str, int] = {}
        self.request_log: list[RequestLogEntry] = []

    def with_structured_output(self, schema: type) -> _StructuredFakeRunner:
        return _StructuredFakeRunner(self, schema)

    async def ainvoke(self, messages: Sequence[Any]) -> FakeAIMessage:
        return await self._dispatch(messages, schema=None)

    def _resolve(
        self,
        messages: Sequence[Any],
        prompt_text: str,
        schema_name: str | None,
    ) -> tuple[str | None, int | None, str, Step]:
        turn_hit = _find_turn_step(messages, self._turns_by_id)
        if turn_hit is not None:
            turn_id, step_idx, step = turn_hit
            return turn_id, step_idx, f"turn:{turn_id}:{step_idx}", step

        for idx, rule in enumerate(self.script.rules):
            if _step_matches(rule, prompt_text, schema_name):
                return None, idx, f"rule:{idx}:{schema_name}", rule

        excerpt = prompt_text.replace("\n", " ")[:160]
        raise LookupError(
            f"ScriptedFakeLLM found no matching turn/rule for schema={schema_name!r}, "
            f"prompt={excerpt!r}"
        )

    async def _dispatch(self, messages: Sequence[Any], schema: type | None) -> Any:
        schema_name = schema.__name__ if schema is not None else None
        prompt_text = _render_prompt(messages)
        turn_id, step_idx, fail_key, step = self._resolve(messages, prompt_text, schema_name)
        excerpt = prompt_text.replace("\n", " ")[:120]

        if step.fail_times > 0:
            used = self._failure_counts.get(fail_key, 0)
            if used < step.fail_times:
                self._failure_counts[fail_key] = used + 1
                self.request_log.append(
                    RequestLogEntry(
                        seq=len(self.request_log) + 1,
                        turn_id=turn_id,
                        step_index=step_idx,
                        schema_name=schema_name,
                        prompt_excerpt=excerpt,
                        outcome="error",
                    )
                )
                raise (step.error or TimeoutError(f"Injected transient fault on {fail_key}"))

        result = _coerce_output(step.output, prompt_text, schema)
        self.request_log.append(
            RequestLogEntry(
                seq=len(self.request_log) + 1,
                turn_id=turn_id,
                step_index=step_idx,
                schema_name=schema_name,
                prompt_excerpt=excerpt,
                outcome="ok",
            )
        )
        return result
