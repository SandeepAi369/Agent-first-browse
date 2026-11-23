"""Structural water-filling JSON & text result truncation.

Built on Python 3.11+ stdlib so oversized browser snapshots, MCP responses, and
tool outputs stay within strict character/token budgets while remaining valid
JSON of the original shape with explicit `--- TRUNCATED ---` markers.

Water-filling strategy:
  - Small values keep their full size; the largest values absorb cuts first.
  - Lists drop elements from the tail (preserving order) with a count marker.
  - Dicts drop their largest values first and record omitted keys in a marker
    entry (disambiguated so it never shadows a real key).
  - Unserializable values (cycles, non-JSON objects) return unchanged rather
    than crashing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.errors import ConfigValidationError

CHARS_PER_TOKEN: int = 4
DEFAULT_MAX_TOKENS: int = 6000
TRUNCATION_MARKER: str = "--- TRUNCATED ---"
MIN_SLOT: int = 128


@dataclass(frozen=True)
class TruncateOptions:
    """Budget options for `truncate_text()` and `truncate_result()`."""

    max_chars: int | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_pos_int(self.max_chars, "truncate.max_chars")
        _validate_optional_pos_int(self.max_tokens, "truncate.max_tokens")


def _validate_optional_pos_int(value: object, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigValidationError(f"{field_name} must be an integer > 0, got {value!r}")


def _resolve_budget(
    max_chars: int | None,
    max_tokens: int | None,
) -> tuple[int, int]:
    _validate_optional_pos_int(max_chars, "truncate.max_chars")
    _validate_optional_pos_int(max_tokens, "truncate.max_tokens")
    tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    chars = max_chars if max_chars is not None else tokens * CHARS_PER_TOKEN
    return chars, tokens


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _is_shrinkable(value: Any) -> bool:
    return isinstance(value, (str, list, dict))


def _floor_size(value: Any, max_chars: int) -> int:
    full = _json_size(value)
    if not _is_shrinkable(value):
        return full
    return min(full, MIN_SLOT, max(2, max_chars // 2))


def _shrink_string(value: str, max_chars: int) -> str:
    suffix = f" {TRUNCATION_MARKER} {len(value):,} chars"
    keep = max(0, max_chars - len(suffix) - 2)
    out = value[:keep] + suffix
    while keep > 0 and _json_size(out) > max_chars:
        overshoot = _json_size(out) - max_chars
        keep = max(0, keep - overshoot)
        out = value[:keep] + suffix
    if _json_size(out) <= max_chars:
        return out
    return TRUNCATION_MARKER[: max(0, max_chars - 2)]


def _allocate(values: list[Any], available: int) -> list[int]:
    sizes = [_json_size(v) for v in values]
    order = sorted(range(len(values)), key=lambda idx: sizes[idx])
    allocation = [0] * len(values)
    remaining = max(0, available)
    count = len(values)
    for idx in order:
        share = remaining // count if count > 0 else 0
        alloc = min(sizes[idx], share)
        allocation[idx] = alloc
        remaining -= alloc
        count -= 1
    return allocation


def _shrink_list(items: list[Any], max_chars: int) -> list[Any]:
    def marker(dropped: int) -> str:
        return f"{TRUNCATION_MARKER} {dropped:,} more items"

    def skeleton(count: int) -> int:
        return 2 + max(0, count - 1)

    floors = [_floor_size(item, max_chars) for item in items]
    keep = len(items)
    if sum(floors) + skeleton(keep) > max_chars:
        tail_cost = _json_size(marker(len(items))) + 1
        used = skeleton(0) + tail_cost
        keep = 0
        while keep < len(items) and used + floors[keep] + 1 <= max_chars:
            used += floors[keep] + 1
            keep += 1

    kept = items[:keep]
    dropped = len(items) - keep
    tail = _json_size(marker(dropped)) + (1 if keep > 0 else 0) if dropped > 0 else 0
    allocation = _allocate(kept, max_chars - skeleton(keep) - tail)
    out = [_shrink(item, allocation[i]) for i, item in enumerate(kept)]
    if dropped > 0 and _json_size([*out, marker(dropped)]) <= max_chars:
        out.append(marker(dropped))
    return out


def _pick_marker_key(mapping: dict[str, Any]) -> str:
    marker_key = TRUNCATION_MARKER
    while marker_key in mapping:
        marker_key += " "
    return marker_key


def _select_dropped_dict_indices(
    entries: list[tuple[str, Any]],
    marker_key: str,
    max_chars: int,
) -> tuple[set[int], int]:
    key_cost = lambda k: _json_size(k) + 1  # noqa: E731
    floors = [key_cost(k) + _floor_size(v, max_chars) for k, v in entries]
    by_value_size = sorted(
        range(len(entries)),
        key=lambda idx: _json_size(entries[idx][1]),
        reverse=True,
    )

    dropped: set[int] = set()
    present = len(entries)
    floors_sum = sum(floors)
    names_len = 0

    def current_cost() -> int:
        marker_cost = 0
        if dropped:
            note_len = (
                len(f"{len(dropped):,} keys omitted: ")
                + names_len
                + 2 * (len(dropped) - 1)
            )
            note_floor = min(note_len + 2, MIN_SLOT, max(2, max_chars // 2))
            marker_cost = key_cost(marker_key) + note_floor
        count = present + (1 if dropped else 0)
        return 2 + max(0, count - 1) + floors_sum + marker_cost

    for idx in by_value_size:
        if current_cost() <= max_chars:
            break
        dropped.add(idx)
        present -= 1
        floors_sum -= floors[idx]
        names_len += _json_size(entries[idx][0]) - 2

    return dropped, current_cost()


def _shrink_dict(mapping: dict[str, Any], max_chars: int) -> dict[str, Any]:
    entries = list(mapping.items())
    marker_key = _pick_marker_key(mapping)
    dropped, est_cost = _select_dropped_dict_indices(entries, marker_key, max_chars)

    kept = [pair for i, pair in enumerate(entries) if i not in dropped]
    omitted_keys = [entries[i][0] for i in sorted(dropped)]
    note = f"{len(omitted_keys):,} keys omitted: {', '.join(omitted_keys)}"

    if est_cost > max_chars:
        room = max_chars - 2 - (_json_size(marker_key) + 1)
        return {marker_key: _shrink(note, room)} if room >= 2 else {}

    combined: list[tuple[str, Any]] = list(kept)
    if omitted_keys:
        combined.append((marker_key, note))

    fixed = (
        2
        + max(0, len(combined) - 1)
        + sum(_json_size(k) + 1 for k, _ in combined)
    )
    allocation = _allocate([v for _, v in combined], max_chars - fixed)
    return {k: _shrink(v, allocation[i]) for i, (k, v) in enumerate(combined)}


def _shrink(value: Any, max_chars: int) -> Any:
    if _json_size(value) <= max_chars:
        return value
    if isinstance(value, str):
        return _shrink_string(value, max_chars)
    if isinstance(value, list):
        return _shrink_list(value, max_chars)
    if isinstance(value, dict):
        return _shrink_dict(value, max_chars)
    return value


def truncate_text(
    text: str,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
) -> str:
    """Truncate a plain string to a character/token budget with a size marker."""
    char_limit, token_limit = _resolve_budget(max_chars, max_tokens)
    if len(text) <= char_limit:
        return text
    est_tokens = (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN
    return (
        text[:char_limit]
        + f"\n\n{TRUNCATION_MARKER}\n"
        + f"Response was ~{est_tokens:,} tokens (limit: {token_limit:,}). "
        + "Narrow the request to reduce response size."
    )


def truncate_result(
    value: Any,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Truncate a string or structured JSON value within budget, preserving shape.

    Returns `value` unchanged (same object reference) when already within budget
    or when not JSON-serializable.
    """
    char_limit, token_limit = _resolve_budget(max_chars, max_tokens)
    if isinstance(value, str):
        return truncate_text(value, max_chars=char_limit, max_tokens=token_limit)

    try:
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError):
        return value

    if len(serialized) <= char_limit:
        return value

    canonical = json.loads(serialized)
    shrunk = _shrink(canonical, char_limit)
    if _json_size(shrunk) <= char_limit:
        return shrunk
    return truncate_text(serialized, max_chars=char_limit, max_tokens=token_limit)


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_MAX_TOKENS",
    "MIN_SLOT",
    "TRUNCATION_MARKER",
    "TruncateOptions",
    "truncate_result",
    "truncate_text",
]
