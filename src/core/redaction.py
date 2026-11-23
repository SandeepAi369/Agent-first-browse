"""Unified base64, screenshot, and V34 secret redaction for logs, MCP, and traces.

Combines Cloudflare's `core/base64-redaction.ts` (cycle-, depth-, and
node-bounded traversal that preserves binary types) with Agent-first-browse's
V34 secret-field redaction and raw credential pattern masking.
"""

from __future__ import annotations

import re
from typing import Any

from core.errors import ConfigValidationError

BASE64_REDACTION_THRESHOLD: int = 4096
MAX_REDACTION_DEPTH: int = 20
MAX_REDACTION_NODES: int = 10_000
SECRET_MASK: str = "[REDACTED]"

_DATA_URL_RE = re.compile(
    r"^data:([^;,]+)(?:;[^;,]*)*;base64,([A-Za-z0-9+/]*={0,2})$",
    re.IGNORECASE,
)
_BASE64_BODY_RE = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")

# Secret-bearing field names (case-insensitive normalized)
_SECRET_KEY_RE = re.compile(
    r"(?:^|[_\-.])("
    r"password|passwd|passcode|passphrase|secret|client_secret|"
    r"api_key|apikey|x_api_key|access_token|refresh_token|id_token|"
    r"auth_token|bearer|authorization|private_key|cookie|set_cookie|"
    r"otp|cvv|cvc|ssn|pin|credit_card|card_number"
    r")(?:$|[_\-.])",
    re.IGNORECASE,
)

# Benign telemetry/metric keys containing "token" that must NEVER be masked
_BENIGN_TOKEN_METRIC_RE = re.compile(
    r"(?:^|_)(?:(?:max|min|avg|total|est|prompt|completion|input|output|cached)_tokens?"
    r"|tokens?_(?:count|budget|limit))$",
    re.IGNORECASE,
)

_INLINE_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{20,}\b")),
    ("github_pat", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("nvidia_key", re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b")),
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
            r"(?:[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----)?"
        ),
    ),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE)),
    (
        "url_credentials",
        re.compile(r"(https?://)[^/\s:@]+:[^/\s@]+@"),
    ),
)

_SCREENSHOT_KEY_HINTS: frozenset[str] = frozenset(
    {"screenshot_base64", "image_base64", "base64_image"}
)


def _base64_details(
    value: str,
    min_length: int = BASE64_REDACTION_THRESHOLD,
) -> tuple[str | None, int, int] | None:
    if len(value) < min_length:
        return None
    data_match = _DATA_URL_RE.match(value)
    media_type = data_match.group(1) if data_match else None
    encoded = data_match.group(2) if data_match else value
    if not encoded or len(encoded) % 4 != 0:
        return None
    if not _BASE64_BODY_RE.match(encoded):
        return None
    padding = 2 if encoded.endswith("==") else (1 if encoded.endswith("=") else 0)
    byte_len = (len(encoded) // 4) * 3 - padding
    return media_type, len(encoded), max(0, byte_len)


def redact_base64_string(
    value: str,
    *,
    media_type: str | None = None,
    min_length: int = BASE64_REDACTION_THRESHOLD,
) -> str:
    """Replace oversized base64 or `data:...;base64,...` strings with a summary."""
    details = _base64_details(value, min_length=min_length)
    if details is None:
        return value
    detected_mime, chars, byte_len = details
    resolved_mime = media_type or detected_mime
    mime_part = f" {resolved_mime}" if resolved_mime else ""
    return (
        f"[base64{mime_part} data omitted: {chars:,} chars, "
        f"approximately {byte_len:,} bytes]"
    )


def is_secret_key(key: str) -> bool:
    """Return True if a mapping key names a secret/credential field."""
    norm = key.strip().replace("-", "_")
    if not norm:
        return False
    if _BENIGN_TOKEN_METRIC_RE.search(norm):
        return False
    return bool(_SECRET_KEY_RE.search(norm))


def redact_inline_secrets(text: str) -> str:
    """Mask raw API keys, Bearer tokens, PEM blocks, and URL credentials in text."""
    result = text
    for kind, pattern in _INLINE_SECRET_PATTERNS:
        if kind == "url_credentials":
            result = pattern.sub(r"\1[REDACTED]@", result)
        elif kind == "bearer_token":
            result = pattern.sub("Bearer [REDACTED]", result)
        else:
            result = pattern.sub(f"[REDACTED:{kind}]", result)
    return result


def _validate_positive_int(value: object, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"{name} must be an integer, got {value!r}")
    if (allow_zero and value < 0) or (not allow_zero and value <= 0):
        raise ConfigValidationError(f"{name} out of valid range: {value}")


class _RedactionWalker:
    """Stateful depth-, node-, and cycle-bounded payload redactor."""

    def __init__(
        self,
        *,
        redact_secrets: bool,
        redact_base64: bool,
        base64_threshold: int,
        max_depth: int,
        max_nodes: int,
    ) -> None:
        self.redact_secrets = redact_secrets
        self.redact_base64 = redact_base64
        self.base64_threshold = base64_threshold
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.nodes = 0
        self.ancestors: set[int] = set()

    def _redact_string(self, value: str) -> str:
        out = value
        if self.redact_base64:
            out = redact_base64_string(out, min_length=self.base64_threshold)
        if self.redact_secrets:
            out = redact_inline_secrets(out)
        return out

    def _visit_dict_entry(
        self,
        key: Any,
        val: Any,
        *,
        is_screenshot_record: bool,
        media_type: str | None,
        depth: int,
    ) -> Any:
        key_str = str(key)
        if self.redact_base64 and isinstance(val, str):
            if is_screenshot_record and key_str == "data":
                return redact_base64_string(val, media_type=media_type, min_length=0)
            if key_str.lower() in _SCREENSHOT_KEY_HINTS:
                return redact_base64_string(val, media_type=media_type, min_length=0)
        if self.redact_secrets and is_secret_key(key_str) and isinstance(val, str) and val:
            return SECRET_MASK
        return self.visit(val, depth + 1)

    def _visit_container(self, current: Any, depth: int) -> Any:
        oid = id(current)
        if oid in self.ancestors:
            return "[circular reference omitted]"
        self.ancestors.add(oid)
        try:
            if isinstance(current, list):
                return [self.visit(item, depth + 1) for item in current]
            if isinstance(current, tuple):
                return tuple(self.visit(item, depth + 1) for item in current)
            is_screenshot = (
                current.get("type") == "browser_screenshot"
                and isinstance(current.get("mediaType"), str)
            )
            media_type = current.get("mediaType") if is_screenshot else None
            return {
                k: self._visit_dict_entry(
                    k,
                    v,
                    is_screenshot_record=is_screenshot,
                    media_type=media_type,
                    depth=depth,
                )
                for k, v in current.items()
            }
        finally:
            self.ancestors.remove(oid)

    def visit(self, current: Any, depth: int) -> Any:
        if depth > self.max_depth:
            return "[nested value omitted]"
        self.nodes += 1
        if self.nodes > self.max_nodes:
            return "[remaining values omitted]"
        if isinstance(current, str):
            return self._redact_string(current)
        if isinstance(current, (bytes, bytearray, memoryview)):
            return current
        if isinstance(current, (list, tuple, dict)):
            return self._visit_container(current, depth)
        return current


def redact_payload(
    value: Any,
    *,
    redact_secrets: bool = True,
    redact_base64: bool = True,
    base64_threshold: int = BASE64_REDACTION_THRESHOLD,
    max_depth: int = MAX_REDACTION_DEPTH,
    max_nodes: int = MAX_REDACTION_NODES,
) -> Any:
    """Recursively redact secrets, screenshots, and oversized base64 payloads.

    Safe on cyclic containers, deeply nested payloads, and binary `bytes` values.
    """
    _validate_positive_int(base64_threshold, "redaction.base64_threshold", allow_zero=True)
    _validate_positive_int(max_depth, "redaction.max_depth", allow_zero=True)
    _validate_positive_int(max_nodes, "redaction.max_nodes")
    walker = _RedactionWalker(
        redact_secrets=redact_secrets,
        redact_base64=redact_base64,
        base64_threshold=base64_threshold,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )
    return walker.visit(value, 0)


__all__ = [
    "BASE64_REDACTION_THRESHOLD",
    "MAX_REDACTION_DEPTH",
    "MAX_REDACTION_NODES",
    "SECRET_MASK",
    "is_secret_key",
    "redact_base64_string",
    "redact_inline_secrets",
    "redact_payload",
]
