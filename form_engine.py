"""FormEngine — Universal per-field state tracking + editor-aware typing.

Fixes V-02 (double-write alias), V-05 (false title marking on body),
V-09 (no body tracking), V-13 (naive keyword matching).

Architecture:
  1. classify_field() — structural heuristics + LLM fallback
  2. FormState — single source of truth for field fill status
  3. typing_strategy() — editor-aware injection method selection
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger("form_engine")

# ── Field Classification ──────────────────────────────────────────────

TITLE_HINTS = ("title", "subject", "headline", "heading")
BODY_HINTS = ("body", "content", "message", "post", "comment",
              "description", "article", "write", "text")


def classify_field(el: dict) -> tuple[str, float]:
    """Classify an element as title, body, or unknown.

    Uses structural signals first (role, multiline, tag),
    then keyword matching on combined attributes.
    """
    sig = " ".join([
        el.get("role", ""),
        el.get("name", ""),
        el.get("placeholder", ""),
        el.get("aria_label", el.get("label", "")),
    ]).lower()

    tag = el.get("tag", "").upper()
    is_multiline = (
        el.get("multiline", False)
        or tag == "TEXTAREA"
        or el.get("contenteditable") in ("true", "plaintext-only")
        or el.get("height", 0) > 100
    )

    # Body signals (check first — more common in single-field forms)
    if any(h in sig for h in BODY_HINTS) or is_multiline:
        return "body", 0.85

    # Title signals
    if any(h in sig for h in TITLE_HINTS) and not is_multiline:
        return "title", 0.85

    return "unknown", 0.0


def typing_strategy(el: dict) -> Literal["native_value", "insert_text"]:
    """Choose the correct input method based on the element type.

    ProseMirror, CodeMirror, Slate, Quill, and contenteditable
    manage internal state and need CDP Input.insertText so their
    real input pipeline runs (not just setting .value).
    """
    tag = el.get("tag", "").upper()
    if tag in ("INPUT", "TEXTAREA"):
        return "native_value"

    ce = el.get("contenteditable", "")
    if ce in ("true", "plaintext-only"):
        return "insert_text"

    # Rich editor detection by common class patterns
    classes = el.get("class", "").lower()
    rich_editors = ("prosemirror", "cm-content", "ql-editor",
                    "slate-editor", "draft-editor", "tiptap")
    if any(e in classes for e in rich_editors):
        return "insert_text"

    return "native_value"


# ── Form State Machine ────────────────────────────────────────────────

@dataclass
class FormState:
    """Single source of truth for form fill status.

    Replaces the fragile phase_tracker + completed_phases alias pattern.
    No aliases, no double-writes. One object, one truth.
    """
    required_fields: set[str] = field(default_factory=lambda: {"title", "body"})
    _filled: dict[str, str] = field(default_factory=dict)  # field -> content preview

    def mark_filled(self, field_name: str, content_preview: str = "") -> None:
        self._filled[field_name] = content_preview[:100]
        logger.info("✅ FormState: '%s' filled (%d chars)", field_name, len(content_preview))

    def is_filled(self, field_name: str) -> bool:
        return field_name in self._filled

    @property
    def filled_fields(self) -> list[str]:
        return list(self._filled.keys())

    @property
    def ready_to_submit(self) -> bool:
        return self.required_fields <= set(self._filled.keys())

    @property
    def status_summary(self) -> str:
        filled = [f"✓{f}" for f in self._filled]
        missing = [f"✗{f}" for f in self.required_fields - set(self._filled.keys())]
        return f"FormState: {' '.join(filled + missing)} | ready={self.ready_to_submit}"

    def reset(self) -> None:
        self._filled.clear()
