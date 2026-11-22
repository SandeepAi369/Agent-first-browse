"""
a11y_parser.py — V12.0 Accessibility-First Extraction Engine
═════════════════════════════════════════════════════════════
Replaces the heavy V11 God-Mode JS DOM parser as the primary
extraction layer. Assimilated patterns from:
  • browser-use   — CDP Accessibility.getFullAXTree + element indexing
  • Playwright     — Internal snapshotForAI with ref tags
  • OpenClaw       — RefID-based action dispatch

Architecture:
  Tier-1: Playwright snapshotForAI (internal, ~20ms, ref-tagged)
  Tier-2: CDP Accessibility.getFullAXTree (robust, backend node IDs)
  Tier-3: page.accessibility.snapshot() (stable public API)
  Tier-4: V11 dom_parser.extract() (legacy fallback)

Output: Ultra-compact RefID list — ~200 tokens for a full page
  Example: [e1: textbox "Search"] [e2: button "Go"] [e3: link "Home"]

Token savings vs V11: 85-90% reduction
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from typing import Any

log = logging.getLogger("a11y_parser")

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "checkbox", "radio",
    "combobox", "listbox", "menuitem", "menuitemcheckbox",
    "menuitemradio", "option", "searchbox", "slider",
    "spinbutton", "switch", "tab", "treeitem",
})

# Map HTML tags → ARIA roles for normalization
_TAG_TO_ROLE = {
    "a": "link", "button": "button", "input": "textbox",
    "textarea": "textbox", "select": "combobox",
    "summary": "button", "details": "group",
}

# Compatible role groups for fuzzy matching
_COMPAT_GROUPS = [
    {"textbox", "searchbox", "input", "textarea"},
    {"button", "menuitem", "summary", "tab", "menuitemcheckbox", "menuitemradio"},
    {"link", "a"},
    {"combobox", "select", "listbox"},
    {"checkbox", "switch"},
]

_DEFAULT_TIMEOUT = 5.0
_SECRET_FIELD_RE = re.compile(
    r"\b(password|passwd|passcode|pin|otp|one[-\s]?time|cvv|cvc|csc|card[-\s]?number|secret)\b",
    re.IGNORECASE,
)
_QUOTED_NAME_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"\s*(.*)$')
_UNESCAPE_RE = re.compile(r'\\(["\\])')


# ═══════════════════════════════════════════════════════════════════════════════
#  Validation & Sanitization Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize_timeout(timeout: Any, default: float = _DEFAULT_TIMEOUT) -> float:
    """Return a positive, finite timeout in seconds."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return default
    val = float(timeout)
    if not math.isfinite(val) or val <= 0:
        return default
    return val


def _normalize_text(value: Any, max_len: int) -> str:
    """Safely coerce a text field to a whitespace-normalized string."""
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    text = value if isinstance(value, str) else str(value)
    return re.sub(r"\s+", " ", text).strip()[:max_len]


def _extract_ax_str(obj: Any, max_len: int) -> str:
    """Extract a string from a CDP AXValue dict ({type, value}) or raw scalar."""
    raw = obj.get("value") if isinstance(obj, dict) else obj
    return _normalize_text(raw, max_len)


def _is_secret_field(
    role: str,
    name: str,
    description: str = "",
    properties: dict[str, Any] | None = None,
) -> bool:
    """Detect password/OTP/card fields whose values must never leak into prompts or logs."""
    if properties and properties.get("protected"):
        return True
    if role == "password":
        return True
    return bool(_SECRET_FIELD_RE.search(f"{name} {description}"))


def _is_finite_number(val: Any) -> bool:
    """True for finite int/float values (excluding bool)."""
    return not isinstance(val, bool) and isinstance(val, (int, float)) and math.isfinite(val)


def _valid_bbox_center(bbox: Any) -> tuple[float, float] | None:
    """Validate bbox center coordinates: finite and non-negative (0 is valid)."""
    if not isinstance(bbox, dict):
        return None
    cx = bbox.get("cx")
    cy = bbox.get("cy")
    if not (_is_finite_number(cx) and _is_finite_number(cy)):
        return None
    cx_f, cy_f = float(cx), float(cy)
    if cx_f < 0 or cy_f < 0:
        return None
    return cx_f, cy_f


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

class RefElement:
    """A single interactive element with its RefID."""
    __slots__ = (
        "ref", "role", "name", "description", "bbox",
        "selector_hint", "backend_node_id", "properties",
    )

    def __init__(
        self, ref: str, role: str, name: str = "",
        description: str = "", bbox: dict | None = None,
        selector_hint: str = "", backend_node_id: int | None = None,
        properties: dict | None = None,
    ):
        self.ref = ref
        self.role = role
        self.name = name
        self.description = description
        self.bbox = bbox
        self.selector_hint = selector_hint
        self.backend_node_id = backend_node_id
        self.properties = properties or {}

    def to_compact(self) -> str:
        """Ultra-compact format: [e1: button "Submit"]"""
        name_part = f' "{self.name}"' if self.name else ""
        flags = ""
        if self.properties.get("checked"):
            flags += " ✓"
        if self.properties.get("disabled"):
            flags += " [disabled]"
        if self.properties.get("required"):
            flags += " *"
        if self.properties.get("secret"):
            flags += " [value hidden]"
        elif self.properties.get("value"):
            val = str(self.properties["value"])[:30]
            flags += f' val="{val}"'
        return f"[{self.ref}: {self.role}{name_part}{flags}]"

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "role": self.role,
            "name": self.name,
            "bbox": self.bbox,
            "backend_node_id": self.backend_node_id,
            "properties": self.properties,
        }


class A11ySnapshot:
    """Container for an accessibility extraction result."""

    def __init__(
        self, elements: list[RefElement],
        raw_text: str = "", page_title: str = "", url: str = "",
    ):
        self.elements = elements
        self.raw_text = raw_text
        self.page_title = page_title
        self.url = url
        self._ref_map: dict[str, RefElement] = {e.ref: e for e in elements}

    @property
    def refid_text(self) -> str:
        """Ultra-compact element list for LLM context (~15 chars/element)."""
        return " ".join(e.to_compact() for e in self.elements)

    @property
    def semantic_hash(self) -> int:
        """Deterministic hash for change detection (used by Critic V12)."""
        if not self.elements:
            return 0
        content = "|".join(
            f"{e.role}:{e.name}:{sorted((str(k), str(v)) for k, v in e.properties.items())}"
            for e in self.elements
        )
        digest = hashlib.sha256(content.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    @property
    def element_count(self) -> int:
        return len(self.elements)

    def resolve(self, ref: str) -> RefElement | None:
        """Resolve a RefID to its element data."""
        return self._ref_map.get(ref)

    def to_dict(self) -> dict:
        return {
            "elements": [e.to_dict() for e in self.elements],
            "refid_text": self.refid_text,
            "semantic_hash": self.semantic_hash,
            "element_count": self.element_count,
            "page_title": self.page_title,
            "url": self.url,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier-1: Playwright Internal snapshotForAI
# ═══════════════════════════════════════════════════════════════════════════════

async def _extract_via_snapshot_for_ai(page, timeout_ms: int = 10000) -> A11ySnapshot | None:
    """Primary extraction via Playwright's internal snapshotForAI API.

    Returns an ARIA snapshot with [ref=eN] tags on interactive elements.
    This is the most token-efficient method (~20ms, pre-filtered).
    """
    try:
        raw_sec = timeout_ms / 1000.0 if _is_finite_number(timeout_ms) else None
        timeout_sec = _sanitize_timeout(raw_sec, 10.0)
        safe_timeout_ms = max(100, int(timeout_sec * 1000))

        # Try Page channel first, then Frame channel
        result = None
        candidates = [page]
        main_frame = getattr(page, "main_frame", None)
        if main_frame is not None:
            candidates.append(main_frame)

        for channel_obj in candidates:
            try:
                impl = getattr(channel_obj, "_impl_obj", None)
                if impl is None:
                    continue
                ch = getattr(impl, "_channel", None)
                if ch is None:
                    continue
                result = await asyncio.wait_for(
                    ch.send("snapshotForAI", {"timeout": safe_timeout_ms}),
                    timeout=timeout_sec,
                )
                if result:
                    break
            except Exception:
                continue

        if result is None:
            return None

        # Handle dict or string result
        if isinstance(result, dict):
            snapshot_text = result.get("text", result.get("snapshot", ""))
            if not snapshot_text:
                snapshot_text = json.dumps(result, indent=2)
        else:
            snapshot_text = str(result)

        if not snapshot_text or len(snapshot_text) < 5:
            return None

        elements = _parse_aria_snapshot_text(snapshot_text)

        title = ""
        try:
            title = await page.title()
        except Exception:
            pass

        url = ""
        try:
            url = page.url
        except Exception:
            pass

        return A11ySnapshot(
            elements=elements,
            raw_text=snapshot_text,
            page_title=title,
            url=url,
        )
    except Exception as e:
        log.debug("Tier-1 snapshotForAI failed: %s", e)
        return None


_BOOL_TAG_MAP: dict[str, tuple[str, Any]] = {
    "checked": ("checked", True),
    "checked=true": ("checked", True),
    "checked=mixed": ("checked", "mixed"),
    "disabled": ("disabled", True),
    "disabled=true": ("disabled", True),
    "required": ("required", True),
    "required=true": ("required", True),
    "expanded": ("expanded", True),
    "expanded=true": ("expanded", True),
    "selected": ("selected", True),
    "selected=true": ("selected", True),
    "pressed": ("pressed", True),
    "pressed=true": ("pressed", True),
}


def _parse_bracket_tags(tag_text: str, role: str) -> tuple[str, dict[str, Any]]:
    """Parse ARIA metadata tags (outside any quoted element name).

    Returns (ref_id, properties).
    """
    ref = ""
    properties: dict[str, Any] = {}

    for raw_tag in re.findall(r"\[([^\]]+)\]", tag_text):
        tag = raw_tag.strip()
        if not tag:
            continue
        low = tag.lower()

        if low.startswith("ref="):
            candidate = tag.split("=", 1)[1].strip()
            if candidate and not ref:
                ref = candidate
        elif low in _BOOL_TAG_MAP:
            k, v = _BOOL_TAG_MAP[low]
            properties[k] = v
        elif low.startswith("level=") and role == "heading":
            lvl = tag.split("=", 1)[1].strip()
            if lvl.isdigit():
                properties["level"] = int(lvl)

    return ref, properties


def _parse_aria_snapshot_text(text: str) -> list[RefElement]:
    """Parse ARIA snapshot text to extract interactive elements with [ref=...] tags.

    Example input lines:
      - link "Home" [ref=e1]
      - textbox "Email" [ref=e3]
      - button "Submit" [ref=e5]
      - checkbox "Remember me" [ref=e6] [checked]
      - button "Save [Draft]" [ref=e7] [disabled]
    """
    if not isinstance(text, str) or not text.strip():
        return []

    elements: list[RefElement] = []
    seen_refs: set[str] = set()

    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped or "[ref=" not in line_stripped:
            continue

        # Strip leading YAML list bullet ("- " or "* ") without touching hyphens in roles
        body = re.sub(r"^[-*]\s+", "", line_stripped)
        role_match = re.match(r"^([A-Za-z][\w-]*)(?:\s+(.*))?$", body)
        if not role_match:
            continue

        role = role_match.group(1).lower()
        rest = (role_match.group(2) or "").strip()

        # Parse quoted name first so brackets/keywords inside quotes are preserved
        name = ""
        quoted = _QUOTED_NAME_RE.match(rest)
        if quoted:
            raw_name = _UNESCAPE_RE.sub(r"\1", quoted.group(1))
            name = _normalize_text(raw_name, 80)
            tag_region = quoted.group(2)
        else:
            tag_region = rest
            before_tags = rest.split("[", 1)[0].strip(" :\t")
            if before_tags and len(before_tags) < 80:
                name = _normalize_text(before_tags, 80)

        ref, properties = _parse_bracket_tags(tag_region, role)
        if not ref or ref in seen_refs:
            continue
        seen_refs.add(ref)

        elements.append(RefElement(
            ref=ref,
            role=role,
            name=name,
            properties=properties,
        ))

    return elements


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier-2: CDP Accessibility.getFullAXTree
# ═══════════════════════════════════════════════════════════════════════════════

async def _extract_via_cdp_ax_tree(page, timeout: float = _DEFAULT_TIMEOUT) -> A11ySnapshot | None:
    """Robust extraction via CDP Accessibility.getFullAXTree.

    Provides backend node IDs for precise element targeting.
    Assimilated from browser-use's DomService._get_ax_tree_for_all_frames().
    """
    timeout = _sanitize_timeout(timeout)
    try:
        client = await page.context.new_cdp_session(page)
    except Exception as e:
        log.debug("CDP session creation failed: %s", e)
        return None

    try:
        # Enable accessibility domain
        await client.send("Accessibility.enable")

        # Get full accessibility tree
        ax_result = await asyncio.wait_for(
            client.send("Accessibility.getFullAXTree"),
            timeout=timeout,
        )

        if not isinstance(ax_result, dict):
            return None

        nodes = ax_result.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            return None

        elements: list[RefElement] = []
        ref_counter = 0

        for node in nodes:
            if not isinstance(node, dict) or node.get("ignored", False):
                continue

            role = _extract_ax_str(node.get("role"), 40).lower()
            if role not in INTERACTIVE_ROLES:
                continue

            name = _extract_ax_str(node.get("name"), 80)
            desc = _extract_ax_str(node.get("description"), 60)
            backend_node_id = node.get("backendDOMNodeId")

            # Extract properties (checked, disabled, etc.)
            properties: dict[str, Any] = {}
            raw_value = None
            raw_props = node.get("properties")
            if isinstance(raw_props, list):
                for prop in raw_props:
                    if not isinstance(prop, dict):
                        continue
                    prop_name = prop.get("name", "")
                    prop_val = prop.get("value", {})
                    val = prop_val.get("value") if isinstance(prop_val, dict) else prop_val
                    if prop_name in ("checked", "disabled", "required", "expanded", "selected", "protected"):
                        properties[prop_name] = val
                    elif prop_name == "value" and val is not None and str(val) != "":
                        raw_value = str(val)[:50]

            if raw_value is not None:
                if _is_secret_field(role, name, desc, properties):
                    properties["secret"] = True
                else:
                    properties["value"] = raw_value

            ref_counter += 1
            ref = f"e{ref_counter}"

            elements.append(RefElement(
                ref=ref,
                role=role,
                name=name,
                description=desc,
                backend_node_id=backend_node_id,
                properties=properties,
            ))

        title = ""
        try:
            title = await page.title()
        except Exception:
            pass

        url = ""
        try:
            url = page.url
        except Exception:
            pass

        return A11ySnapshot(
            elements=elements,
            raw_text="",  # No raw text from CDP
            page_title=title,
            url=url,
        )
    except Exception as e:
        log.debug("Tier-2 CDP AXTree failed: %s", e)
        return None
    finally:
        try:
            await client.detach()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier-3: Standard page.accessibility.snapshot()
# ═══════════════════════════════════════════════════════════════════════════════

async def _extract_via_a11y_snapshot(page, timeout: float = _DEFAULT_TIMEOUT) -> A11ySnapshot | None:
    """Fallback extraction via the stable public accessibility API."""
    timeout = _sanitize_timeout(timeout)
    try:
        snapshot = await asyncio.wait_for(
            page.accessibility.snapshot(),
            timeout=timeout,
        )
        if not isinstance(snapshot, dict) or not snapshot:
            return None

        elements: list[RefElement] = []
        ref_counter = [0]
        visited: set[int] = set()

        def walk(node: Any, depth: int = 0):
            if not isinstance(node, dict) or depth > 15:
                return
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)

            role = _normalize_text(node.get("role"), 40).lower()
            name = _normalize_text(node.get("name"), 80)
            desc = _normalize_text(node.get("description"), 60)

            if role in INTERACTIVE_ROLES:
                ref_counter[0] += 1
                ref = f"e{ref_counter[0]}"
                properties: dict[str, Any] = {}
                if node.get("checked") is not None:
                    properties["checked"] = node["checked"]
                if node.get("disabled"):
                    properties["disabled"] = True
                if node.get("required"):
                    properties["required"] = True
                if node.get("expanded") is not None:
                    properties["expanded"] = node["expanded"]
                if node.get("selected"):
                    properties["selected"] = True
                if node.get("protected"):
                    properties["protected"] = True

                raw_val = node.get("value")
                if raw_val is not None and str(raw_val) != "":
                    if _is_secret_field(role, name, desc, properties):
                        properties["secret"] = True
                    else:
                        properties["value"] = str(raw_val)[:50]

                elements.append(RefElement(
                    ref=ref,
                    role=role,
                    name=name,
                    description=desc,
                    properties=properties,
                ))

            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    walk(child, depth + 1)

        walk(snapshot)

        title = ""
        try:
            title = await page.title()
        except Exception:
            pass

        url = ""
        try:
            url = page.url
        except Exception:
            pass

        return A11ySnapshot(
            elements=elements,
            raw_text="",
            page_title=title,
            url=url,
        )
    except Exception as e:
        log.debug("Tier-3 a11y.snapshot() failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Bounding Box Enrichment (shared across all tiers)
# ═══════════════════════════════════════════════════════════════════════════════

_BBOX_JS = r"""
() => {
    const sels = [
        '[role]','a[href]','button','input','textarea','select',
        '[tabindex]:not([tabindex="-1"])','[contenteditable="true"]',
        'summary','label[for]'
    ];
    const seen = new WeakSet();
    const out = [];
    for (const s of sels) {
        try {
            for (const el of document.querySelectorAll(s)) {
                if (seen.has(el)) continue;
                seen.add(el);
                const r = el.getBoundingClientRect();
                if (r.width < 2 || r.height < 2) continue;
                const st = window.getComputedStyle(el);
                if (st.display === 'none' || st.visibility === 'hidden') continue;
                if (parseFloat(st.opacity) <= 0) continue;
                const role = el.getAttribute('role')
                    || el.tagName.toLowerCase();
                const name = el.getAttribute('aria-label')
                    || el.getAttribute('title')
                    || el.innerText?.slice(0, 80)?.trim()
                    || el.getAttribute('placeholder')
                    || el.getAttribute('name')
                    || '';
                out.push({
                    role: role.toLowerCase(),
                    name: name,
                    x: Math.round(r.x),
                    y: Math.round(r.y),
                    w: Math.round(r.width),
                    h: Math.round(r.height),
                    cx: Math.round(r.x + r.width / 2),
                    cy: Math.round(r.y + r.height / 2),
                });
            }
        } catch(_) {}
    }
    return out;
}
"""


def _extract_dom_bbox(dom_el: Any) -> dict[str, int | float] | None:
    """Validate and normalize a raw DOM bounding-box record from _BBOX_JS."""
    if not isinstance(dom_el, dict):
        return None
    keys = ("x", "y", "w", "h", "cx", "cy")
    if not all(_is_finite_number(dom_el.get(k)) for k in keys):
        return None
    if dom_el["w"] <= 0 or dom_el["h"] <= 0:
        return None
    return {
        "x": dom_el["x"],
        "y": dom_el["y"],
        "width": dom_el["w"],
        "height": dom_el["h"],
        "cx": dom_el["cx"],
        "cy": dom_el["cy"],
    }


async def _enrich_with_bounding_boxes(page, snapshot: A11ySnapshot) -> None:
    """Match A11y elements to DOM elements and attach bounding boxes."""
    if not snapshot.elements:
        return
    try:
        dom_elements = await asyncio.wait_for(
            page.evaluate(_BBOX_JS),
            timeout=3.0,
        )
        if not isinstance(dom_elements, list) or not dom_elements:
            return

        # Greedy best-match: for each a11y element, find the best valid DOM match
        available = [d for d in dom_elements if _extract_dom_bbox(d) is not None]
        for el in snapshot.elements:
            best_idx = -1
            best_score = 0
            for i, dom in enumerate(available):
                score = _match_score(el, dom)
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx >= 0 and best_score >= 2:
                matched = available.pop(best_idx)
                el.bbox = _extract_dom_bbox(matched)
    except Exception as e:
        log.debug("Bounding box enrichment failed: %s", e)


def _match_score(a11y_el: RefElement, dom_el: dict) -> int:
    """Score how well an A11y element matches a DOM element."""
    if not isinstance(dom_el, dict):
        return 0

    score = 0
    raw_dom_role = _normalize_text(dom_el.get("role"), 40).lower()
    dom_role = _TAG_TO_ROLE.get(raw_dom_role, raw_dom_role)
    a11y_role = _normalize_text(a11y_el.role, 40).lower()

    # Role match
    if dom_role and dom_role == a11y_role:
        score += 2
    elif dom_role and a11y_role:
        for group in _COMPAT_GROUPS:
            if dom_role in group and a11y_role in group:
                score += 1
                break

    # Name match
    dom_name = _normalize_text(dom_el.get("name"), 80).lower()
    a11y_name = _normalize_text(a11y_el.name, 80).lower()
    if a11y_name and dom_name:
        if a11y_name == dom_name:
            score += 4
        elif a11y_name in dom_name or dom_name in a11y_name:
            score += 3
        elif a11y_name[:15] == dom_name[:15] and len(a11y_name) > 3:
            score += 2

    return score


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API — extract()
# ═══════════════════════════════════════════════════════════════════════════════

def _empty_extract_result(url: str = "") -> dict[str, Any]:
    """Return the canonical empty result payload when extraction finds nothing."""
    return {
        "elements": [],
        "refid_text": "(no interactive elements found)",
        "markdown": "(no interactive elements found)",
        "semantic_hash": 0,
        "element_count": 0,
        "page_title": "",
        "url": url,
        "source": "empty",
    }


async def extract(page, timeout: float = _DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Extract the accessibility tree from the page.

    Returns a dict with:
      - elements:       list[dict] — each has ref, role, name, bbox
      - refid_text:     str — ultra-compact LLM context (~15 chars/element)
      - markdown:       str — alias for refid_text (V11 backward compat)
      - semantic_hash:  int — deterministic hash for change detection
      - element_count:  int — number of interactive elements
      - page_title:     str
      - url:            str
      - source:         str — which tier produced the result
    """
    if page is None:
        return _empty_extract_result()

    safe_timeout = _sanitize_timeout(timeout)
    snapshot: A11ySnapshot | None = None
    source = "empty"

    # ── Tier 1: snapshotForAI (fastest, pre-filtered) ──
    snapshot = await _extract_via_snapshot_for_ai(page, timeout_ms=int(safe_timeout * 1000))
    if snapshot and snapshot.element_count >= 1:
        source = "snapshotForAI"
        log.info(
            "A11y Tier-1 (snapshotForAI): %d elements, %d chars",
            snapshot.element_count, len(snapshot.refid_text),
        )
    else:
        # ── Tier 2: CDP AXTree (robust, has backend node IDs) ──
        snapshot = await _extract_via_cdp_ax_tree(page, timeout=safe_timeout)
        if snapshot and snapshot.element_count >= 1:
            source = "cdp_ax_tree"
            log.info(
                "A11y Tier-2 (CDP AXTree): %d elements",
                snapshot.element_count,
            )
        else:
            # ── Tier 3: Standard a11y snapshot (stable public API) ──
            snapshot = await _extract_via_a11y_snapshot(page, timeout=safe_timeout)
            if snapshot and snapshot.element_count >= 1:
                source = "a11y_snapshot"
                log.info(
                    "A11y Tier-3 (a11y.snapshot): %d elements",
                    snapshot.element_count,
                )

    if snapshot and snapshot.element_count >= 1:
        # Enrich all elements with pixel-perfect bounding boxes
        await _enrich_with_bounding_boxes(page, snapshot)

        result = snapshot.to_dict()
        result["source"] = source
        result["markdown"] = snapshot.refid_text  # V11 backward compat key
        return result

    # ── Tier 4: V11 dom_parser (legacy fallback, called by consumer) ──
    log.warning("A11y: All 3 tiers found 0 elements — consumer should try V11 dom_parser")
    url = ""
    try:
        url = page.url
    except Exception:
        pass
    return _empty_extract_result(url=url)


# ═══════════════════════════════════════════════════════════════════════════════
#  RefID Resolver — Maps RefIDs to Playwright Actions
# ═══════════════════════════════════════════════════════════════════════════════

class RefIDResolver:
    """Resolves RefIDs to Playwright actions.

    Usage:
        data = await a11y_parser.extract(page)
        resolver = RefIDResolver(page, data)
        await resolver.click("e5")
        await resolver.type_text("e3", "hello@example.com")
    """

    def __init__(self, page, snapshot_data: dict | A11ySnapshot | None):
        self._page = page
        self._elements: dict[str, dict[str, Any]] = {}
        if isinstance(snapshot_data, A11ySnapshot):
            self._elements = {e.ref: e.to_dict() for e in snapshot_data.elements if e.ref}
        elif isinstance(snapshot_data, dict):
            raw_els = snapshot_data.get("elements")
            if isinstance(raw_els, list):
                for e in raw_els:
                    if isinstance(e, dict):
                        ref = e.get("ref")
                        if isinstance(ref, str) and ref.strip():
                            self._elements[ref] = e

    def get(self, ref: str) -> dict | None:
        """Get element data by RefID."""
        if not isinstance(ref, str) or not ref:
            return None
        return self._elements.get(ref)

    async def click(self, ref: str) -> bool:
        """Click an element by its RefID. Uses bbox center → role locator fallback."""
        if self._page is None or not isinstance(ref, str) or not ref:
            return False

        el = self._elements.get(ref)
        if not el:
            log.warning("RefID '%s' not found in %d elements", ref, len(self._elements))
            return False

        role = el.get("role", "") if isinstance(el.get("role"), str) else ""
        name = el.get("name", "") if isinstance(el.get("name"), str) else ""
        center = _valid_bbox_center(el.get("bbox"))

        # Strategy 1: Pixel-perfect click via bounding box center
        if center is not None:
            cx, cy = center
            try:
                await self._page.mouse.click(cx, cy)
                log.info("Clicked %s '%s' at (%.0f, %.0f) via bbox", role, name, cx, cy)
                return True
            except Exception as e:
                log.debug("Bbox click failed for %s: %s", ref, e)

        # Strategy 2: Playwright get_by_role locator
        if role:
            try:
                locator = self._page.get_by_role(role, name=name)
                count = await locator.count()
                if count > 0:
                    await locator.first.click(timeout=5000)
                    log.info("Clicked %s '%s' via get_by_role", role, name)
                    return True
            except Exception as e:
                log.debug("Role locator click failed for %s: %s", ref, e)

        # Strategy 3: Try by text content
        if name:
            try:
                locator = self._page.get_by_text(name, exact=False)
                count = await locator.count()
                if count > 0:
                    await locator.first.click(timeout=5000)
                    log.info("Clicked %s '%s' via text match", role, name)
                    return True
            except Exception:
                pass

        log.warning("All click strategies failed for RefID '%s' (%s '%s')", ref, role, name)
        return False

    async def type_text(self, ref: str, text: str) -> bool:
        """Type text into an element by its RefID."""
        if self._page is None or not isinstance(ref, str) or not ref or not isinstance(text, str):
            return False

        el = self._elements.get(ref)
        if not el:
            log.warning("RefID '%s' not found", ref)
            return False

        role = el.get("role", "") if isinstance(el.get("role"), str) else ""
        name = el.get("name", "") if isinstance(el.get("name"), str) else ""
        center = _valid_bbox_center(el.get("bbox"))

        # Strategy 1: Click bbox center → keyboard type
        if center is not None:
            cx, cy = center
            try:
                await self._page.mouse.click(cx, cy)
                await asyncio.sleep(0.1)
                await self._page.keyboard.type(text, delay=25)
                log.info("Typed %d chars into %s '%s' via bbox", len(text), role, name)
                return True
            except Exception as e:
                log.debug("Bbox type failed for %s: %s", ref, e)

        # Strategy 2: Fill via role locator
        if role:
            try:
                locator = self._page.get_by_role(role, name=name)
                count = await locator.count()
                if count > 0:
                    await locator.first.fill(text, timeout=5000)
                    log.info("Filled %d chars into %s '%s' via locator", len(text), role, name)
                    return True
            except Exception as e:
                log.debug("Role fill failed for %s: %s", ref, e)

        # Strategy 3: Label-based fill
        if name:
            try:
                locator = self._page.get_by_label(name)
                count = await locator.count()
                if count > 0:
                    await locator.first.fill(text, timeout=5000)
                    log.info("Filled %d chars into %s '%s' via label", len(text), role, name)
                    return True
            except Exception:
                pass

        log.warning("All type strategies failed for RefID '%s' (%s '%s')", ref, role, name)
        return False

    async def scroll(self, delta_y: int = 500) -> bool:
        """Scroll the page by delta pixels."""
        if self._page is None or not _is_finite_number(delta_y):
            return False
        try:
            await self._page.mouse.wheel(0, float(delta_y))
            return True
        except Exception:
            return False
