"""Unit and integration tests for a11y_parser.py (accessibility-tree extractor).

Covers:
  - ARIA snapshot text parsing edge cases (quoted brackets, escaped quotes,
    false-positive property isolation, duplicate/empty refs, invalid inputs).
  - Cross-process deterministic A11ySnapshot.semantic_hash.
  - Tier-2 CDP AXTree and Tier-3 a11y.snapshot malformed payloads, cycle
    protection, timeout forwarding, and secret/password value redaction.
  - Bounding-box scoring normalization and malformed DOM record resilience.
  - RefIDResolver edge cases (zero-coordinate bboxes, negative/NaN coords,
    malformed snapshot data, fallback chain, invalid inputs).
  - Live headless Chromium smoke test verifying extract() + secret masking.

Run: .venv/bin/python -m pytest tests/test_a11y_parser.py -v
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import a11y_parser as ap
from a11y_parser import (
    A11ySnapshot,
    RefElement,
    RefIDResolver,
    _enrich_with_bounding_boxes,
    _extract_via_a11y_snapshot,
    _extract_via_cdp_ax_tree,
    _match_score,
    _parse_aria_snapshot_text,
    _sanitize_timeout,
    _valid_bbox_center,
)

SRC = Path(__file__).resolve().parents[1] / "src"


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Helper Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_sanitize_timeout_handles_invalid_values():
    assert _sanitize_timeout(3.5) == 3.5
    assert _sanitize_timeout(0) == 5.0
    assert _sanitize_timeout(-2.0) == 5.0
    assert _sanitize_timeout(float("nan")) == 5.0
    assert _sanitize_timeout(float("inf")) == 5.0
    assert _sanitize_timeout(True) == 5.0
    assert _sanitize_timeout("5") == 5.0
    assert _sanitize_timeout(None, default=7.0) == 7.0


def test_valid_bbox_center_accepts_zero_and_rejects_bad_coords():
    assert _valid_bbox_center({"cx": 0, "cy": 0}) == (0.0, 0.0)
    assert _valid_bbox_center({"cx": 0, "cy": 42}) == (0.0, 42.0)
    assert _valid_bbox_center({"cx": 120.5, "cy": 80}) == (120.5, 80.0)
    assert _valid_bbox_center({"cx": -1, "cy": 20}) is None
    assert _valid_bbox_center({"cx": 20, "cy": -0.5}) is None
    assert _valid_bbox_center({"cx": math.nan, "cy": 10}) is None
    assert _valid_bbox_center({"cx": math.inf, "cy": 10}) is None
    assert _valid_bbox_center({"cx": True, "cy": 10}) is None
    assert _valid_bbox_center({"cx": "10", "cy": 20}) is None
    assert _valid_bbox_center(None) is None


# ═══════════════════════════════════════════════════════════════════════════════
#  2. ARIA Snapshot Text Parser Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_parse_aria_snapshot_preserves_brackets_and_escaped_quotes_in_names():
    raw = '\n'.join([
        '- button "Save [Draft]" [ref=e1] [disabled]',
        '- link "Say \\"Hello\\" \\\\ World" [ref=e2]',
        '- button "checked=true [disabled] [required]" [ref=e3]',
        '- heading "Release [v2]" [level=2] [ref=e4]',
    ])
    els = _parse_aria_snapshot_text(raw)
    assert len(els) == 4

    assert els[0].ref == "e1"
    assert els[0].role == "button"
    assert els[0].name == "Save [Draft]"
    assert els[0].properties == {"disabled": True}

    assert els[1].ref == "e2"
    assert els[1].name == 'Say "Hello" \\ World'
    assert els[1].properties == {}

    # Keywords inside quoted labels must NOT leak into element properties
    assert els[2].ref == "e3"
    assert els[2].name == "checked=true [disabled] [required]"
    assert els[2].properties == {}

    assert els[3].ref == "e4"
    assert els[3].role == "heading"
    assert els[3].name == "Release [v2]"
    assert els[3].properties == {"level": 2}


def test_parse_aria_snapshot_rich_properties_and_dedup_and_invalid_inputs():
    raw = "\n".join([
        '- checkbox "Remember me" [ref=e1] [checked]',
        '- checkbox "Select all" [ref=e2] [checked=mixed]',
        '- tab "Settings" [selected] [expanded] [ref=e3]',
        '- button "Toggle" [pressed] [ref=e4]',
        '- textbox "Email" [ref=e5] [required]: user@example.com',
        '- button [ref=e6]',
        '- button "Duplicate" [ref=e1]',
        '- button "Empty ref" [ref=   ]',
        '- generic "No ref here"',
    ])
    els = _parse_aria_snapshot_text(raw)
    by_ref = {e.ref: e for e in els}
    assert list(by_ref.keys()) == ["e1", "e2", "e3", "e4", "e5", "e6"]

    assert by_ref["e1"].properties == {"checked": True}
    assert by_ref["e2"].properties == {"checked": "mixed"}
    assert by_ref["e3"].properties == {"selected": True, "expanded": True}
    assert by_ref["e4"].properties == {"pressed": True}
    assert by_ref["e5"].name == "Email"
    assert by_ref["e5"].properties == {"required": True}
    assert by_ref["e6"].role == "button" and by_ref["e6"].name == ""

    assert _parse_aria_snapshot_text("") == []
    assert _parse_aria_snapshot_text("   \n\t ") == []
    assert _parse_aria_snapshot_text(None) == []  # type: ignore[arg-type]
    assert _parse_aria_snapshot_text(12345) == []  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Deterministic Semantic Hash & Compact Secret Rendering
# ═══════════════════════════════════════════════════════════════════════════════

def test_semantic_hash_is_deterministic_across_python_hash_seeds():
    script = (
        f"import sys; sys.path.insert(0, {str(SRC)!r}); "
        "import a11y_parser as ap; "
        "s = ap.A11ySnapshot([ap.RefElement('e1', 'button', 'Submit', properties={'disabled': False})]); "
        "print(s.semantic_hash)"
    )
    env1 = {**os.environ, "PYTHONHASHSEED": "1"}
    env2 = {**os.environ, "PYTHONHASHSEED": "42"}
    out1 = subprocess.check_output([sys.executable, "-c", script], env=env1, text=True).strip()
    out2 = subprocess.check_output([sys.executable, "-c", script], env=env2, text=True).strip()
    assert out1 == out2 and int(out1) != 0

    assert A11ySnapshot([]).semantic_hash == 0
    s_a = A11ySnapshot([RefElement("e1", "button", "Save")])
    s_b = A11ySnapshot([RefElement("e1", "button", "Cancel")])
    assert s_a.semantic_hash != s_b.semantic_hash


def test_ref_element_to_compact_masks_secrets():
    el = RefElement("e1", "textbox", "Password", properties={"secret": True, "value": "should-never-show"})
    compact = el.to_compact()
    assert "[value hidden]" in compact
    assert "should-never-show" not in compact


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Tier-2 CDP AXTree & Tier-3 a11y.snapshot Hardening
# ═══════════════════════════════════════════════════════════════════════════════

class _FakeCDPClient:
    def __init__(self, ax_result):
        self._ax_result = ax_result
        self.detached = False

    async def send(self, method, params=None):
        if method == "Accessibility.enable":
            return {}
        if method == "Accessibility.getFullAXTree":
            if isinstance(self._ax_result, Exception):
                raise self._ax_result
            return self._ax_result
        return {}

    async def detach(self):
        self.detached = True


def _fake_page_with_cdp(ax_result):
    client = _FakeCDPClient(ax_result)

    class _Ctx:
        async def new_cdp_session(self, page):
            return client

    class _Page:
        url = "https://example.test/login"
        context = _Ctx()

        async def title(self):
            return "Login"

    return _Page(), client


async def test_cdp_ax_tree_handles_malformed_nodes_and_redacts_secrets():
    ax_result = {
        "nodes": [
            "not-a-dict",
            {"ignored": True, "role": {"value": "button"}, "name": {"value": "Hidden"}},
            {"role": None, "name": None, "properties": None},
            {
                "role": {"value": "textbox"},
                "name": {"value": "Email\n  Address"},
                "description": None,
                "backendDOMNodeId": 101,
                "properties": [
                    "malformed-prop",
                    {"name": "required", "value": {"value": True}},
                    {"name": "value", "value": {"value": "alice@example.com"}},
                ],
            },
            {
                "role": {"value": "textbox"},
                "name": {"value": "Password"},
                "backendDOMNodeId": 102,
                "properties": [
                    {"name": "protected", "value": {"value": True}},
                    {"name": "value", "value": {"value": "SuperSecret!123"}},
                ],
            },
            {
                "role": {"value": "textbox"},
                "name": {"value": "One-Time OTP"},
                "backendDOMNodeId": 103,
                "properties": [
                    {"name": "value", "value": {"value": "998877"}},
                ],
            },
            {
                "role": {"value": "button"},
                "name": None,
                "backendDOMNodeId": 104,
            },
        ]
    }
    page, client = _fake_page_with_cdp(ax_result)
    snap = await _extract_via_cdp_ax_tree(page, timeout=2.0)

    assert client.detached is True
    assert snap is not None
    assert snap.element_count == 4

    email_el, pwd_el, otp_el, unnamed_btn = snap.elements
    assert email_el.name == "Email Address"
    assert email_el.properties["value"] == "alice@example.com"
    assert email_el.properties["required"] is True

    # Secrets must be redacted in both properties and rendered text
    assert pwd_el.properties.get("secret") is True
    assert "value" not in pwd_el.properties
    assert otp_el.properties.get("secret") is True
    assert "value" not in otp_el.properties
    assert "SuperSecret!123" not in snap.refid_text
    assert "998877" not in snap.refid_text

    # Explicit None name must become "", never literal "None"
    assert unnamed_btn.name == ""


async def test_a11y_snapshot_handles_none_fields_cycles_and_secrets():
    cyclic_child: dict = {"role": "button", "name": "Loop"}
    cyclic_child["children"] = [cyclic_child]

    tree = {
        "role": "WebArea",
        "name": None,
        "children": [
            None,
            {"role": None, "name": None, "children": None},
            {"role": "textbox", "name": "Passcode", "value": "4321", "protected": True},
            {"role": "textbox", "name": "Username", "value": "sandeep"},
            cyclic_child,
        ],
    }

    class _A11y:
        async def snapshot(self):
            return tree

    class _Page:
        url = "https://example.test/form"
        accessibility = _A11y()

        async def title(self):
            return "Form"

    snap = await _extract_via_a11y_snapshot(_Page(), timeout=2.0)
    assert snap is not None
    assert snap.element_count == 3

    pwd, user, loop_btn = snap.elements
    assert pwd.properties.get("secret") is True
    assert "value" not in pwd.properties
    assert "4321" not in snap.refid_text

    assert user.properties.get("value") == "sandeep"
    assert loop_btn.name == "Loop"


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Bounding-Box Matching & Enrichment Resilience
# ═══════════════════════════════════════════════════════════════════════════════

def test_match_score_normalizes_uppercase_tags_and_handles_none():
    link_el = RefElement("e1", "link", "Pricing")
    assert _match_score(link_el, {"role": "A", "name": "Pricing"}) == 6
    assert _match_score(link_el, {"role": None, "name": None}) == 0
    assert _match_score(link_el, "invalid") == 0  # type: ignore[arg-type]


async def test_enrich_with_bounding_boxes_skips_malformed_entries():
    dom_rows = [
        "not-a-dict",
        {"role": "button", "name": "Submit", "x": 10, "y": 20},  # missing w/h/cx/cy
        {"role": "button", "name": "Submit", "x": 0, "y": 0, "w": 0, "h": 10, "cx": 0, "cy": 5},  # zero width
        {"role": "BUTTON", "name": "Submit", "x": 10, "y": 20, "w": 80, "h": 30, "cx": 50, "cy": 35},
    ]

    class _Page:
        async def evaluate(self, js):
            return dom_rows

    snap = A11ySnapshot([RefElement("e1", "button", "Submit")])
    await _enrich_with_bounding_boxes(_Page(), snap)
    assert snap.elements[0].bbox == {"x": 10, "y": 20, "width": 80, "height": 30, "cx": 50, "cy": 35}


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Public extract() Validation & Tier Cascade
# ═══════════════════════════════════════════════════════════════════════════════

async def test_extract_none_page_and_tier_fallback(monkeypatch):
    empty = await ap.extract(None)
    assert empty["source"] == "empty"
    assert empty["elements"] == []
    assert empty["semantic_hash"] == 0

    seen_timeouts: list[float] = []

    async def fake_t1(page, timeout_ms=10000):
        return None

    async def fake_t2(page, timeout=5.0):
        seen_timeouts.append(timeout)
        return None

    async def fake_t3(page, timeout=5.0):
        seen_timeouts.append(timeout)
        return A11ySnapshot([RefElement("e1", "button", "Go")], page_title="Demo", url="https://x.test")

    async def fake_bbox(page, snap):
        return None

    monkeypatch.setattr(ap, "_extract_via_snapshot_for_ai", fake_t1)
    monkeypatch.setattr(ap, "_extract_via_cdp_ax_tree", fake_t2)
    monkeypatch.setattr(ap, "_extract_via_a11y_snapshot", fake_t3)
    monkeypatch.setattr(ap, "_enrich_with_bounding_boxes", fake_bbox)

    dummy_page = SimpleNamespace(url="https://x.test")
    res = await ap.extract(dummy_page, timeout=-3.0)
    assert res["source"] == "a11y_snapshot"
    assert res["element_count"] == 1
    assert res["markdown"] == res["refid_text"]
    # Invalid negative timeout was sanitized to 5.0 and forwarded to Tier-2 and Tier-3
    assert seen_timeouts == [5.0, 5.0]


# ═══════════════════════════════════════════════════════════════════════════════
#  7. RefIDResolver Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class _RecordingLocator:
    def __init__(self, count_val: int, log_list: list, label: str):
        self._count = count_val
        self._log = log_list
        self._label = label

    async def count(self):
        return self._count

    @property
    def first(self):
        return self

    async def click(self, timeout=5000):
        self._log.append((f"click:{self._label}",))

    async def fill(self, text, timeout=5000):
        self._log.append((f"fill:{self._label}", text))


class _FakeActionPage:
    def __init__(self):
        self.events: list = []
        self.mouse = SimpleNamespace(
            click=self._mouse_click,
            wheel=self._mouse_wheel,
        )
        self.keyboard = SimpleNamespace(type=self._kb_type)

    async def _mouse_click(self, x, y):
        self.events.append(("mouse.click", x, y))

    async def _mouse_wheel(self, dx, dy):
        self.events.append(("mouse.wheel", dx, dy))

    async def _kb_type(self, text, delay=0):
        self.events.append(("keyboard.type", text))

    def get_by_role(self, role, name=""):
        return _RecordingLocator(1 if role else 0, self.events, f"role:{role}:{name}")

    def get_by_text(self, name, exact=False):
        return _RecordingLocator(1 if name else 0, self.events, f"text:{name}")

    def get_by_label(self, name):
        return _RecordingLocator(1 if name else 0, self.events, f"label:{name}")


async def test_resolver_handles_malformed_init_and_zero_or_negative_bboxes():
    page = _FakeActionPage()

    # Malformed snapshot inputs must not crash __init__
    assert RefIDResolver(page, None).get("e1") is None
    assert RefIDResolver(page, {"elements": None}).get("e1") is None
    r_mixed = RefIDResolver(page, {"elements": [
        "bad",
        {"role": "button"},  # missing ref
        {"ref": "  ", "role": "button"},  # blank ref
        {"ref": "e0", "role": "button", "name": "Corner", "bbox": {"cx": 0, "cy": 0}},
        {"ref": "e_neg", "role": "button", "name": "Offscreen", "bbox": {"cx": -50, "cy": 10}},
        {"ref": "e_in", "role": "textbox", "name": "Search", "bbox": {"cx": 0, "cy": 25}},
    ]})

    # Zero-coordinate bbox (cx=0, cy=0) MUST be treated as valid, not falsy!
    assert await r_mixed.click("e0") is True
    assert page.events[-1] == ("mouse.click", 0.0, 0.0)

    # Negative off-screen bbox MUST be skipped in favor of locator fallback
    assert await r_mixed.click("e_neg") is True
    assert page.events[-1] == ("click:role:button:Offscreen",)

    # Zero-coordinate bbox in type_text works too
    assert await r_mixed.type_text("e_in", "hello") is True
    assert ("mouse.click", 0.0, 25.0) in page.events
    assert ("keyboard.type", "hello") in page.events

    # Invalid arguments fail cleanly
    assert await r_mixed.click("") is False
    assert await r_mixed.click(None) is False  # type: ignore[arg-type]
    assert await r_mixed.type_text("e_in", None) is False  # type: ignore[arg-type]
    assert await r_mixed.scroll(float("nan")) is False  # type: ignore[arg-type]
    assert await r_mixed.scroll(True) is False  # type: ignore[arg-type]
    assert await r_mixed.scroll(300) is True
    assert page.events[-1] == ("mouse.wheel", 0, 300.0)

    # None page always returns False cleanly
    r_nopage = RefIDResolver(None, {"elements": [{"ref": "e1", "role": "button", "name": "Ok"}]})
    assert await r_nopage.click("e1") is False
    assert await r_nopage.type_text("e1", "hi") is False
    assert await r_nopage.scroll(100) is False


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Live Headless Chromium Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

def test_a11y_extract_live_chromium_smoke():
    html = """<!doctype html><html><head><title>A11y Smoke</title></head><body>
      <main>
        <label for="u">Username</label>
        <input id="u" type="text" value="alice" />
        <label for="p">Password</label>
        <input id="p" type="password" value="TopSecretPass99" />
        <button id="go">Save [Draft]</button>
        <a href="/help">Help Center</a>
      </main>
    </body></html>"""

    async def run():
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1000, "height": 700})
            await page.set_content(html)
            data = await ap.extract(page, timeout=5.0)
            assert data["source"] in ("snapshotForAI", "cdp_ax_tree", "a11y_snapshot"), data
            assert data["element_count"] >= 3, data
            names = [e["name"] for e in data["elements"]]
            assert any("Save [Draft]" in n for n in names), names
            assert "TopSecretPass99" not in data["refid_text"]
            for el in data["elements"]:
                assert "TopSecretPass99" not in str(el.get("properties", {}))
        finally:
            await browser.close()
            await pw.stop()

    asyncio.run(run())
