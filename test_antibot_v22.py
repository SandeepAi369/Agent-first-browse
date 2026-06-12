"""Unit tests for V22 anti-bot realism (virtual display + stealth-script repair).

Guarantees under test:
  - virtual_display: a real DISPLAY short-circuits (no Xvfb spawned); a true
    start/stop cycle sets and clears DISPLAY; missing Xvfb degrades to False.
  - The stealth init script REGRESSION: it must contain no raw control chars
    (the `\\r\\n`-in-a-non-raw-string bug silently aborted WebGL/audio/font/
    WebRTC spoofing); the platform-consistent GPU selection must be present.

Run: .venv/bin/python -m pytest test_antibot_v22.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import virtual_display
from app.browser_promoter.cdp_stealth_launcher import STEALTH_INIT_SCRIPT


# ═══════════════════════════════════════════════════════════════════════════════
#  Stealth-script integrity (the high-value regression — a broken script silently
#  disabled half the fingerprint spoofing)
# ═══════════════════════════════════════════════════════════════════════════════

def test_stealth_script_has_no_raw_control_chars():
    # The bug: STEALTH_INIT_SCRIPT was a NON-raw string, so `\r\n` inside the
    # WebRTC regex became a real CR/LF, breaking the JS regex literal and
    # aborting every layer after it. A raw string keeps them literal.
    assert "\r" not in STEALTH_INIT_SCRIPT, "raw CR present — script will abort in JS"
    # The WebRTC SDP regex must still carry the literal escape for JS.
    assert "\\r\\n" in STEALTH_INIT_SCRIPT


def test_stealth_script_webgl_is_platform_consistent():
    # WebGL spoof must pick Linux (Mesa/OpenGL) vs Windows (D3D11) by platform —
    # a Linux UA with a Direct3D renderer is an instant inconsistency tell.
    assert "_LIN_GPUS" in STEALTH_INIT_SCRIPT
    assert "_WIN_GPUS" in STEALTH_INIT_SCRIPT
    assert "/Linux/.test(navigator.platform" in STEALTH_INIT_SCRIPT
    assert "Mesa" in STEALTH_INIT_SCRIPT  # Linux-style renderer present


def test_stealth_script_core_layers_present():
    for needle in ("webdriver", "UNMASKED", "AudioContext", "plugins"):
        assert needle in STEALTH_INIT_SCRIPT, needle


# ═══════════════════════════════════════════════════════════════════════════════
#  Virtual display
# ═══════════════════════════════════════════════════════════════════════════════

def test_xvfb_available_reflects_binary(monkeypatch):
    monkeypatch.setattr(virtual_display.shutil, "which", lambda _: "/usr/bin/Xvfb")
    assert virtual_display.xvfb_available() is True
    monkeypatch.setattr(virtual_display.shutil, "which", lambda _: None)
    assert virtual_display.xvfb_available() is False


def test_real_display_short_circuits(monkeypatch):
    # When a real DISPLAY exists we must NOT spawn Xvfb.
    monkeypatch.setenv("DISPLAY", ":0")
    spawned = {"called": False}
    monkeypatch.setattr(virtual_display.subprocess, "Popen",
                        lambda *a, **k: spawned.__setitem__("called", True))
    assert virtual_display.start_virtual_display() is True
    assert spawned["called"] is False


def test_no_xvfb_degrades_to_false(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(virtual_display.shutil, "which", lambda _: None)
    assert virtual_display.start_virtual_display() is False


@pytest.mark.skipif(not virtual_display.xvfb_available(),
                    reason="Xvfb not installed")
def test_real_start_stop_cycle(monkeypatch):
    # Genuine Xvfb lifecycle (xvfb is installed in this env).
    monkeypatch.delenv("DISPLAY", raising=False)
    ok = virtual_display.start_virtual_display(800, 600)
    try:
        assert ok is True
        assert os.environ.get("DISPLAY", "").startswith(":")
    finally:
        virtual_display.stop_virtual_display()
    assert os.environ.get("DISPLAY", "") == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
