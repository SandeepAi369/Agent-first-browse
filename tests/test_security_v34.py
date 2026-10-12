"""Security + privacy regressions (V34 hardening).

Covers:
  • Navigation: only http(s) and about:blank — file:, javascript:, data:, … refused.
  • Uploads: only files in an upload folder (no hidden paths) or named in the
    objective; symlinks judged by their real target; upload_file resolves ids via
    the window.__aid registry (native file input AND custom chooser button).
  • Secrets: password fields are perceived as "value hidden" (dom_parser) and are
    not echoed by type verification (cdp_input).
  • PowerShell driver: typed text is emitted as UTF-16 code units, never source.
  • Persistence: the brain checkpoints in memory only; the worker's dedup
    signature carries a keyed digest, not the typed text.
  • Shutdown: Ctrl+C in run_brain / legacy run_agent closes the browser AND stops
    the Playwright driver; run.py stops its driver and keeps TLS/CSP checks on.

Run: .venv/bin/python -m pytest tests/test_security_v34.py -v
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import execution_safety as es


# ═══════════════════════════════════════════════════════════════════════════════
#  Navigation scheme policy
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("url", [
    "https://example.com/a", "http://example.com", "HTTPS://Example.com/x", "about:blank",
])
def test_http_and_blank_allowed(url):
    assert es.is_domain_allowed(url)


@pytest.mark.parametrize("url", [
    "file:///home/user/.ssh/id_rsa", "FILE:///etc/passwd", "  file:///tmp/x",
    "javascript:alert(1)", "data:text/html,<h1>x</h1>", "chrome://settings",
    "view-source:https://example.com", "C:\\Users\\me\\secret.txt", "", "http:///etc/passwd",
])
def test_other_schemes_refused(url):
    assert not es.is_domain_allowed(url)


def test_blocked_domain_still_refused():
    assert not es.is_domain_allowed("https://www.paypal.com/login")


# ═══════════════════════════════════════════════════════════════════════════════
#  Upload policy
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def upload_env(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir()
    outside = tmp_path / "home"
    outside.mkdir()
    monkeypatch.setenv("AGENT_UPLOAD_DIRS", str(root))
    monkeypatch.setattr(es, "_OBJECTIVE_TEXT", "")
    return root, outside


def test_upload_inside_root_allowed(upload_env):
    root, _ = upload_env
    f = root / "photo.png"
    f.write_bytes(b"x")
    ok, resolved = es.is_upload_allowed(str(f))
    assert ok and resolved == str(f.resolve())


def test_upload_outside_root_refused(upload_env):
    _, outside = upload_env
    secret = outside / "id_rsa"
    secret.write_text("KEY")
    ok, reason = es.is_upload_allowed(str(secret))
    assert not ok and "upload folder" in reason


def test_hidden_path_in_root_refused(upload_env):
    root, _ = upload_env
    (root / ".ssh").mkdir()
    key = root / ".ssh" / "id_rsa"
    key.write_text("KEY")
    assert not es.is_upload_allowed(str(key))[0]
    env = root / ".env"
    env.write_text("API_KEY=x")
    assert not es.is_upload_allowed(str(env))[0]


def test_symlink_judged_by_real_target(upload_env):
    root, outside = upload_env
    secret = outside / "secret.txt"
    secret.write_text("S")
    link = root / "innocent.png"
    link.symlink_to(secret)
    assert not es.is_upload_allowed(str(link))[0]


def test_file_named_in_objective_allowed(upload_env):
    _, outside = upload_env
    f = outside / "resume final.pdf"
    f.write_bytes(b"%PDF")
    es.auto_allow_from_objective(f"Upload '{f}' to the careers form")
    assert es.is_upload_allowed(str(f))[0]


def test_substring_of_objective_does_not_count(upload_env, monkeypatch):
    _, outside = upload_env
    f = outside / "pass"
    f.write_text("x")
    monkeypatch.chdir(outside)
    es.auto_allow_from_objective("Log in with my password and post")
    assert not es.is_upload_allowed("pass")[0]


def test_missing_file_refused(upload_env):
    root, _ = upload_env
    ok, reason = es.is_upload_allowed(str(root / "nope.png"))
    assert not ok and "not found" in reason


# ═══════════════════════════════════════════════════════════════════════════════
#  Live browser checks (headless Chromium, network-free)
# ═══════════════════════════════════════════════════════════════════════════════

async def _page(html: str):
    from playwright.async_api import async_playwright
    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.launch(headless=True)
    except Exception as e:  # browser not installed (e.g. CI without playwright install)
        await pw.stop()
        pytest.skip(f"Chromium unavailable: {e}")
    page = await browser.new_page()
    await page.set_content(html, wait_until="domcontentloaded")
    return pw, browser, page


UPLOAD_HTML = """
<!doctype html><html><body>
<input id="native" type="file">
<input id="hidden" type="file" style="display:none">
<button id="custom" onclick="document.getElementById('hidden').click()">Upload CV</button>
<script>window.__aid = {e1: document.getElementById('native'),
                         e2: document.getElementById('custom')};</script>
</body></html>
"""


def test_upload_file_native_custom_and_refused(upload_env):
    import mcp_tools
    root, outside = upload_env
    good = root / "cv.pdf"
    good.write_bytes(b"%PDF")
    bad = outside / "id_rsa"
    bad.write_text("KEY")

    async def run():
        pw, browser, page = await _page(UPLOAD_HTML)
        mcp_tools.set_page(page)
        try:
            r1 = await mcp_tools.mcp_upload_file("e1", str(good))
            assert r1["success"], r1
            assert await page.evaluate("document.getElementById('native').files[0].name") == "cv.pdf"

            r2 = await mcp_tools.mcp_upload_file("e2", str(good))
            assert r2["success"], r2
            assert await page.evaluate("document.getElementById('hidden').files[0].name") == "cv.pdf"

            r3 = await mcp_tools.mcp_upload_file("e1", str(bad))
            assert not r3["success"] and "refused" in r3["error"].lower()

            r4 = await mcp_tools.mcp_upload_file("e99", str(good))
            assert not r4["success"] and "not found" in r4["error"]
        finally:
            await browser.close()
            await pw.stop()

    asyncio.run(run())


SECRET = "SuperSecretPassword!"
FORM_HTML = f"""
<!doctype html><html><body>
<form>
  <input id="user" type="text" aria-label="Username" value="tomsmith">
  <input id="pw" type="password" aria-label="Password" value="{SECRET}">
  <input id="otp" type="text" autocomplete="one-time-code" aria-label="Code" value="482913">
  <button type="submit">Login</button>
</form>
</body></html>
"""


def test_dom_parser_hides_secret_field_values():
    import dom_parser

    async def run():
        pw, browser, page = await _page(FORM_HTML)
        try:
            data = await dom_parser.extract(page)
            blob = repr(data)
            assert SECRET not in blob and "482913" not in blob
            labels = [e["text"] for e in data["elements"]]
            assert any("value hidden" in t and f"{len(SECRET)}ch" in t for t in labels), labels
            assert any('"tomsmith"' in t for t in labels), labels   # normal fields unchanged
        finally:
            await browser.close()
            await pw.stop()

    asyncio.run(run())


def test_type_verification_does_not_echo_password():
    from cdp_input import _verify_typed_text

    async def run():
        pw, browser, page = await _page(FORM_HTML)
        try:
            await page.focus("#pw")
            res = await _verify_typed_text(page, "different")
            assert res["actual_preview"] == "(hidden)"
            assert res["actual_length"] == len(SECRET)
            await page.focus("#user")
            res = await _verify_typed_text(page, "tomsmith")
            assert res["verified"] and res["actual_preview"] == "tomsmith"
        finally:
            await browser.close()
            await pw.stop()

    asyncio.run(run())


# ═══════════════════════════════════════════════════════════════════════════════
#  PowerShell driver — injection-proof literals
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "plain", "it's", "’); Write-Output PWNED; (’", "‘‛‚\"`$()", "emoji 😀", "{+}",
])
def test_ps_literal_round_trips_without_source_text(text):
    from app.browser_promoter.physical_input_driver import PhysicalInputDriver
    lit = PhysicalInputDriver._ps_string_literal(text)
    assert re.fullmatch(r"\[string\]::new\(\[char\[\]\]@\((0x[0-9A-F]{4},?)+\)\)", lit), lit
    units = [int(u, 16) for u in re.findall(r"0x([0-9A-F]{4})", lit)]
    assert b"".join(u.to_bytes(2, "little") for u in units).decode("utf-16-le") == text


# ═══════════════════════════════════════════════════════════════════════════════
#  Persistence + wiring guards
# ═══════════════════════════════════════════════════════════════════════════════

def test_brain_checkpoints_in_memory_only():
    src = (SRC / "brain_graph.py").read_text()
    assert "InMemorySaver()" in src
    assert "AsyncSqliteSaver" not in src and "brain_checkpoints.db\"" not in src


def test_worker_dedup_signature_has_no_raw_text():
    src = (SRC / "workers" / "base_worker.py").read_text()
    assert "text_fingerprint(proposed.get('text'))" in src
    assert "(proposed.get('text') or 'none')[:30]" not in src


def test_run_py_stops_driver_and_keeps_tls_checks():
    src = (SRC / "run.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "await pw.stop()" in code
    assert "ignore_https_errors=True" not in code and "bypass_csp=True" not in code


def test_no_launch_list_disables_site_isolation():
    for f in (SRC / "app/browser_promoter/cdp_stealth_launcher.py",
              SRC / "app/browser_promoter/google_stealth_auth_graph.py",
              SRC / "dom_parser.py"):
        assert not re.search(r"--disable-features=[^\"]*(IsolateOrigins|SitePerProcess|site-per-process)",
                             f.read_text()), f


# ═══════════════════════════════════════════════════════════════════════════════
#  Ctrl+C in the real entry points (fake browser objects, real signal)
# ═══════════════════════════════════════════════════════════════════════════════

class _Fake:
    def __init__(self, log, name):
        self._log, self._name = log, name

    async def close(self):
        self._log.append(f"{self._name}.close")

    async def stop(self):
        self._log.append(f"{self._name}.stop")


@pytest.fixture
def fresh_guard():
    from advanced_agent import SessionGuard
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    SessionGuard._instance = None
    yield SessionGuard
    SessionGuard.get().detach()
    SessionGuard._instance = None
    for s, h in saved.items():
        signal.signal(s, h)


@pytest.mark.parametrize("module, entry, inner", [
    ("advanced_agent", "run_agent", "_run_agent_session"),
    ("brain_graph", "run_brain", "_run_brain_session"),
])
def test_entry_point_ctrl_c_stops_browser_and_driver(fresh_guard, monkeypatch, module, entry, inner):
    import importlib
    mod = importlib.import_module(module)
    log: list[str] = []

    async def fake_session(*_args, **_kw):
        fresh_guard.get().attach(_Fake(log, "context"), playwright=_Fake(log, "playwright"))
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(5)
        log.append("not-cancelled")

    monkeypatch.setattr(mod, inner, fake_session)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(getattr(mod, entry)("objective"))
    assert log == ["context.close", "playwright.stop"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
