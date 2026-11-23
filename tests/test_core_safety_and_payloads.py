"""Real-world unit & adversarial tests for url_policy, truncate, and redaction."""

from __future__ import annotations

import base64
import json

import pytest

import core
from core import (
    SECRET_MASK,
    TRUNCATION_MARKER,
    ConfigValidationError,
    SafetyPolicyError,
    TruncateOptions,
    UrlPolicy,
    UrlPolicyViolation,
    is_private_or_local_host,
    is_secret_key,
    redact_base64_string,
    redact_inline_secrets,
    redact_payload,
    require_valid_url,
    truncate_result,
    truncate_text,
    validate_url,
)


def _compact_len(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


# ── 1. Public Barrel Coverage ──────────────────────────────────────────────────


def test_core_barrel_exports_chunk3_symbols():
    for name in (
        "UrlPolicy",
        "UrlPolicyViolation",
        "UrlValidation",
        "validate_url",
        "require_valid_url",
        "truncate_text",
        "truncate_result",
        "redact_payload",
        "redact_inline_secrets",
        "redact_base64_string",
    ):
        assert name in core.__all__
        assert hasattr(core, name)


# ── 2. URL Policy & SSRF Defense-in-Depth ──────────────────────────────────────


def test_url_policy_accepts_public_urls_and_strips_fragment():
    res = validate_url("HTTPS://Example.COM/docs/page?q=1#section-2")
    assert res.ok is True
    assert res.code is None
    assert res.host == "example.com"
    assert res.url == "https://Example.COM/docs/page?q=1"

    # Public IPv4 and IPv6 are allowed by default
    assert validate_url("https://1.1.1.1/dns-query").ok is True
    assert validate_url("https://[2606:4700:4700::1111]/").ok is True

    # Internationalized domain name canonicalizes to punycode
    idn = validate_url("https://m\u00fcnchen.example/home")
    assert idn.ok is True
    assert idn.host == "xn--mnchen-3ya.example"


@pytest.mark.parametrize(
    ("bad_input", "expected"),
    [
        (None, UrlPolicyViolation.INVALID_URL),
        (12345, UrlPolicyViolation.INVALID_URL),
        ("", UrlPolicyViolation.INVALID_URL),
        ("   ", UrlPolicyViolation.INVALID_URL),
        ("/relative/path", UrlPolicyViolation.INVALID_URL),
        ("https://example.com/a\r\nb", UrlPolicyViolation.INVALID_URL),
        ("https://example.com:99999/", UrlPolicyViolation.INVALID_URL),
        ("http://evil.com\\@127.0.0.1/", UrlPolicyViolation.INVALID_URL),
        ("file:///etc/passwd", UrlPolicyViolation.UNSUPPORTED_PROTOCOL),
        ("javascript:alert(1)", UrlPolicyViolation.UNSUPPORTED_PROTOCOL),
        ("data:text/html,<h1>x</h1>", UrlPolicyViolation.UNSUPPORTED_PROTOCOL),
        ("ftp://files.example.com/a.zip", UrlPolicyViolation.UNSUPPORTED_PROTOCOL),
        ("about:blank", UrlPolicyViolation.UNSUPPORTED_PROTOCOL),
        ("https://user:pass@example.com/", UrlPolicyViolation.CREDENTIALS),
        ("https://tokenonly@example.com/", UrlPolicyViolation.CREDENTIALS),
    ],
)
def test_url_policy_rejects_malformed_schemes_and_credentials(bad_input, expected):
    res = validate_url(bad_input)
    assert res.ok is False
    assert res.code is expected
    assert res.url is None


@pytest.mark.parametrize(
    "ssrf_url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.1:8080/admin",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/",
        "http://100.64.0.1/",
        # Legacy WHATWG/inet_aton IPv4 obfuscations
        "http://2130706433/",
        "http://0x7f000001/",
        "http://0x7f.1/",
        "http://0177.0.0.1/",
        "http://127.1/",
        # IPv6 loopback, ULA, link-local, zone-id, IPv4-mapped, 6to4
        "http://[::1]/",
        "http://[::]/",
        "http://[fc00::1]/",
        "http://[fe80::1]/",
        "http://[fe80::1%25eth0]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:169.254.169.254]/",
        "http://[2002:7f00:0001::]/",
        # Local-only & cloud metadata hostnames
        "http://localhost/",
        "http://localhost./",
        "http://api.localhost:3000/",
        "http://preview.internal/",
        "http://printer.local/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://metadata.goog/",
    ],
)
def test_url_policy_blocks_ssrf_private_and_obfuscated_hosts(ssrf_url):
    res = validate_url(ssrf_url)
    assert res.ok is False, f"Expected SSRF block for {ssrf_url}"
    assert res.code is UrlPolicyViolation.PRIVATE_HOST


def test_url_policy_opt_in_private_hosts_about_blank_and_length_validation():
    dev_policy = UrlPolicy(allow_private_hosts=True, allow_about_blank=True, max_url_length=80)
    assert validate_url("http://127.0.0.1:8080/test", dev_policy).ok is True
    assert validate_url("http://localhost:3000/app", dev_policy).ok is True
    assert validate_url("about:blank", dev_policy).ok is True

    too_long = "https://example.com/" + ("a" * 100)
    res_long = validate_url(too_long, dev_policy)
    assert res_long.ok is False
    assert res_long.code is UrlPolicyViolation.TOO_LONG

    assert is_private_or_local_host("127.0.0.1") is True
    assert is_private_or_local_host("example.com") is False

    with pytest.raises(ConfigValidationError):
        UrlPolicy(max_url_length=0)
    with pytest.raises(ConfigValidationError):
        UrlPolicy(max_url_length=True)  # type: ignore[arg-type]


def test_url_policy_allowlist_blocklist_precedence_and_require_valid_url():
    policy = UrlPolicy(
        allowed_hosts=("docs.example.com", "*.api.example.com"),
        blocked_hosts=("admin.api.example.com", "*.blocked.test"),
    )
    assert validate_url("https://docs.example.com/guide", policy).ok is True
    assert validate_url("https://v2.api.example.com/items", policy).ok is True
    assert validate_url("https://deep.v2.api.example.com/items", policy).ok is True

    # Wildcard *.api.example.com does NOT match bare api.example.com
    bare = validate_url("https://api.example.com/", policy)
    assert bare.ok is False
    assert bare.code is UrlPolicyViolation.HOST_NOT_ALLOWED

    # blocked_hosts wins even if allowed_hosts matches
    blocked = validate_url("https://admin.api.example.com/secret", policy)
    assert blocked.ok is False
    assert blocked.code is UrlPolicyViolation.BLOCKED_HOST

    # require_valid_url returns normalized string or raises non-retryable SafetyPolicyError
    assert require_valid_url("https://docs.example.com/a#frag", policy) == "https://docs.example.com/a"
    with pytest.raises(SafetyPolicyError) as exc_info:
        require_valid_url("http://169.254.169.254/latest")
    assert exc_info.value.retryable is False
    assert exc_info.value.policy == "url_policy"
    assert exc_info.value.violation == "private_host"


# ── 3. Structural Water-Filling Truncation ─────────────────────────────────────


def test_truncate_passthrough_when_within_budget_and_validation_errors():
    small_str = "hello world"
    assert truncate_text(small_str, max_chars=100) is small_str
    assert truncate_result(small_str, max_chars=100) is small_str

    small_obj = {"ok": True, "items": [1, 2, 3]}
    assert truncate_result(small_obj, max_chars=200) is small_obj

    with pytest.raises(ConfigValidationError):
        TruncateOptions(max_chars=0)
    with pytest.raises(ConfigValidationError):
        TruncateOptions(max_tokens=-1)
    with pytest.raises(ConfigValidationError):
        truncate_result("abc", max_chars=True)  # type: ignore[arg-type]


def test_truncate_text_and_water_filling_dict_preserve_small_fields():
    clipped = truncate_text("x" * 1000, max_chars=120, max_tokens=30)
    assert TRUNCATION_MARKER in clipped
    assert "tokens" in clipped

    payload = {
        "ok": True,
        "mode": "v2",
        "step": 3,
        "huge_dom": "button label text " * 400,
        "notes": "short note",
    }
    shrunk = truncate_result(payload, max_chars=300)
    assert isinstance(shrunk, dict)
    assert _compact_len(shrunk) <= 300
    # Water-filling preserves small sibling fields untouched
    assert shrunk["ok"] is True
    assert shrunk["mode"] == "v2"
    assert shrunk["step"] == 3
    assert shrunk["notes"] == "short note"
    assert TRUNCATION_MARKER in shrunk["huge_dom"]


def test_truncate_wide_dict_omits_largest_keys_and_avoids_marker_collision():
    wide = {f"key_{i}": ("v" * (20 + i * 35)) for i in range(25)}
    # Pre-populate the marker key itself to test collision disambiguation
    wide[TRUNCATION_MARKER] = "real-user-value"

    shrunk = truncate_result(wide, max_chars=450)
    assert isinstance(shrunk, dict)
    assert _compact_len(shrunk) <= 450
    # Original user key must never be overwritten by the truncation note
    if TRUNCATION_MARKER in shrunk:
        assert shrunk[TRUNCATION_MARKER] == "real-user-value"
    assert any(k.startswith(TRUNCATION_MARKER) and "keys omitted" in str(v) for k, v in shrunk.items())


def test_truncate_list_preserves_order_and_handles_escapes_and_unserializable():
    items = [{"id": f"e{i}", "label": f'Line "quoted"\nitem {i} ' + ("z" * 40)} for i in range(40)]
    shrunk = truncate_result(items, max_chars=500)
    assert isinstance(shrunk, list)
    assert _compact_len(shrunk) <= 500
    assert shrunk[0]["id"] == "e0"
    assert isinstance(shrunk[-1], str)
    assert TRUNCATION_MARKER in shrunk[-1]
    assert "more items" in shrunk[-1]

    # Heavy JSON escape inflation stays strictly within max_chars
    escaped_str = {'s': '"\n\\' * 300}
    shrunk_escaped = truncate_result(escaped_str, max_chars=120)
    assert _compact_len(shrunk_escaped) <= 120

    # Unserializable objects (cycles / custom types) return unchanged safely
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    assert truncate_result(cyclic, max_chars=50) is cyclic


# ── 4. Unified Base64, Screenshot & Secret Redaction ───────────────────────────


def test_redact_base64_threshold_data_url_and_screenshot_records():
    short_b64 = base64.b64encode(b"tiny").decode("ascii")
    assert redact_base64_string(short_b64) == short_b64

    big_raw = base64.b64encode(b"X" * 4000).decode("ascii")
    redacted_raw = redact_base64_string(big_raw)
    assert redacted_raw.startswith("[base64 data omitted:")
    assert big_raw not in redacted_raw

    big_data_url = f"data:image/png;base64,{big_raw}"
    redacted_url = redact_base64_string(big_data_url)
    assert redacted_url.startswith("[base64 image/png data omitted:")

    # Screenshot dict redacts even tiny base64 payloads regardless of threshold
    shot = {
        "type": "browser_screenshot",
        "mediaType": "image/jpeg",
        "data": short_b64,
        "width": 1280,
    }
    redacted_shot = redact_payload(shot)
    assert redacted_shot["data"].startswith("[base64 image/jpeg data omitted:")
    assert redacted_shot["width"] == 1280

    # Dedicated screenshot_base64 key is also redacted regardless of length
    mcp_shot = {"ok": True, "screenshot_base64": short_b64}
    assert redact_payload(mcp_shot)["screenshot_base64"].startswith("[base64 data omitted:")


def test_redact_secret_keys_and_preserves_benign_token_metrics():
    assert is_secret_key("password") is True
    assert is_secret_key("x-api-key") is True
    assert is_secret_key("Authorization") is True
    assert is_secret_key("access_token") is True
    assert is_secret_key("client_secret") is True

    # Metric keys containing "token" must NEVER be falsely redacted
    for benign in (
        "est_tokens",
        "avg_est_tokens",
        "total_est_tokens",
        "max_tokens",
        "token_count",
        "prompt_tokens",
    ):
        assert is_secret_key(benign) is False

    # Build synthetic secret values dynamically so this test file contains zero literal secrets
    fake_openai = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
    fake_gh = "ghp_" + "9876543210abcdefghij1234567890"
    fake_bearer = "Bearer " + "eyJhbGciOiJIUzI1NiJ9.payload.sig"
    fake_url_cred = "https://alice:" + "SuperSecretPass99" + "@github.com/org/repo.git"
    fake_pem = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIE...\n-----END " + "RSA PRIVATE KEY-----"

    payload = {
        "headers": {"Authorization": fake_bearer, "X-API-Key": fake_openai},
        "form": {"username": "sandeep", "password": "my-secret-password"},
        "metrics": {"est_tokens": 341, "avg_est_tokens": 341.0, "max_tokens": 6000},
        "log_line": f"Fetched {fake_url_cred} with {fake_gh} and {fake_pem}",
    }
    clean = redact_payload(payload)
    assert clean["headers"]["Authorization"] == SECRET_MASK
    assert clean["headers"]["X-API-Key"] == SECRET_MASK
    assert clean["form"]["username"] == "sandeep"
    assert clean["form"]["password"] == SECRET_MASK
    assert clean["metrics"] == {"est_tokens": 341, "avg_est_tokens": 341.0, "max_tokens": 6000}

    serialized = json.dumps(clean)
    for raw_secret in (fake_openai, fake_gh, fake_bearer, "SuperSecretPass99", "my-secret-password", "MIIE"):
        assert raw_secret not in serialized
    assert "https://[REDACTED]@github.com/org/repo.git" in clean["log_line"]
    assert redact_inline_secrets(f"token={fake_openai}") == "token=[REDACTED:openai_key]"


def test_redact_handles_cycles_shared_dag_depth_nodes_and_binary():
    # Shared non-cyclic dict must NOT trigger circular-reference omission
    shared = {"label": "ok"}
    dag = {"first": shared, "second": shared}
    redacted_dag = redact_payload(dag)
    assert redacted_dag == {"first": {"label": "ok"}, "second": {"label": "ok"}}

    # True cycle terminates cleanly
    cyc: dict[str, object] = {"name": "root"}
    cyc["self"] = cyc
    redacted_cyc = redact_payload(cyc)
    assert redacted_cyc["self"] == "[circular reference omitted]"

    # Depth limit & node limit bounds
    deep = {"a": {"b": {"c": {"d": "leaf"}}}}
    assert redact_payload(deep, max_depth=2)["a"]["b"]["c"] == "[nested value omitted]"

    wide_list = ["x"] * 20
    bounded = redact_payload(wide_list, max_nodes=5)
    assert "[remaining values omitted]" in bounded

    # Binary bytes/bytearray pass through untouched
    raw_blob = b"\x89PNG\r\n\x1a\n"
    assert redact_payload({"blob": raw_blob})["blob"] is raw_blob

    with pytest.raises(ConfigValidationError):
        redact_payload({}, max_nodes=0)


def test_combined_redact_then_truncate_mcp_response_pipeline():
    """End-to-end check simulating an oversized MCP tool response with secrets + screenshot."""
    fake_groq = "gsk_" + "Q" * 30
    big_shot = base64.b64encode(b"P" * 5000).decode("ascii")
    raw_mcp_payload = {
        "ok": True,
        "session": "v2",
        "metrics": {"est_tokens": 341, "settle_ms": 42.5},
        "screenshot": {"type": "browser_screenshot", "mediaType": "image/png", "data": big_shot},
        "debug_note": f"used key {fake_groq}",
        "elements": [{"ref": f"e{i}", "label": "Action item " + ("x" * 80)} for i in range(60)],
    }

    safe_bounded = truncate_result(redact_payload(raw_mcp_payload), max_chars=700)
    encoded = json.dumps(safe_bounded, ensure_ascii=False, separators=(",", ":"))
    assert len(encoded) <= 700
    assert fake_groq not in encoded
    assert big_shot not in encoded
    assert safe_bounded["ok"] is True
    assert safe_bounded["metrics"]["est_tokens"] == 341
