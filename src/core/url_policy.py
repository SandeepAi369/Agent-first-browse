"""Defense-in-depth URL safety & SSRF policy validator.

Adapts Cloudflare's `web/url-policy.ts` to Python 3.11+ stdlib (`urllib.parse`
+ `ipaddress`) with explicit handling of Python-specific URL quirks (legacy
octal/hex/integer IPv4 forms, IPv4-mapped/6to4/Teredo IPv6 addresses, IPv6
zone IDs, IDN punycode canonicalization, and control-character smuggling).

Design principles:
  - `validate_url()` NEVER throws on untrusted input strings; it returns a
    typed `UrlValidation` value carrying a `UrlPolicyViolation` enum code.
  - `require_valid_url()` is a strict wrapper raising `SafetyPolicyError`
    (a `NonRetryableError`) when a caller wants fail-fast enforcement.
  - Private/loopback/link-local/metadata/CGNAT ranges and local hostnames
    (`localhost`, `*.localhost`, `*.internal`, `*.local`) are blocked by
    default, with explicit `allow_private_hosts=True` opt-in for local dev.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import SplitResult, urlsplit, urlunsplit

from core.errors import ConfigValidationError, SafetyPolicyError

DEFAULT_MAX_URL_LENGTH: int = 2000

_LOCAL_EXACT_HOSTS: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)

_LOCAL_SUFFIXES: tuple[str, ...] = (
    ".localhost",
    ".localhost.localdomain",
    ".local",
    ".internal",
    ".home.arpa",
)

_CONTROL_OR_WHITESPACE_RE = re.compile(r"[\x00-\x20\x7f]")


class UrlPolicyViolation(StrEnum):
    """Machine-readable reason why `validate_url()` rejected a URL."""

    INVALID_URL = "invalid_url"
    TOO_LONG = "too_long"
    UNSUPPORTED_PROTOCOL = "unsupported_protocol"
    CREDENTIALS = "credentials"
    BLOCKED_HOST = "blocked_host"
    HOST_NOT_ALLOWED = "host_not_allowed"
    PRIVATE_HOST = "private_host"


@dataclass(frozen=True)
class UrlPolicy:
    """Policy governing which URLs a browser or fetch tool may request.

    Host pattern syntax for `allowed_hosts` and `blocked_hosts`:
      - `"example.com"` matches that exact host only.
      - `"*.example.com"` matches subdomains at any depth, NOT `"example.com"`.
      - `"*"` matches any host.
    `blocked_hosts` is always evaluated before `allowed_hosts`.
    """

    allowed_hosts: tuple[str, ...] = ()
    blocked_hosts: tuple[str, ...] = ()
    allow_private_hosts: bool = False
    allow_about_blank: bool = False
    max_url_length: int = DEFAULT_MAX_URL_LENGTH

    def __post_init__(self) -> None:
        if isinstance(self.max_url_length, bool) or not isinstance(self.max_url_length, int):
            raise ConfigValidationError(
                f"url_policy.max_url_length must be an integer > 0, got {self.max_url_length!r}"
            )
        if self.max_url_length <= 0:
            raise ConfigValidationError(
                f"url_policy.max_url_length must be > 0, got {self.max_url_length}"
            )


@dataclass(frozen=True)
class UrlValidation:
    """Outcome of `validate_url()`.

    On success (`ok=True`), `url` holds the normalized URL with `#fragment`
    stripped and `host` holds the canonical lowercase ASCII host.
    On failure (`ok=False`), `code` and `message` explain why it was rejected.
    """

    ok: bool
    url: str | None = None
    code: UrlPolicyViolation | None = None
    message: str = ""
    host: str | None = None


def _reject(
    code: UrlPolicyViolation,
    message: str,
    host: str | None = None,
) -> UrlValidation:
    return UrlValidation(ok=False, url=None, code=code, message=message, host=host)


def _parse_ipv4_part(part: str) -> int | None:
    if not part:
        return None
    if part.startswith(("0x", "0X")):
        digits = part[2:]
        if not digits or not all(ch in "0123456789abcdefABCDEF" for ch in digits):
            return None
        return int(digits, 16)
    if len(part) > 1 and part.startswith("0"):
        if not all(ch in "01234567" for ch in part):
            return None
        return int(part, 8)
    if not part.isdigit():
        return None
    return int(part, 10)


def _parse_legacy_ipv4(host: str) -> ipaddress.IPv4Address | None:
    """Parse WHATWG/inet_aton IPv4 forms (integer, hex, octal, 2/3-part dotted)."""
    parts = host.split(".")
    if not (1 <= len(parts) <= 4):
        return None
    values: list[int] = []
    for part in parts:
        val = _parse_ipv4_part(part)
        if val is None:
            return None
        values.append(val)

    for prefix_val in values[:-1]:
        if prefix_val > 255:
            return None
    tail_limit = 256 ** (5 - len(values))
    if values[-1] >= tail_limit:
        return None

    addr_int = values[-1]
    for idx, prefix_val in enumerate(values[:-1]):
        addr_int += prefix_val * (256 ** (3 - idx))
    return ipaddress.IPv4Address(addr_int)


def _parse_ip_address(
    host: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    candidate = host.strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    # Strip IPv6 zone identifier (e.g. fe80::1%eth0 or fe80::1%25eth0)
    if "%" in candidate:
        candidate = candidate.split("%", 1)[0]
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        return _parse_legacy_ipv4(candidate)


def _is_non_public_ip(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None and _is_non_public_ip(addr.ipv4_mapped):
            return True
        if addr.sixtofour is not None and _is_non_public_ip(addr.sixtofour):
            return True
        if addr.teredo is not None:
            server, client = addr.teredo
            if _is_non_public_ip(server) or _is_non_public_ip(client):
                return True
    return (
        not addr.is_global
        or addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _canonicalize_host(raw_host: str) -> str | None:
    stripped = raw_host.strip().rstrip(".")
    if not stripped:
        return None
    ip_obj = _parse_ip_address(stripped)
    if ip_obj is not None:
        return str(ip_obj)
    if "%" in stripped or "\\" in stripped:
        return None
    try:
        ascii_host = stripped.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None
    if not ascii_host or ".." in ascii_host:
        return None
    return ascii_host


def is_private_or_local_host(host: str) -> bool:
    """Return True if `host` is a private/special IP or local-only hostname."""
    canonical = _canonicalize_host(host)
    if canonical is None:
        return True
    ip_obj = _parse_ip_address(canonical)
    if ip_obj is not None:
        return _is_non_public_ip(ip_obj)
    if canonical in _LOCAL_EXACT_HOSTS:
        return True
    return any(canonical.endswith(suffix) for suffix in _LOCAL_SUFFIXES)


def _matches_host_pattern(host: str, pattern: str) -> bool:
    raw = pattern.strip().lower().rstrip(".")
    if not raw:
        return False
    if raw == "*":
        return True
    if raw.startswith("*."):
        base = _canonicalize_host(raw[2:])
        if not base:
            return False
        return host.endswith("." + base) and len(host) > len(base) + 1
    canonical_pattern = _canonicalize_host(raw)
    return canonical_pattern is not None and host == canonical_pattern


def _matches_any_pattern(host: str, patterns: tuple[str, ...]) -> bool:
    return any(_matches_host_pattern(host, pat) for pat in patterns)


def _validate_authority(parsed: SplitResult) -> str | UrlValidation:
    if not parsed.netloc or "\\" in parsed.netloc:
        return _reject(UrlPolicyViolation.INVALID_URL, "URL is missing a valid host.")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        return _reject(
            UrlPolicyViolation.CREDENTIALS,
            "URLs with embedded username or password credentials are not allowed.",
        )
    try:
        _ = parsed.port
    except ValueError:
        return _reject(UrlPolicyViolation.INVALID_URL, "URL contains an invalid port.")

    raw_hostname = parsed.hostname
    if not raw_hostname:
        return _reject(UrlPolicyViolation.INVALID_URL, "URL is missing a valid hostname.")

    canonical_host = _canonicalize_host(raw_hostname)
    if canonical_host is None:
        return _reject(UrlPolicyViolation.INVALID_URL, f"Invalid hostname: {raw_hostname!r}.")
    return canonical_host


def _validate_structure(raw_url: str, active_policy: UrlPolicy) -> tuple[SplitResult, str] | UrlValidation:
    if len(raw_url) > active_policy.max_url_length:
        return _reject(
            UrlPolicyViolation.TOO_LONG,
            f"The URL is {len(raw_url)} characters; the limit is {active_policy.max_url_length}.",
        )
    if _CONTROL_OR_WHITESPACE_RE.search(raw_url):
        return _reject(
            UrlPolicyViolation.INVALID_URL,
            "URL must not contain whitespace or control characters.",
        )

    parsed = urlsplit(raw_url)
    scheme = parsed.scheme.lower()
    if not scheme:
        return _reject(UrlPolicyViolation.INVALID_URL, "Not a valid absolute URL.")
    if scheme not in ("http", "https"):
        return _reject(
            UrlPolicyViolation.UNSUPPORTED_PROTOCOL,
            f"Only http and https URLs are allowed, not {scheme}.",
        )

    authority = _validate_authority(parsed)
    if isinstance(authority, UrlValidation):
        return authority
    return parsed, authority


def validate_url(
    url: object,
    policy: UrlPolicy | None = None,
) -> UrlValidation:
    """Validate `url` against `policy` without throwing on untrusted input."""
    active_policy = policy if policy is not None else UrlPolicy()
    if not isinstance(url, str):
        return _reject(UrlPolicyViolation.INVALID_URL, "The URL must be a string.")

    trimmed = url.strip()
    if not trimmed:
        return _reject(UrlPolicyViolation.INVALID_URL, "The URL must not be empty.")

    if trimmed.lower() == "about:blank":
        if active_policy.allow_about_blank:
            return UrlValidation(ok=True, url="about:blank", host=None)
        return _reject(
            UrlPolicyViolation.UNSUPPORTED_PROTOCOL,
            "Only http and https URLs are allowed, not about.",
        )

    structural = _validate_structure(trimmed, active_policy)
    if isinstance(structural, UrlValidation):
        return structural
    parsed, canonical_host = structural

    if _matches_any_pattern(canonical_host, active_policy.blocked_hosts):
        return _reject(
            UrlPolicyViolation.BLOCKED_HOST,
            f"The host {canonical_host} is blocked.",
            host=canonical_host,
        )

    if active_policy.allowed_hosts and not _matches_any_pattern(
        canonical_host, active_policy.allowed_hosts
    ):
        return _reject(
            UrlPolicyViolation.HOST_NOT_ALLOWED,
            f"The host {canonical_host} is not allowed.",
            host=canonical_host,
        )

    if not active_policy.allow_private_hosts and is_private_or_local_host(canonical_host):
        return _reject(
            UrlPolicyViolation.PRIVATE_HOST,
            f"The host {canonical_host} is a private or local address.",
            host=canonical_host,
        )

    normalized = urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path, parsed.query, "")
    )
    return UrlValidation(ok=True, url=normalized, host=canonical_host)


def require_valid_url(
    url: object,
    policy: UrlPolicy | None = None,
) -> str:
    """Validate `url` and return its normalized form, or raise `SafetyPolicyError`."""
    result = validate_url(url, policy=policy)
    if not result.ok or result.url is None:
        violation = result.code.value if result.code is not None else "invalid_url"
        target = url[:200] if isinstance(url, str) else repr(url)[:200]
        raise SafetyPolicyError(
            result.message,
            policy="url_policy",
            violation=violation,
            target=target,
        )
    return result.url


__all__ = [
    "DEFAULT_MAX_URL_LENGTH",
    "UrlPolicy",
    "UrlPolicyViolation",
    "UrlValidation",
    "is_private_or_local_host",
    "require_valid_url",
    "validate_url",
]
