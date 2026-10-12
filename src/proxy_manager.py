"""ProxyManager — Proxy rotation with pre-burn detection.

Fixes V-19: Zero proxy support in the Agent First IDE codebase.

Provides:
  - Pool management (residential/datacenter/mobile tiers)
  - Health monitoring with pre-burn signal detection (403/429/503 spikes)
  - Integration with Playwright browser.new_context(proxy=...)
  - Automatic failover and quarantine on burn signals
  - Environment-based proxy loading (PROXY_POOL env var)

Usage:
    from proxy_manager import ProxyManager
    pm = ProxyManager()
    pm.load_from_env()  # reads PROXY_POOL="http://user:pass@host:port,..."

    proxy = pm.get_next()
    if proxy:
        context = await playwright.chromium.launch_persistent_context(
            ..., proxy=proxy.to_playwright_proxy()
        )
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger("proxy_manager")


# ═══════════════════════════════════════════════════════════════════════════════
#  Proxy Entry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProxyEntry:
    """A single proxy with health tracking."""
    server: str          # "http://host:port" or "socks5://host:port"
    username: str = ""
    password: str = ""
    proxy_type: str = "residential"  # residential | datacenter | mobile
    region: str = ""

    # Health tracking
    total_requests: int = 0
    failures: int = 0
    last_used: float = 0.0
    quarantined_until: float = 0.0
    latency_ema: float = 0.0       # Exponential moving average of response time
    burn_signals: int = 0           # Consecutive soft-block signals
    _status_history: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def is_healthy(self) -> bool:
        """Check if this proxy is usable right now."""
        if time.monotonic() < self.quarantined_until:
            return False
        if self.burn_signals >= 3:
            return False
        return True

    @property
    def success_rate(self) -> float:
        """Success rate over the last 20 requests."""
        if not self._status_history:
            return 1.0
        return sum(1 for s in self._status_history if s) / len(self._status_history)

    def to_playwright_proxy(self) -> dict:
        """Convert to Playwright proxy format for context creation."""
        result = {"server": self.server}
        if self.username:
            result["username"] = self.username
        if self.password:
            result["password"] = self.password
        return result

    def status_line(self) -> str:
        state = "HEALTHY" if self.is_healthy else "QUARANTINED"
        return (
            f"[{self.proxy_type}:{state}] {self.server[:30]} "
            f"({self.total_requests} reqs, {self.failures} fails, "
            f"burn={self.burn_signals}, latency={self.latency_ema:.0f}ms)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Proxy Manager
# ═══════════════════════════════════════════════════════════════════════════════

class ProxyManager:
    """Proxy pool manager with pre-burn detection and health monitoring.

    Pre-Burn Detection (from research):
      - Response Time Anomalies: Exponential moving average baseline
      - Status Code Patterns: Soft-block codes (403, 429, 503) before hard blocks
      - Challenge Redirects: URL pattern changes indicating imminent blocks
      - Consecutive burn signals trigger quarantine
    """

    # HTTP codes that signal impending block (pre-burn)
    BURN_STATUS_CODES = {403, 429, 503}

    # Quarantine duration in seconds after burn
    QUARANTINE_SECS = 300.0  # 5 minutes

    # Burn signal streak threshold before quarantine
    BURN_THRESHOLD = 3

    def __init__(self, proxies: list[dict] | None = None):
        self._pool: list[ProxyEntry] = []
        if proxies:
            for p in proxies:
                self._pool.append(ProxyEntry(**p))

    @property
    def has_proxies(self) -> bool:
        return len(self._pool) > 0

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    @property
    def healthy_count(self) -> int:
        return sum(1 for p in self._pool if p.is_healthy)

    def add_proxy(
        self,
        server: str,
        username: str = "",
        password: str = "",
        proxy_type: str = "residential",
        region: str = "",
    ) -> None:
        """Add a proxy to the pool."""
        self._pool.append(ProxyEntry(
            server=server,
            username=username,
            password=password,
            proxy_type=proxy_type,
            region=region,
        ))

    def get_next(self) -> ProxyEntry | None:
        """Get the next healthy proxy, preferring mobile > residential > datacenter.

        If all proxies are burned, returns the one with the shortest quarantine
        remaining (it will recover soonest).
        """
        if not self._pool:
            return None

        healthy = [p for p in self._pool if p.is_healthy]
        if not healthy:
            # All burned — return the one that will recover soonest
            self._pool.sort(key=lambda p: p.quarantined_until)
            chosen = self._pool[0]
            logger.warning(
                "⚠️ All proxies burned. Using least-quarantined: %s",
                chosen.server[:30],
            )
            return chosen

        # Priority: mobile > residential > datacenter
        # Within same type: fewer burn signals, fewer failures
        healthy.sort(key=lambda p: (
            0 if p.proxy_type == "mobile" else
            1 if p.proxy_type == "residential" else 2,
            p.burn_signals,
            p.failures,
            -p.success_rate,
        ))
        return healthy[0]

    def record_success(self, proxy: ProxyEntry, latency_ms: float = 0.0) -> None:
        """Record a successful request through this proxy."""
        proxy.total_requests += 1
        proxy.last_used = time.monotonic()
        proxy.burn_signals = 0  # Reset streak on success
        proxy._status_history.append(True)

        if latency_ms > 0:
            alpha = 0.3
            proxy.latency_ema = alpha * latency_ms + (1 - alpha) * proxy.latency_ema

    def record_failure(
        self,
        proxy: ProxyEntry,
        status_code: int = 0,
        is_challenge_redirect: bool = False,
    ) -> None:
        """Record a failed request. Triggers pre-burn detection."""
        proxy.total_requests += 1
        proxy.failures += 1
        proxy.last_used = time.monotonic()
        proxy._status_history.append(False)

        # Pre-burn signal detection
        is_burn = status_code in self.BURN_STATUS_CODES or is_challenge_redirect
        if is_burn:
            proxy.burn_signals += 1
            logger.warning(
                "🔥 Pre-burn signal on %s (code=%d, streak=%d/%d)",
                proxy.server[:30], status_code,
                proxy.burn_signals, self.BURN_THRESHOLD,
            )

        # Quarantine if burn threshold reached
        if proxy.burn_signals >= self.BURN_THRESHOLD:
            proxy.quarantined_until = time.monotonic() + self.QUARANTINE_SECS
            logger.warning(
                "🚫 Proxy burned and quarantined: %s — %ds cooldown",
                proxy.server[:30], int(self.QUARANTINE_SECS),
            )

    def load_from_env(self) -> None:
        """Load proxies from PROXY_POOL environment variable.

        Format: comma-separated proxy URLs
        Example: PROXY_POOL="http://user:pass@host1:port,socks5://host2:port"

        Optional type prefix: "residential://user:pass@host:port"
        """
        raw = os.getenv("PROXY_POOL", "").strip()
        if not raw:
            return

        for url in raw.split(","):
            url = url.strip()
            if not url:
                continue

            # Check for type prefix (e.g., "residential://...")
            proxy_type = "residential"
            for ptype in ("mobile", "residential", "datacenter"):
                if url.startswith(f"{ptype}://"):
                    proxy_type = ptype
                    url = url.replace(f"{ptype}://", "http://", 1)
                    break

            try:
                parsed = urlparse(url)
                server = f"{parsed.scheme}://{parsed.hostname}"
                if parsed.port:
                    server += f":{parsed.port}"
                self._pool.append(ProxyEntry(
                    server=server,
                    username=parsed.username or "",
                    password=parsed.password or "",
                    proxy_type=proxy_type,
                ))
            except Exception as e:
                logger.warning("Failed to parse proxy URL '%s': %s", url[:30], e)

        if self._pool:
            logger.info("📡 Loaded %d proxies from PROXY_POOL", len(self._pool))

    def status_report(self) -> str:
        """Human-readable status of all proxies."""
        if not self._pool:
            return "ProxyManager: No proxies configured"
        lines = [f"ProxyManager: {len(self._pool)} proxies ({self.healthy_count} healthy)"]
        for p in self._pool:
            lines.append(f"  {p.status_line()}")
        return "\n".join(lines)
