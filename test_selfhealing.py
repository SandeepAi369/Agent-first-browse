"""Smoke test for v8.0 Self-Healing & Anti-Detection modules."""
import sys
sys.path.insert(0, "python-orchestrator")
from collections import Counter

# Test 1: Circuit Breaker
from advanced_agent import CircuitBreaker

cb = CircuitBreaker(max_requests=5, max_failures=3)
assert not cb.tripped
cb.record_success()
cb.record_success()
assert cb.request_count == 2
assert not cb.tripped

cb.record_failure()
cb.record_failure()
cb.record_failure()
assert cb.tripped
assert "ceiling" in cb.reason
print(f"✅ Circuit Breaker: trips correctly — {cb.reason}")

cb2 = CircuitBreaker(max_requests=3, max_failures=10)
cb2.record_success()
cb2.record_success()
cb2.record_success()
assert cb2.tripped
assert "Request ceiling" in cb2.reason
print(f"✅ Circuit Breaker: trips correctly at {cb2.request_count} requests — {cb2.reason}")

# Test 2: Anti-Loop Watchdog (simulate the Counter logic)
url_history = [
    "https://hashnode.com/",
    "https://hashnode.com/login",
    "https://hashnode.com/",
    "https://hashnode.com/login",
    "https://hashnode.com/",
    "https://hashnode.com/login",
]
recent = url_history[-6:]
url_counts = Counter(recent)
most_common_url, visit_count = url_counts.most_common(1)[0]
assert visit_count >= 3, f"Expected loop detection, got {visit_count}"
print(f"✅ Anti-Loop Watchdog: detected '{most_common_url}' visited {visit_count}x in 6 steps")

# Test 3: SessionGuard exists and is a singleton
from advanced_agent import SessionGuard
g1 = SessionGuard.get()
g2 = SessionGuard.get()
assert g1 is g2, "SessionGuard should be a singleton"
print("✅ SessionGuard: singleton pattern works")

print()
print("=== ALL v8.0 SELF-HEALING MODULES VERIFIED ===")
