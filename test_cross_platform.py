"""Quick import and auto-detection verification test."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "python-orchestrator"))

from app.browser_promoter.browser_runtime import (
    BrowserRuntime,
    _auto_resolve_browser_mode,
    _is_windows_native,
    _is_linux_native,
    _is_wsl,
    _detect_environment,
    _linux_browser_candidates,
)
from app.browser_promoter.cdp_stealth_launcher import (
    STEALTH_USER_AGENT,
    get_stealth_user_agent,
    _LINUX_USER_AGENT,
    _WINDOWS_USER_AGENT,
)
from app.browser_promoter.state import BrowserMode
from app import config

print("=== CROSS-PLATFORM AUTO-DETECTION TEST ===")
print()
print(f"  Environment:      {_detect_environment()}")
print(f"  Windows Native:   {_is_windows_native()}")
print(f"  WSL:              {_is_wsl()}")
print(f"  Linux Native:     {_is_linux_native()}")
print()
print(f"  BROWSER_STRATEGY: {config.BROWSER_STRATEGY}")
print(f"  User Agent:       {STEALTH_USER_AGENT[:60]}...")
print(f"  Windows UA:       {_WINDOWS_USER_AGENT[:60]}...")
print(f"  Linux UA:         {_LINUX_USER_AGENT[:60]}...")
print()

resolved = _auto_resolve_browser_mode(BrowserMode.ISOLATED_SERVER)
print(f"  Auto-resolved mode: {resolved.value}")
print(f"  Linux browser candidates: {_linux_browser_candidates('chrome')[:4]}")
print()
print("=== ALL IMPORTS PASSED ===")
