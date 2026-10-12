"""Launch Windows Chrome with a DevTools (CDP) endpoint for the agent.

Run this with Windows Python (not inside WSL):  python windows_chrome_bridge.py

Security:
  * The debugging port listens on 127.0.0.1 ONLY. Whoever can reach it fully
    controls the browser (cookies, sessions, every page), so it is never bound
    to 0.0.0.0 / the LAN. (Current Chrome/Edge ignore --remote-debugging-address
    for headed browsers anyway and always bind loopback.)
  * Chrome 136+ refuses remote debugging on the default profile, so the bridge
    always uses a dedicated profile folder — your everyday Chrome profile is
    never exposed.

Reaching it from WSL2: the default NAT networking cannot see Windows' 127.0.0.1.
Either run the agent with Windows Python (.venv-windows), or enable WSL mirrored
networking (networkingMode=mirrored in %UserProfile%\\.wslconfig).
"""

import os
import subprocess
import sys
import time

DEBUG_PORT = 9222
PROFILE_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "AgentFirstBrowse", "chrome-cdp-profile",
)


def find_chrome():
    paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def main():
    print("==================================================")
    print("  AGENT FIRST BROWSE — LOCAL CHROME BRIDGE (WINDOWS)")
    print("==================================================\n")

    chrome_path = find_chrome()
    if not chrome_path:
        print("[ERROR] Chrome not found in standard paths.")
        input("Press Enter to exit...")
        sys.exit(1)

    os.makedirs(PROFILE_DIR, exist_ok=True)
    print(f"[INFO] Found Chrome: {chrome_path}")
    print(f"[INFO] Dedicated agent profile: {PROFILE_DIR}")

    args = [
        chrome_path,
        f"--remote-debugging-port={DEBUG_PORT}",  # binds 127.0.0.1 only
        f"--user-data-dir={PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
    ]

    try:
        process = subprocess.Popen(args, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        print(f"[SUCCESS] Chrome launched (PID: {process.pid})")
        print(f"[SUCCESS] CDP endpoint: http://127.0.0.1:{DEBUG_PORT} (this machine only)")
        print("\nAttach a CDP client to that endpoint (e.g. Playwright's")
        print("chromium.connect_over_cdp). Note: ./agent.sh / run_v16.py currently")
        print("launch their own browser and do not attach to this one.")
        print("\nYou can minimize this window; close it to stop the bridge.")

        # Keep alive
        while True:
            time.sleep(10)

    except Exception as e:
        print(f"\n[ERROR] Failed to launch Chrome: {e}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
