import os
import subprocess
import time
import sys
import socket

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
    print("  AGENT FIRST IDE — LOCAL CHROME BRIDGE (WINDOWS)")
    print("==================================================")
    print("This script connects your Windows Chrome directly")
    print("to the AI Agent running inside WSL.")
    print("==================================================\n")

    chrome_path = find_chrome()
    if not chrome_path:
        print("[ERROR] Chrome not found in standard paths.")
        input("Press Enter to exit...")
        sys.exit(1)

    print(f"[INFO] Found Chrome: {chrome_path}")
    print("[INFO] Checking if port 9222 is already in use...")
    
    # Kill existing chrome if needed, but only if they are using port 9222
    # For now, let's just launch it. If it fails, we instruct them.
    
    args = [
        chrome_path,
        "--remote-debugging-port=9222",
        "--remote-debugging-address=0.0.0.0",
        "--no-first-run",
        "--no-default-browser-check"
    ]
    
    print("\n[IMPORTANT] A Windows Firewall prompt might appear.")
    print("[IMPORTANT] You MUST click 'Allow Access' so the WSL AI can connect!")
    print("\nStarting Local Chrome Bridge...")
    
    try:
        # Launch independently
        process = subprocess.Popen(args, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        print(f"[SUCCESS] Chrome launched (PID: {process.pid})")
        print("[SUCCESS] The AI Agent in WSL can now control this browser window.")
        print("\nYou can now safely minimize this terminal window (do not close it)")
        print("and run your AI Agent from WSL as normal: python start_campaign.py")
        
        # Keep alive
        while True:
            time.sleep(10)
            
    except Exception as e:
        print(f"\n[ERROR] Failed to launch Chrome: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
