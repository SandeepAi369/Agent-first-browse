import sys
import os
from pathlib import Path
import asyncio

# Setup path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "python-orchestrator"))

# Load env
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from app.browser_promoter.state import BrowserConfig
from app.browser_promoter.browser_runtime import BrowserRuntime

async def main():
    print("Launching persistent Playwright browser...")
    page = await BrowserRuntime.ensure_page(browser_config=BrowserConfig(headless=False))
    await page.goto("https://accounts.google.com")
    print("Browser is now OPEN on your screen.")
    print("Please log into Gmail, Reddit, or any other platforms.")
    print("This window will remain open for 15 minutes or until you close the terminal.")
    
    # Keep it open for 15 minutes
    await asyncio.sleep(900)
    print("Closed.")

if __name__ == "__main__":
    asyncio.run(main())
