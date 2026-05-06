import asyncio
from playwright.async_api import async_playwright
import os

async def take_screenshot():
    async with async_playwright() as p:
        user_data_dir = os.path.join(os.getcwd(), "persistence", "browser_sessions", "agent_main")
        print(f"Launching persistent profile: {user_data_dir}")
        
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=True,  # Run headless to ensure consistent rendering for the screenshot
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        
        page = browser.pages[0] if browser.pages else await browser.new_page()
        
        target_url = "https://dev.to/chidari_sandeep_c8e0478a1/why-i-rewrote-my-90-engine-meta-search-in-rust-41l5"
        print(f"Navigating to {target_url}...")
        
        try:
            await page.goto(target_url, wait_until="networkidle", timeout=30000)
        except Exception as e:
            print(f"Navigation timed out or failed: {e}")
            
        await asyncio.sleep(5) # Let any JS finish rendering
        
        output_path = r"C:\Users\Sandeep\.gemini\antigravity\brain\3b76f2d8-aa52-4b61-9082-9cc5eed929e4\browser_screenshot.png"
        
        # In WSL, we need to map the Windows path or save locally and then copy
        local_path = "browser_screenshot.png"
        await page.screenshot(path=local_path, full_page=True)
        print(f"Screenshot saved locally to {local_path}")
        
        # Copy to the Windows artifact directory so the UI can embed it
        os.system(f'cp "{local_path}" "/mnt/c/Users/Sandeep/.gemini/antigravity/brain/3b76f2d8-aa52-4b61-9082-9cc5eed929e4/"')
        print("Copied to artifacts directory.")
        
        html = await page.content()
        with open("page_dump.html", "w", encoding="utf-8") as f:
            f.write(html)
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(take_screenshot())
