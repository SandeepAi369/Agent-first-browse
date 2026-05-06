import asyncio
from playwright.async_api import async_playwright
import os

async def get_latest_post():
    async with async_playwright() as p:
        user_data_dir = os.path.join(os.getcwd(), "persistence", "browser_sessions", "agent_main")
        print(f"Launching persistent profile: {user_data_dir}")
        
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        
        page = browser.pages[0] if browser.pages else await browser.new_page()
        print("Navigating to Dev.to dashboard...")
        await page.goto("https://dev.to/dashboard", wait_until="domcontentloaded")
        
        await asyncio.sleep(2)
        
        print("Finding latest post...")
        await page.wait_for_selector("a", state="attached", timeout=10000)
        
        # Grab all links and filter by the slug we know the agent typed
        links = await page.eval_on_selector_all('a', 'elements => elements.map(e => e.href)')
        
        found = False
        for href in links:
            if 'searchwala' in href.lower() or 'rust' in href.lower() or 'rewrote' in href.lower() or 'engine' in href.lower():
                if href.startswith('https://dev.to/') and len(href.split('/')) > 4 and href.split('/')[3] != 'search':
                    print(f"\nSUCCESS! Latest post found:\nURL: {href}")
                    found = True
                    break
                
        if not found:
            print("Could not find post by slug.")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(get_latest_post())
