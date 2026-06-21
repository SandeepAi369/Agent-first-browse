# google_stealth_auth_graph.py
# 2026 Elite Google Wall Fighter — LangGraph + Playwright + CloakBrowser
import os
import asyncio
import random
import subprocess
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from playwright.async_api import Page, BrowserContext

try:
    from cloakbrowser import launch  # pip install cloakbrowser[geoip]
except ImportError:
    launch = None
from playwright.async_api import async_playwright
    
try:
    from agentmailr import AgentMailr  # pip install agentmailr
except ImportError:
    AgentMailr = None

try:
    from twocaptcha import TwoCaptcha
except ImportError:
    TwoCaptcha = None

# ====================== CONFIG ======================
PROFILES_DIR = "./persistence/browser_sessions"
AUTH_STATES_DIR = "./persistence/auth_states"

agentmailr_key = os.getenv("AGENTMAILR_KEY") or "DUMMY_KEY"
twocaptcha_key = os.getenv("TWOCAPTCHA_KEY") or "DUMMY_KEY"
CAPTCHA_SOLVER = TwoCaptcha(twocaptcha_key) if TwoCaptcha else None
MAILR = AgentMailr(api_key=agentmailr_key) if AgentMailr else None

class AgentState(TypedDict):
    page: Page
    context: BrowserContext
    email: str
    password: str
    action: str  # "login" or "continue"
    result: str | None
    vision_calls: int
    retries: int

# ====================== STEALTH LAUNCHER ======================
def _find_system_chrome() -> str | None:
    if os.path.exists("/usr/bin/google-chrome"):
        return "/usr/bin/google-chrome"
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for p in edge_paths:
        if os.path.exists(p):
            return p
            
    if "WSL_DISTRO_NAME" in os.environ:
        candidate_list = ",".join(f"'{p}'" for p in edge_paths)
        ps_script = (
            "$ErrorActionPreference='Stop';"
            f"$candidates=@({candidate_list});"
            "$exe=$candidates | Where-Object { Test-Path $_ } | Select-Object -First 1;"
            "Write-Output $exe;"
        )
        try:
            res = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps_script], capture_output=True, text=True)
            path = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else None
            if path and path.lower().endswith(".exe"):
                return path
        except Exception:
            pass
    return None

async def launch_google_stealth_context(profile_name: str) -> BrowserContext:
    """Dual-Mode Stealth Launcher: Native Chrome (Local) or Playwright/CloakBrowser (Cloud)"""
    context_dir = f"{PROFILES_DIR}/{profile_name}"
    os.makedirs(context_dir, exist_ok=True)
    
    is_local = os.getenv("ENVIRONMENT", "local").lower() == "local"
    chrome_exe = _find_system_chrome()
    
    if is_local and chrome_exe:
        # MODE 1: Local Native Chrome + CDP Connect
        if chrome_exe.endswith(".exe") and "WSL_DISTRO_NAME" in os.environ:
            res = subprocess.run(["wslpath", "-w", os.path.abspath(context_dir)], capture_output=True, text=True)
            win_data_dir = res.stdout.strip() if res.stdout.strip() else context_dir
            args = [
                f"--remote-debugging-port=9222",
                f"--user-data-dir={win_data_dir}",
                "--no-first-run",
                "--no-default-browser-check",
                "--password-store=basic",
                "--disable-features=IsolateOrigins,SitePerProcess",
                "--disable-blink-features=AutomationControlled"
            ]
            args_str = "','".join(args)
            ps_cmd = f"Start-Process -FilePath '{chrome_exe}' -ArgumentList @('{args_str}')"
            subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", ps_cmd])
        else:
            chrome_cmd = [
                chrome_exe,
                "--remote-debugging-port=9222",
                f"--user-data-dir={os.path.abspath(context_dir)}",
                "--no-first-run",
                "--no-default-browser-check",
                "--password-store=basic",
                "--disable-features=IsolateOrigins,SitePerProcess",
                "--disable-blink-features=AutomationControlled"
            ]
            subprocess.Popen(chrome_cmd)
            
        time.sleep(2)  # wait for CDP endpoint
        p = await async_playwright().start()
        browser = await p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0] if browser.contexts else await browser.new_context()

    else:
        # MODE 2: Cloud / Remote mode
        args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,SitePerProcess,OptimizeImageLoading",
            "--no-first-run", "--no-default-browser-check",
            "--password-store=basic",
            "--disable-features=EnableOverscroll",
            f"--fingerprint=marketing-agent-{profile_name}",
        ]
        
        if launch is None:
            p = await async_playwright().start()
            context = await p.chromium.launch_persistent_context(
                user_data_dir=context_dir,
                headless=True,
                args=args,
                ignore_default_args=["--enable-automation"],
                viewport={"width": 1920, "height": 1080},
            )
        else:
            browser = await launch(
                headless=True,
                humanize=True,
                proxy={"server": os.getenv("PROXY_URL"), "bypass": ".google.com,.gstatic.com"} if os.getenv("PROXY_URL") else None,
                persistent_context_dir=context_dir,
                geoip=True,
                args=args,
                ignore_default_args=["--enable-automation"],
            )
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            )
    
    # Final CDP kill-switch (extra layer)
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
        Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
        delete window.cdc_goog_xxx;
        delete window.cdc_goog_;
    """)
    
    return context

# ====================== HUMAN BEHAVIOR HELPERS ======================
async def human_scroll(page: Page, distance: int = 600):
    current = 0
    steps = random.randint(8, 14)
    for _ in range(steps):
        delta = distance / steps * (0.4 + random.random() * 0.6)
        await page.mouse.move(0, current + delta, steps=3)
        current += delta
        await asyncio.sleep(random.uniform(0.008, 0.028))
    await page.mouse.move(0, current + random.randint(-12, 12))

async def human_type(page: Page, selector: str, text: str):
    await page.focus(selector)
    for char in text:
        if random.random() < 0.07:
            await page.keyboard.type(random.choice("abcdefghijklmnopqrstuvwxyz"))
            await asyncio.sleep(0.04)
            await page.keyboard.press("Backspace")
        await page.keyboard.type(char)
        await asyncio.sleep(random.uniform(0.035, 0.14))

# ====================== OTP & CAPTCHA HANDLERS ======================
async def solve_otp(page: Page, email: str) -> str:
    """Zero-vision OTP via AgentMailr temp inbox"""
    if MAILR is None:
        return "NO_MAILR"
    inbox = await MAILR.inboxes.create(username=f"stealth-{email.split('@')[0]}-{random.randint(1000,9999)}")
    
    # Trigger Google OTP email
    await page.fill('input[type="tel"], input[autocomplete="one-time-code"]', "")
    await asyncio.sleep(2)  # let email fire
    
    otp = await inbox.wait_for_otp(timeout=90000, subject_contains="Google")
    await page.fill('input[autocomplete="one-time-code"]', otp)
    await page.keyboard.press("Enter")
    return otp

async def solve_captcha_if_present(page: Page) -> bool:
    """2Captcha fallback — only if DOM detects captcha"""
    if CAPTCHA_SOLVER is None:
        return False
    if "recaptcha" not in await page.content().lower():
        return False
    
    sitekey = await page.evaluate("""() => {
        const el = document.querySelector('.g-recaptcha') || document.querySelector('div[data-sitekey]');
        return el ? el.getAttribute('data-sitekey') : null;
    }""")
    if not sitekey:
        return False
    
    result = CAPTCHA_SOLVER.recaptcha(sitekey=sitekey, url=page.url)
    await page.evaluate(f"""(token) => {{
        document.getElementById('g-recaptcha-response').value = token;
        document.querySelector('form').submit();
    }}""", result['code'])
    return True

# ====================== CORE LANGGRAPH NODE ======================
async def google_auth_node(state: AgentState) -> AgentState:
    page = state["page"]
    retries = state.get("retries", 0)
    
    # 1. Text/DOM-first login (zero token)
    try:
        await human_scroll(page, 300)
        await page.wait_for_selector('input[type="email"], input[autocomplete="username"]', timeout=8000)
        await human_type(page, 'input[type="email"], input[autocomplete="username"]', state["email"])
        await page.keyboard.press("Enter")
        
        await asyncio.sleep(random.uniform(1.2, 2.8))
        
        # Password
        await human_type(page, 'input[type="password"]', state["password"])
        await page.keyboard.press("Enter")
        
        # 2. Smart OTP / Captcha check (text-first)
        await asyncio.sleep(3)
        
        if "one-time-code" in await page.content().lower() or "otp" in await page.content().lower():
            await solve_otp(page, state["email"])
            state["result"] = "otp_success"
            return {**state, "retries": 0}
        
        if await solve_captcha_if_present(page):
            state["result"] = "captcha_solved"
            return {**state, "retries": 0}
        
        # Success check
        if "myaccount.google.com" in page.url or "mail.google.com" in page.url:
            state["result"] = "login_success"
            # Save state for resurrection
            os.makedirs(AUTH_STATES_DIR, exist_ok=True)
            await state["context"].storage_state(path=f"{AUTH_STATES_DIR}/{state['email']}.json")
            return {**state, "retries": 0}
        
    except Exception as e:
        pass
    
    # 3. Retry with vision only as absolute last resort (after 2 fails)
    if retries >= 2:
        # Minimal vision fallback (screenshot + cheap LLM if you have one)
        screenshot = await page.screenshot()
        # vision_call_here(screenshot, "extract login error or next step")  # <5% usage
        state["vision_calls"] = state.get("vision_calls", 0) + 1
        state["result"] = "vision_fallback"
        return {**state, "retries": retries + 1}
    
    # 4. Smart retry (new profile + proxy rotation possible here)
    state["retries"] = retries + 1
    state["result"] = "retrying"
    return state

# ====================== FULL LANGGRAPH WORKFLOW ======================
def build_google_stealth_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("stealth_auth", google_auth_node)
    
    # Conditional edges
    def should_continue(state: AgentState):
        if state.get("result") in ["login_success", "otp_success", "captcha_solved"]:
            return END
        if state["retries"] >= 4:
            return END  # hard fail after 4 tries
        return "stealth_auth"
    
    workflow.set_entry_point("stealth_auth")
    workflow.add_conditional_edges("stealth_auth", should_continue)
    
    return workflow.compile()

# ====================== USAGE EXAMPLE ======================
async def run_google_stealth_login(email: str, password: str, profile_name: str):
    context = await launch_google_stealth_context(profile_name)
    page = await context.new_page()
    
    await page.goto("https://accounts.google.com/signin", wait_until="domcontentloaded")
    
    initial_state: AgentState = {
        "page": page,
        "context": context,
        "email": email,
        "password": password,
        "action": "login",
        "result": None,
        "vision_calls": 0,
        "retries": 0,
    }
    
    graph = build_google_stealth_graph()
    final_state = await graph.ainvoke(initial_state)
    
    print(f"✅ Google stealth login result: {final_state['result']}")
    print(f"Vision calls: {final_state['vision_calls']} | Retries: {final_state['retries']}")
    
    return final_state

# Run it
# asyncio.run(run_google_stealth_login("sandeep@razam.in", "YOUR_PASS", "profile_sandeep_01"))
