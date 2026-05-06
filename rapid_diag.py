"""Rapid-fire diagnostic: ping every model in the failover chain with a 8s timeout."""
import sys, os, time, asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), "python-orchestrator"))

from dotenv import load_dotenv
load_dotenv()

import httpx

PROVIDERS = [
    {
        "name": "NVIDIA Key-1 (z-ai/glm-5.1)",
        "base_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "key": os.getenv("NVIDIA_NIM_API_KEY", ""),
        "model": "z-ai/glm-5.1",
    },
    {
        "name": "NVIDIA Key-2 (z-ai/glm-5.1)",
        "base_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "key": os.getenv("NVIDIA_NIM_API_KEYS", "").split(",")[0].strip() if os.getenv("NVIDIA_NIM_API_KEYS") else "",
        "model": "z-ai/glm-5.1",
    },
    {
        "name": "GROQ (openai/gpt-oss-120b)",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "key": os.getenv("GROQ_API_KEY", ""),
        "model": "openai/gpt-oss-120b",
    },
    {
        "name": "Cerebras (gpt-oss-120b)",
        "base_url": "https://api.cerebras.ai/v1/chat/completions",
        "key": os.getenv("CEREBRAS_API_KEY", ""),
        "model": "gpt-oss-120b",
    },
]

HEADERS_SPOOF = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

async def ping(provider):
    name = provider["name"]
    if not provider["key"]:
        return f"| {name} | ⛔ NO KEY | - | Missing API key |"
    
    headers = {
        "Authorization": f"Bearer {provider['key']}",
        "Content-Type": "application/json",
        **HEADERS_SPOOF,
    }
    body = {
        "model": provider["model"],
        "messages": [{"role": "user", "content": "Reply with one word: PING"}],
        "max_tokens": 5,
        "temperature": 0,
    }
    
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(provider["base_url"], json=body, headers=headers)
            latency = int((time.time() - start) * 1000)
            
            if resp.status_code == 200:
                data = resp.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "???")[:20]
                return f"| {name} | ✅ OK ({resp.status_code}) | {latency}ms | Reply: {reply} |"
            else:
                error_body = resp.text[:80].replace("|", "-").replace("\n", " ")
                return f"| {name} | ❌ {resp.status_code} | {latency}ms | {error_body} |"
    except httpx.TimeoutException:
        return f"| {name} | ⏱️ TIMEOUT | >8000ms | No response in 8s |"
    except Exception as e:
        return f"| {name} | ❌ ERROR | - | {str(e)[:60]} |"

async def main():
    print("=" * 80)
    print("  RAPID API DIAGNOSTIC (8s timeout per provider)")
    print("=" * 80)
    
    # Run ALL pings concurrently for speed
    tasks = [ping(p) for p in PROVIDERS]
    results = await asyncio.gather(*tasks)
    
    print("\n| Provider | Status | Latency | Details |")
    print("| :--- | :--- | :--- | :--- |")
    for r in results:
        print(r)
    
    print("\n" + "=" * 80)

asyncio.run(main())
