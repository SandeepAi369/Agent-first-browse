import sys
import os
import time
import asyncio

# Add orchestrator to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "python-orchestrator"))

from app.browser_promoter.worker_planner import _build_worker_llms
from langchain_core.messages import HumanMessage

async def main():
    print("Fetching API Configuration...")
    primary, secondary, deepseek, tertiary = _build_worker_llms()
    
    # Matches the failover chain from advanced_agent.py
    failover_chain = secondary + deepseek + primary + tertiary
    
    if not failover_chain:
        print("No models found in the failover chain! Check your .env file.")
        return

    print(f"Discovered {len(failover_chain)} models in the failover chain.")
    print("Pinging endpoints (Timeout: 10s)...\n")

    results = []
    
    for i, model in enumerate(failover_chain):
        model_name = getattr(model, 'model_name', getattr(model, 'model', 'Unknown'))
        
        start_time = time.time()
        status = "❌ FAILED"
        error_msg = "None"
        latency_str = "-"
        
        try:
            msg = [HumanMessage(content="System diagnostic. Reply with exactly one word: PING_OK")]
            response = await asyncio.wait_for(model.ainvoke(msg), timeout=10.0)
            
            latency = time.time() - start_time
            latency_str = f"{int(latency * 1000)}ms"
            
            if "PING_OK" in response.content:
                status = "✅ OK"
            else:
                status = "⚠️ BAD RESPONSE"
                error_msg = f"Unexpected reply: {response.content.strip()[:20]}"
                
        except asyncio.TimeoutError:
            error_msg = "HTTP 408: Timeout (>10s)"
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Rate limit" in err_str or "rate_limit" in err_str:
                error_msg = "HTTP 429: Rate Limit Exceeded"
            elif "404" in err_str or "not_found" in err_str:
                error_msg = "HTTP 404: Not Found (Model doesn't exist)"
            elif "401" in err_str or "unauthorized" in err_str.lower():
                error_msg = "HTTP 401: Unauthorized (Bad Key)"
            else:
                error_msg = err_str.split('\n')[0][:50] + "..."

        results.append([model_name, status, latency_str, error_msg])

    print("\n=== API HEALTH CHECK REPORT ===")
    print("| Model Name | Status | Latency | Exact Error (If Failed) |")
    print("| :--- | :--- | :--- | :--- |")
    for row in results:
        # Sanitize pipes in error messages
        clean_err = row[3].replace("|", "-")
        print(f"| {row[0]} | {row[1]} | {row[2]} | {clean_err} |")

if __name__ == "__main__":
    asyncio.run(main())
