import sys
import asyncio
import os

# Add orchestrator to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "python-orchestrator"))

from advanced_agent import run_agent

objective = """
TARGET PLATFORM: hashnode.com
OBJECTIVE: Publish a fresh, highly technical post about "SearchWala" utilizing all newly upgraded Swarm modules.

EXECUTION STEPS:
1. MEMORY CHECK: Call CampaignMemory.has_posted_on('hashnode.com'). If false, proceed.
2. NAVIGATE & AUTH: Go to hashnode.com. Utilize existing local cookies to authenticate (Continue with Google if necessary).
3. SMART DOM: Use the _compress_dom_for_llm engine to locate the "Write", "Draft", or "New Post" button.
4. CRITIC AGENT PIPELINE: 
   - Draft a new article about SearchWala. Angle: "Why I chose Rust over Python to build a 90+ engine Meta-Search aggregator."
   - MUST pass the draft through ContentCritic. Ensure the tone score is > 0.8 and ZERO jargon (no "revolutionize", "cutting-edge", etc.) is used.
5. GHOST STEALTH: Use ghost_click() to navigate the editor interface. Use ghost_type() with human entropy to type the Title and Body.
6. PUBLISH & RECORD: Click the publish button using Bezier curves. Once the URL is live, record the success in CampaignMemory.
"""

if __name__ == "__main__":
    print("Starting Live Enterprise Deployment Test on Hashnode...")
    asyncio.run(run_agent(objective))
