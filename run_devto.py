import sys
import asyncio
import os

# Add orchestrator to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "python-orchestrator"))

from advanced_agent import run_agent

objective = """
MISSION: REAL-WORLD DEPLOYMENT - PUBLISH SEARCHWALA ARTICLE ON DEV.TO

EXECUTION STEPS:
1. CONTENT GENERATION: 
   - Draft a short, highly engaging technical post (approx 150-200 words).
   - Topic: Building "SearchWala" - a custom meta-search engine aggregating 90+ search engines. 
   - Key Highlight: Mention the recent architectural migration from Python to Rust, explaining how it drastically reduced RAM usage and improved execution speed.
   - Call to Action: Include a link urging developers to check out the code on GitHub (https://github.com/yourusername/searchwala).

2. NAVIGATION & LOGIN: 
   - Navigate directly to https://dev.to/new (or navigate to dev.to and click 'Create Post').

3. HUMAN-LIKE EXECUTION (Ghost Engine):
   - Use ghost_type to enter a catchy title (e.g., "Why I rewrote my 90+ Engine Meta-Search in Rust 🦀").
   - Use ghost_type to write the main body. Ensure human-cadence typing delays are active.
   - Use Heavy Mode (Vision + Clean DOM) if the agent gets stuck finding the editor fields.

4. PUBLISH: 
   - Use GhostCursor to locate and click the "Publish" button. 

OUTPUT: 
Execute this live run flawlessly. Do not exit until the post is live. Return the final published URL by outputting action_type="done" when finished.
"""

if __name__ == "__main__":
    print("Starting Live Enterprise Deployment Test on Dev.to...")
    asyncio.run(run_agent(objective))
